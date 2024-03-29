import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer


"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True


class Criterion(torch.nn.Module):
    def __init__(self, opt):
        """
        Args:
            opt: Namespace containing all relevant parameters.
        """
        super(Criterion, self).__init__()

        ####
        self.num_proxies        = opt.n_classes
        self.embed_dim          = opt.embed_dim

        self.proxies            = torch.nn.Parameter(torch.randn(self.num_proxies, self.embed_dim)/8)
        self.class_idxs         = torch.arange(self.num_proxies)

        self.name           = 'proxynca'

        self.optim_dict_list = [{'params':self.proxies, 'lr':opt.lr * opt.loss_proxynca_lrmulti}]

        self.scale = opt.loss_proxynca_scale
        self.dim = opt.loss_proxynca_dim

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM



    def forward(self, batch, labels, **kwargs):
        #Empirically, multiplying the embeddings during the computation of the loss seem to allow for more stable training;
        #Acts as a temperature in the NCA objective.
        #This is the same as in [Calibrated neighborhood aware confidence measure]
        batch   = self.scale * torch.nn.functional.normalize(batch, dim=1)
        proxies = self.scale * torch.nn.functional.normalize(self.proxies, dim=1)
        #Group required proxies
        pos_proxies = torch.stack([proxies[pos_label:pos_label+1,:] for pos_label in labels])
        neg_proxies = torch.stack([torch.cat([self.class_idxs[:class_label],self.class_idxs[class_label+1:]]) for class_label in labels])
        neg_proxies = torch.stack([proxies[neg_labels,:] for neg_labels in neg_proxies])
        #Compute Proxy-distances
        dist_to_neg_proxies = torch.sum((batch[:,None,:]-neg_proxies).pow(2),dim=-1)
        dist_to_pos_proxies = torch.sum((batch[:,None,:]-pos_proxies).pow(2),dim=-1)
        #Compute final proxy-based NCA loss
        if self.dim == 0:
            loss = torch.mean(dist_to_pos_proxies[:,0]) + torch.mean(torch.logsumexp(-dist_to_neg_proxies, dim=0))
        elif self.dim == 1:
            loss = torch.mean(dist_to_pos_proxies[:,0] + torch.logsumexp(-dist_to_neg_proxies, dim=1))
        elif self.dim == 2:
            loss = torch.mean(dist_to_pos_proxies[:,0]) + torch.mean(torch.logsumexp(-dist_to_neg_proxies, dim=1))

        return loss

    def smat(self, A, B, mode='cosine'):
        if mode=='cosine':
            return A.mm(B.T)
        elif mode=='euclidean':
            As, Bs = A.shape, B.shape
            return (A.mm(A.T).diag().unsqueeze(-1)+B.mm(B.T).diag().unsqueeze(0)-2*A.mm(B.T)).clamp(min=1e-20).sqrt()
