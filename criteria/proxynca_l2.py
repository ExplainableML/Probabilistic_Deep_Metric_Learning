import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from utilities.misc import log_ppk_vmf_vec


"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True


class Criterion(torch.nn.Module):
    def __init__(self, opt, embeds=None, targets=None):
        """
        Args:
            opt: Namespace containing all relevant parameters.
        """
        super(Criterion, self).__init__()

        ####
        self.pars = opt
        self.num_proxies = opt.n_classes
        self.embed_dim = opt.embed_dim
        self.rho = opt.loss_proxyvmf_rho
        self.n_samples = opt.loss_proxyvmf_n_samples

        self.proxies = torch.nn.Linear(in_features=self.embed_dim, out_features=self.num_proxies, bias=False)
        if embeds is None:
            nn.init.xavier_normal_(self.proxies.weight, gain=opt.loss_proxyvmf_concentration)
        else:
            inits = self.init_proxies(embeds, targets)
            inits = torch.nn.functional.normalize(inits, dim=1) * opt.loss_proxyvmf_concentration
            self.proxies.weight.data = inits
        self.proxies = nn.utils.weight_norm(self.proxies, dim=0, name="weight")
        self.class_idxs = torch.arange(self.num_proxies).to(opt.device)

        self.temp = torch.nn.Parameter(torch.ones(1) * opt.loss_proxyvmf_temp,
                                       requires_grad=opt.loss_proxyvmf_learnable_temp)

        self.name = 'proxynca_l2'

        self.optim_dict_list = [
            {'params': self.proxies.weight_v, 'lr': opt.lr * opt.loss_proxyvmf_proxylrmulti},
            {'params': self.proxies.weight_g, 'lr': opt.lr * opt.loss_proxyvmf_conclrmulti},
            {'params': self.temp, 'lr': opt.lr * opt.loss_proxyvmf_templrmulti}
        ]

        self.subsample_p = opt.loss_proxyvmf_subsample_p

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def init_proxies(self, embeds, targets):
        with torch.no_grad():
            proxies = torch.zeros((self.num_proxies, self.embed_dim))
            for l in torch.arange(self.num_proxies):
                avg = torch.mean(embeds[targets == l, :], dim=0)
                proxies[l, :] = avg

        return proxies

    def forward(self, batch, labels, **kwargs):
        # In general, _v stands for the direction and _g for the norm (scale)
        # Note: This DOES NOT norm the proxies; it's only used as sanity check the weight_v part. weight_g stays intact.
        if self.subsample_p < 1:
            n_proxies = len(self.proxies.weight_v)
            unique_labels = np.unique(labels)
            p = self.subsample_p - len(unique_labels) / n_proxies
            neg_proxy_idxs = list(set(range(n_proxies)) - set(unique_labels))
            neg_proxy_idxs = np.random.choice(neg_proxy_idxs, int(p * len(neg_proxy_idxs)), replace=False)
            proxy_normed = torch.cat([self.proxies.weight_v[unique_labels], self.proxies.weight_v[neg_proxy_idxs]],
                                     dim=0)
            kappa_pos = torch.maximum(self.proxies.weight_g, torch.ones(1).to(batch.device) * 0.1)
            kappa_pos = torch.cat([kappa_pos[unique_labels], kappa_pos[neg_proxy_idxs]], dim=0)
            idx_map = []
            for lab in labels:
                idx_map.append(np.where(unique_labels == lab.numpy())[0][0])
        else:
            proxy_normed = torch.nn.functional.normalize(self.proxies.weight_v, dim=-1)
            kappa_pos = torch.maximum(self.proxies.weight_g, torch.ones(1).to(batch.device) * 0.1)

        # Split vectors into norm and direction, just like for vMFs, to make it fair
        batch_v = torch.nn.functional.normalize(batch, dim=1)
        batch_g = torch.norm(batch, dim=1)

        # Compute L2 similarities
        sim_to_proxies = torch.stack([
            self.log_sim(
                mu1=batch_v[i, :],
                kappa1=batch_g[i],
                mu2=proxy_normed,
                kappa2=kappa_pos,
                rho=self.rho,
                temp=self.temp,
                p=batch.shape[-1]
            )
            for i in torch.arange(batch_v.shape[0])
        ])

        # Compute final proxy-based NCA loss
        # Remove unnecessary log(exp()) and (-1) * (-1) terms from the equation
        if self.pars.loss_proxyvmf_bindev:
            same_labels = labels.to(batch.device).unsqueeze(1) == self.class_idxs.unsqueeze(0)
            diff_labels = ~same_labels
            pos_sim = self.masked_bindev_logsumexp(
                -self.pars.loss_proxyvmf_bindev_alpha * (sim_to_proxies - self.pars.loss_proxyvmf_bindev_delta),
                mask=same_labels.type(torch.bool),
                dim=self.pars.loss_proxyvmf_bindev_dim
            )
            neg_sim = self.masked_bindev_logsumexp(
                self.pars.loss_proxyvmf_bindev_alpha * (sim_to_proxies - self.pars.loss_proxyvmf_bindev_delta),
                mask=diff_labels.type(torch.bool),
                dim=self.pars.loss_proxyvmf_bindev_dim
            )
            loss = pos_sim.mean() + neg_sim.mean()
        else:
            if self.subsample_p < 1:
                loss = torch.mean(-sim_to_proxies[range(len(batch)), idx_map] + torch.logsumexp(sim_to_proxies, dim=1))
            else:
                loss = torch.mean(-sim_to_proxies[range(len(batch)), labels] + torch.logsumexp(sim_to_proxies, dim=1))


        return loss

    def log_sim(self, mu1, kappa1, mu2, kappa2, rho=0.5, p=512, temp=0.01):
        if mu1.dim() == 1:
            mu1 = mu1.unsqueeze(0)
        mu1 = F.normalize(mu1, dim=1)
        mu2 = F.normalize(mu2, dim=1)

        sim = -torch.sum((mu1 * kappa1 - mu2 * kappa2)**2, dim=1)
        sim = sim / temp
        return sim.squeeze(sim.dim() - 1)

    def masked_bindev_logsumexp(self, sims, dim=0, mask=None):
        # Adapted from https://github.com/KevinMusgrave/pytorch-metric-learning/\
        # blob/master/src/pytorch_metric_learning/utils/loss_and_miner_utils.py.
        if mask is not None:
            sims = sims.masked_fill(~mask, torch.finfo(sims.dtype).min)
        dims = list(sims.shape)
        dims[dim] = 1
        zeros = torch.zeros(dims, dtype=sims.dtype, device=sims.device)
        sims = torch.cat([sims, zeros], dim=dim)
        logsumexp_sims = torch.logsumexp(sims, dim=dim, keepdim=True)
        if mask is not None:
            logsumexp_sims = logsumexp_sims.masked_fill(~torch.any(mask, dim=dim, keepdim=True), 0)
        return logsumexp_sims
