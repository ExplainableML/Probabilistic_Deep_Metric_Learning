"""============================================================================================================="""
######## LIBRARIES #####################
import numpy as np
import torch.nn.functional as f
import utilities.vmf_sampler as vmf



"""============================================================================================================="""
################# ACQUIRE NUMBER OF WEIGHTS #################
def gimme_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


################# SAVE TRAINING PARAMETERS IN NICE STRING #################
def gimme_save_string(opt):
    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key],dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t'+str(sub_key)+': '+str(sub_item)
        else:
            base_str += '\n\t'+str(varx[key])
        base_str+='\n\n'
    return base_str


#############################################################################
import torch, torch.nn as nn

class DataParallel(nn.Module):
    def __init__(self, model, device_ids, dim):
        super().__init__()
        self.model    = model.model
        self.network  = nn.DataParallel(model, device_ids, dim)

    def forward(self, x):
        return self.network(x)


############## HELPER FUNCTIONS FOR VMF ###################
# def approx_log_cp(kappa, p=512):
#     '''
#     Approximates the normalizing factor of vMF using https://arxiv.org/pdf/1812.04616.pdf Appendix 8.2
#     '''
#     return torch.sqrt(p**2 / 4 + kappa**2) - (p/2 - 2) * torch.log(p/2 - 2 + torch.sqrt(p**2/4 + kappa**2))

def approx_log_cp(kappa, p=512, mode='new'):
    '''
    Approximates the normalizing factor of vMF using Taylor approximation
    '''
    if mode == 'new':
        if p==128:
            est = 127 - 0.01909 * kappa - 0.003355 * kappa**2
        else:
            est = 868 - 0.0002662 * kappa - 0.0009685 * kappa ** 2
    else:
        est = 868 - 0.0002662 * kappa - 0.0009685 * kappa ** 2
    return est

def log_ppk_vmf(mu1, kappa1, mu2, kappa2, rho=0.5, p=512):
    '''
    computes the log of the Probability Product Kernel of order rho of two p-dimensional vMFs
    given their normalized means and scales
    '''
    kappa3 = torch.linalg.norm(kappa1 * mu1 + kappa2 * mu2)
    return rho * (approx_log_cp(kappa1, p) + approx_log_cp(kappa2, p)) - approx_log_cp(rho * kappa3, p)



def log_ppk_vmf_vec(mu1, kappa1, mu2, kappa2, rho=0.5, p=512, temp=0.01, n_samples=10, mode='new'):
    # mu1: normalized vectors, kappa1: norms
    # mu2: normalized vectors, kappa2: norms
    if mu1.dim() == 1:
        mu1 = mu1.unsqueeze(0)
    if kappa1.dim() == 0:
        kappa1 = kappa1.unsqueeze(0).unsqueeze(1)
    if kappa2.dim() == 0:
        kappa2.unsqueeze(0)
    if kappa2.dim() == 1:
        kappa2.unsqueeze(1)
    mu1 = torch.nn.functional.normalize(mu1, dim=1)
    mu2 = torch.nn.functional.normalize(mu2, dim=1)

    # Draw samples (scales with batchsize, not proxysize).
    distr = vmf.VonMisesFisher(loc=mu1, scale=kappa1)
    # Sampling is the most time-consuming part.
    samples = distr.rsample(torch.Size([n_samples]))

    if mu1.shape[0] == 1:
        # We have one sample image in mu1 being compared to multiple proxies in mu2
        samples = samples.expand(n_samples, mu2.shape[0], mu1.shape[1])
    samples = samples.reshape((-1, samples.shape[2]))
    mu2 = mu2 * kappa2
    norm_mu2 = torch.norm(mu2, dim=1)
    mu2 = mu2.unsqueeze(0).expand(n_samples, -1, -1).reshape((-1, mu2.shape[1]))

    cos_sim = torch.nn.functional.cosine_similarity(samples, mu2).unsqueeze(1)
    cos_sim = cos_sim.reshape((n_samples, -1, 1))

    logcp = approx_log_cp(norm_mu2, p=p, mode=mode)
    detterm = torch.sum(torch.log(kappa2), dim=1) - torch.log(norm_mu2)

    logl = logcp.unsqueeze(0).unsqueeze(2) + norm_mu2.unsqueeze(0).unsqueeze(2) * cos_sim + detterm.unsqueeze(0).unsqueeze(2)
    logl = logl / temp

    one_over_n = torch.log(torch.ones(1) * n_samples).to(logl.device)

    logl = logl - one_over_n
    logl = torch.logsumexp(logl, dim=0)

    return logl.squeeze(logl.dim() - 1)

# def log_ppk_vmf_vec(mu1, kappa1, mu2, kappa2, rho=0.5, p=512, temp=1):
#     '''
#     same as log_ppk_vmf, but mu2 is a tensor of [n, dim] and kappa2 is [n, 1] and we compute the ppk from mu1 to each of
#     these ones
#     '''
#     if mu1.dim() == 1:
#         mu1 = mu1.unsqueeze(0)
#     cos_sim = torch.nn.functional.cosine_similarity(mu1, mu2).unsqueeze(1)
#     kappa_dist2 = (kappa1 - kappa2)**2
#     #metric_cos = torch.log(torch.ones(1).cuda()) - torch.log(1 + torch.exp(-(cos_sim - 0.992) * 500))
#     # This function is tailored to what logPPK does when it has temperature 0.01
#     # If you want a different temperature, you need to play around with the constants in the following two lines
#     # Just run visualize_distances to see if it approximately matches the logPPK with the temperature you are intending
#     metric_cos = - (cos_sim - 1)**2 * 1000 * (kappa1 + kappa2 / 2)
#     metric_kappa = -kappa_dist2 / 10 * 2**temp
#     metric = metric_cos + metric_kappa
#     return metric.squeeze(metric.dim() - 1)
