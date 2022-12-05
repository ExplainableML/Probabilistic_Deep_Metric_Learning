### Standard DML criteria
from criteria import triplet, margin, proxynca, npair, oproxy, proxyvmf_panc
from criteria import proxynca_l2, proxynca_cos, proxyvmf_kl, proxyvmf, proxynca_nivmf
from criteria import lifted, contrastive, softmax
from criteria import angular, snr, histogram, arcface
from criteria import softtriplet, multisimilarity, quadruplet
### Non-Standard Criteria
from criteria import adversarial_separation
### Basic Libs
import copy


"""================================================================================================="""
def select(loss, opt, to_optim, batchminer=None, embeds=None, targets=None):
    #####
    losses = {'triplet': triplet,
              'margin':margin,
              'proxynca':proxynca,
              'proxyvmf_panc':proxyvmf_panc,
              "proxynca_l2":proxynca_l2,
              "proxynca_cos":proxynca_cos,
              "proxynca_kl":proxyvmf_kl,
              "proxyvmf":proxyvmf,
              "proxynca_nivmf":proxynca_nivmf,
              'oproxy':oproxy,
              'npair':npair,
              'angular':angular,
              'contrastive':contrastive,
              'lifted':lifted,
              'snr':snr,
              'multisimilarity':multisimilarity,
              'histogram':histogram,
              'softmax':softmax,
              'softtriplet':softtriplet,
              'arcface':arcface,
              'quadruplet':quadruplet,
              'adversarial_separation':adversarial_separation}


    if loss not in losses: raise NotImplementedError('Loss {} not implemented!'.format(loss))

    loss_lib = losses[loss]
    if loss_lib.REQUIRES_BATCHMINER:
        if batchminer is None:
            raise Exception('Loss {} requires one of the following batch mining methods: {}'.format(loss, loss_lib.ALLOWED_MINING_OPS))
        else:
            if batchminer.name not in loss_lib.ALLOWED_MINING_OPS:
                raise Exception('{}-mining not allowed for {}-loss!'.format(batchminer.name, loss))


    loss_par_dict  = {'opt':opt}
    if loss_lib.REQUIRES_BATCHMINER:
        loss_par_dict['batchminer'] = batchminer

    if "proxyvmf" in opt.loss and opt.loss_proxyvmf_warmstart:
        loss_par_dict['embeds'] = embeds
        loss_par_dict['targets'] = targets

    criterion = loss_lib.Criterion(**loss_par_dict)

    if loss_lib.REQUIRES_OPTIM:
        if hasattr(criterion,'optim_dict_list') and criterion.optim_dict_list is not None:
            to_optim += criterion.optim_dict_list
        else:
            to_optim    += [{'params':criterion.parameters(), 'lr':criterion.lr}]

    return criterion, to_optim
