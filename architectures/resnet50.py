"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import pretrainedmodels as ptm
import contextlib




"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars  = opt
        self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet' if not opt.not_pretrained else None)

        self.name = opt.arch

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)

        # Start with a higher norm for images
        if 'proxyvmf' in opt.loss:
            self.model.last_linear.weight.data = self.model.last_linear.weight * opt.loss_proxyvmf_init_norm_multiplier

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

        self.out_adjust = None


    def forward(self, x, warmup=False, **kwargs):
        context = torch.no_grad() if warmup else contextlib.nullcontext()
        with context:
            x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
            for layerblock in self.layer_blocks:
                x = layerblock(x)
            no_avg_feat = x
            x = self.model.avgpool(x)
            enc_out = x = x.view(x.size(0),-1)

        x = self.model.last_linear(x)

        if 'normalize' in self.pars.arch:
            x = torch.nn.functional.normalize(x, dim=-1)
        if self.out_adjust and not self.train:
            x = self.out_adjust(x)

        return x, (enc_out, no_avg_feat)
