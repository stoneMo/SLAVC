import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torch.distributed as dist


class MoVSL(nn.Module):
    def __init__(self, tau, dim, dropout_img, dropout_aud, momentum_img, momentum_aud, use_mom_eval, num_neg=None):
        super(MoVSL, self).__init__()
        self.tau = tau
        self.num_neg = num_neg

        # Vision model
        self.imgnet = self.build_imgnet()
        self.img_dropout = nn.Dropout(p=dropout_img)
        self.img_proj1 = nn.Conv2d(512, dim, kernel_size=(1, 1))
        self.img_proj2 = nn.Conv2d(512, dim, kernel_size=(1, 1))

        # Audio model
        self.audnet = self.build_audnet()
        self.aud_proj1 = nn.Linear(512, dim)
        self.aud_proj2 = nn.Linear(512, dim)
        self.aud_dropout = nn.Dropout(p=dropout_aud)

        # Initialize weights (except pretrained visual model)
        for net in [self.audnet, self.img_proj1, self.aud_proj1, self.img_proj2, self.aud_proj2]:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(
                        m.weight, mean=0.0, std=0.01)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    nn.init.constant_(m.bias, 0)

        # momentum vision & audio models
        self.momentum_imgnet = self.build_imgnet()
        self.momentum_img_proj1 = nn.Conv2d(512, dim, kernel_size=(1, 1))
        self.momentum_img_proj2 = nn.Conv2d(512, dim, kernel_size=(1, 1))
        self.momentum_audnet = self.build_audnet()
        self.momentum_aud_proj1 = nn.Linear(512, dim)
        self.momentum_aud_proj2 = nn.Linear(512, dim)

        self.m_img = momentum_img
        self.m_aud = momentum_aud
        self.use_mom_eval = use_mom_eval

        # initialize momentum_encoders
        self.initialize_momentum_encoder(self.imgnet, self.momentum_imgnet)
        self.initialize_momentum_encoder(self.img_proj1, self.momentum_img_proj1)
        self.initialize_momentum_encoder(self.img_proj2, self.momentum_img_proj2)
        self.initialize_momentum_encoder(self.audnet, self.momentum_audnet)
        self.initialize_momentum_encoder(self.aud_proj1, self.momentum_aud_proj1)
        self.initialize_momentum_encoder(self.aud_proj2, self.momentum_aud_proj2)

    @torch.no_grad()
    def initialize_momentum_encoder(self, base_encoder, momentum_encoder):
        for param_b, param_m in zip(base_encoder.parameters(), momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _update_momentum_encoder(self, m, base_encoder, momentum_encoder):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(base_encoder.parameters(), momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def build_imgnet(self):
        imgnet = resnet18(pretrained=True)
        imgnet.avgpool = nn.Identity()
        imgnet.fc = nn.Identity()
        return imgnet

    def build_audnet(self):
        audnet = resnet18()
        audnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        audnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        audnet.fc = nn.Identity()
        return audnet

    def forward_img_features(self, imgnet, improj1, improj2, image):
        # Image
        img = imgnet(image).unflatten(1, (512, 7, 7))
        img = self.img_dropout(img)
        img1 = improj1(img)
        img1 = nn.functional.normalize(img1, dim=1)
        img2 = improj2(img)
        img2 = nn.functional.normalize(img2, dim=1)
        return img1, img2

    def forward_aud_features(self, audnet, audproj1, audproj2, audio):
        # Audio
        aud = audnet(audio)
        aud = self.aud_dropout(aud)
        aud1 = audproj1(aud)
        aud1 = nn.functional.normalize(aud1, dim=1)
        aud2 = audproj2(aud)
        aud2 = nn.functional.normalize(aud2, dim=1)
        return aud1, aud2

    def max_xmil_loss(self, img, aud):
        B = img.shape[0]
        if img.ndim == 4 and aud.ndim == 2:
            Slogits = torch.einsum('nchw,mc->nmhw', img, aud) / self.tau
            labels = torch.arange(B).long().to(img.device)
        elif img.ndim == 5 and aud.ndim == 2:
            Slogits = torch.einsum('nmchw,nc->nmhw', img, aud) / self.tau
            labels = torch.zeros(B).long().to(img.device)
        elif img.ndim == 4 and aud.ndim == 3:
            Slogits = torch.einsum('nchw,nmc->nmhw', img, aud) / self.tau
            labels = torch.zeros(B).long().to(img.device)
        logits = Slogits.flatten(-2, -1).max(dim=-1)[0]
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(self, image, audio, mode='train'):
        # compute features
        img1, img2 = self.forward_img_features(self.imgnet, self.img_proj1, self.img_proj2, image)
        aud1, aud2 = self.forward_aud_features(self.audnet, self.aud_proj1, self.aud_proj2, audio)

        with torch.no_grad():  # no gradient
            if mode == 'train':
                self._update_momentum_encoder(self.m_img, self.imgnet, self.momentum_imgnet)  # update the vision momentum encoder
                self._update_momentum_encoder(self.m_img, self.img_proj1, self.momentum_img_proj1)  # update the vision momentum projection
                self._update_momentum_encoder(self.m_img, self.img_proj2, self.momentum_img_proj2)  # update the vision momentum projection
                self._update_momentum_encoder(self.m_aud, self.audnet, self.momentum_audnet)  # update the audio momentum encoder
                self._update_momentum_encoder(self.m_aud, self.aud_proj1, self.momentum_aud_proj1)  # update the audio momentum projection
                self._update_momentum_encoder(self.m_aud, self.aud_proj2, self.momentum_aud_proj2)  # update the audio momentum projection

            # compute momentum features as targets
            img1_trg, img2_trg = self.forward_img_features(self.momentum_imgnet, self.momentum_img_proj1, self.momentum_img_proj2, image)
            aud1_trg, aud2_trg = self.forward_aud_features(self.momentum_audnet, self.momentum_aud_proj1, self.momentum_aud_proj2, audio)

        # Compute loss
        i2a_1 = F.softmax(torch.einsum('nchw,mc->nmhw', img1, aud1_trg).flatten(-2, -1) / self.tau, dim=1)
        i2a_2 = F.softmax(torch.einsum('nchw,mc->nmhw', img2, aud2_trg).flatten(-2, -1) / self.tau, dim=2)
        i2a = torch.log((i2a_1 * i2a_2).sum(2))    # nm

        a2i_1 = F.softmax(torch.einsum('nchw,mc->nmhw', img1_trg, aud1).flatten(-2, -1) / self.tau, dim=1)
        a2i_2 = F.softmax(torch.einsum('nchw,mc->nmhw', img2_trg, aud2).flatten(-2, -1) / self.tau, dim=2)
        a2i = torch.log((a2i_1 * a2i_2).sum(2))    # nm

        B = img1.shape[0]
        labels = torch.arange(B).long().to(img1.device)
        loss = F.cross_entropy(a2i, labels) + F.cross_entropy(i2a, labels)

        # Compute avl maps
        with torch.no_grad():
            if self.use_mom_eval:
                Savl1 = torch.einsum('nchw,nc->nhw', img1_trg, aud1_trg) / self.tau
                Savl2 = torch.einsum('nchw,nc->nhw', img2_trg, aud2_trg) / self.tau
            else:
                Savl1 = torch.einsum('nchw,nc->nhw', img1, aud1) / self.tau
                Savl2 = torch.einsum('nchw,nc->nhw', img2, aud2) / self.tau
            Savl = (Savl1 + Savl2) / 2

        return loss, Savl


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: dist.all_gather has no gradient.
    """
    if not dist.is_initialized():
        return tensor
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output