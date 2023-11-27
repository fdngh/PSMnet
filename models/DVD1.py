import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace
import torchvision
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.backbone import FEM
from util.utils import safe_norm


class TVDeeplabRes101Encoder(nn.Module):
    """
    FCN-Resnet101 backbone from torchvision deeplabv3
    No ASPP is used as we found emperically it hurts performance
    """

    def __init__(self, use_coco_init, aux_dim_keep = 64, use_aspp = False):
        super().__init__()
        _model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=use_coco_init, progress=True, num_classes=21, aux_loss=None)
        if use_coco_init:
            print("###### NETWORK: Using ms-coco initialization ######")
        else:
            print("###### NETWORK: Training from scratch ######")

        _model_list = list(_model.children())
        self.aux_dim_keep = aux_dim_keep
        self.backbone = _model_list[0]
        self.localconv = nn.Conv2d(2048, 256,kernel_size = 1, stride = 1, bias = False) # reduce feature map dimension                 
        self.crm_dot = FEM.crm(256)
        self.crm_gobal = FEM.crm(256)
        self.crm_local = FEM.crm(256)  

    def forward(self, x_in,fg_mask,low_level):
        """
        Args:
            low_level: whether returning aggregated low-level features in FCN
        """
        fts = self.backbone(x_in)
        fts2048 = fts['out']
        high_level_fts = self.localconv(fts2048)      
        return high_level_fts

    def jz(self, sup_fts, qry_fts, proto,gl_fg_proto,fg_mask):
           
        ####全局原型
        gl_fg_proto=gl_fg_proto.unsqueeze(-1).unsqueeze(-2)
        gl_fg_proto=gl_fg_proto.expand(1,-1,32,32)       
        goal_qry_fts = self.crm_gobal(qry_fts, gl_fg_proto*fg_mask)       
        ######局部原型        
        proto=proto.transpose(0,1)
        sp_proto = proto[..., None, None].repeat(1, 1, 32,32)###[256,num,32,32]
        sup_fts_feat_=sup_fts.squeeze(0)
        cos_sim_map = F.cosine_similarity(sp_proto, sup_fts_feat_.unsqueeze(1), dim=0, eps=1e-7)##num,h,w
        guide_map = cos_sim_map.max(0)[1]  ##[32,32]
        sp_guide_feat = proto[:, guide_map]  # [256,32,32]
        sp_guide_feat = sp_guide_feat.unsqueeze(0)
        local_qry_fts =  self.crm_local(goal_qry_fts,sp_guide_feat*fg_mask)
        ##像素点
        dot_qry_fts = self.crm_dot(local_qry_fts , sup_fts*fg_mask) 
        return  dot_qry_fts 


class FewShotSeg(nn.Module):
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super(FewShotSeg, self).__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.get_encoder(in_channels)
        
      
        

    def get_encoder(self, in_channels):
        # if self.config['which_model'] == 'deeplab_res101':
        if self.config['which_model'] == 'dlfcn_res101':
            use_coco_init = self.config['use_coco_init']
            self.encoder = TVDeeplabRes101Encoder(use_coco_init)
        else:
            raise NotImplementedError(f'Backbone network {self.config["which_model"]} not implemented')
        if self.pretrained_path:
            self.load_state_dict(torch.load(self.pretrained_path))
            print(f'###### Pre-trained model f{self.pretrained_path} has been loaded ######')

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, tra_wsize, isval, val_wsize, show_viz=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            show_viz: return the visualization dictionary
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)

        assert n_ways == 1, "Multi-shot has not been implemented yet"  # NOTE: actual shot in support goes in batch dimension
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]

        assert sup_bsize == qry_bsize == 1

        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad=True)
     
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'
        res_fg_msk = torch.stack([F.interpolate(fore_mask_w, size=32, mode='bilinear') for fore_mask_w in fore_mask],
                                 dim=0)
      
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts = self.encoder(imgs_concat, res_fg_msk.squeeze(0), low_level=False)

        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
            n_queries, qry_bsize, -1, *fts_size)

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        for epi in range(1):  # batch dimension, fixed to 1
            fg_masks = []  # keep the way part

            res_bg_msk = torch.stack(
                [F.interpolate(back_mask_w, size=fts_size, mode='bilinear') for back_mask_w in back_mask],
                dim=0)  # [nway, ns, nb, nh', nw']
            scores = []
            qg_raw_score, bg_raw_score = self.score(supp_fts, qry_fts, res_fg_msk, res_bg_msk, tra_wsize, isval,
                                                    val_wsize)
            scores.append(bg_raw_score)
            # .unsqueeze(0)
            scores.append(qg_raw_score)

            pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'

            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi], tra_wsize)
                align_loss += align_loss_epi
        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])

        return output, align_loss / sup_bsize

    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask, tra_wsize):

        n_ways, n_shots = len(fore_mask), len(fore_mask[0])
        # Masks for getting query prototype
        pred_mask = pred.argmax(dim=1).unsqueeze(0)  # 1 x  N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        # skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        # FIXME: fix this in future we here make a stronger assumption that a positive class must be there to avoid undersegmentation/ lazyness
        skip_ways = []
        ### added for matching dimensions to the new data format
        qry_fts = qry_fts.unsqueeze(0).unsqueeze(2)  # added to nway(1) and nb(1)
        ### end of added part
        loss = []
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                img_fts = supp_fts[way: way + 1,
                          shot: shot + 1]  # actual local query [way(1), nb(1, nb is now nshot), nc, h, w]

                qry_pred_fg_msk = F.interpolate(binary_masks[way + 1].float(), size=img_fts.shape[-2:],
                                                mode='bilinear')  # [1 (way), n (shot), h, w]

                # background
                qry_pred_bg_msk = F.interpolate(binary_masks[0].float(), size=img_fts.shape[-2:],
                                                mode='bilinear')  # 1, n, h ,w
                scores = []

                fg_raw_score, bg_raw_score = self.score(sup_x=qry_fts, qry=img_fts, sup_y=qry_pred_fg_msk.unsqueeze(-3),
                                                        sup_by=qry_pred_bg_msk.unsqueeze(-3), tra_wsize=tra_wsize)
                scores.append(bg_raw_score)
                scores.append(fg_raw_score)

                supp_pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear')

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()

                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0

                # Compute Loss
                loss.append(F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways)

        return torch.sum(torch.stack(loss))

    
    def proto(self, sup_x, sup_y):
        protos = []
        for i in range(len(sup_y)):           
            proto = torch.sum(sup_x * sup_y[i], dim=(-1, -2)) \
                    / (sup_y[i].sum(dim=(-1, -2)) + 1e-5)  # nb x C
            proto = proto.mean(dim=0, keepdim=True)
            protos.append(proto)
        return protos
    def Gproto(self, sup_x, sup_y):
        proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5)  # nb x C
        proto = proto.mean(dim=0, keepdim=True)
        return proto
    def GP(self, proto, qry):  # class-level prototype only     
        proto=torch.cat(proto,dim=0)
        proto=safe_norm(prot