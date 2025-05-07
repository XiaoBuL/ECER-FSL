import torch
import torch.nn as nn
import numpy as np

class FewShotModel2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
        elif args.backbone_class == 'Visformer':
            if args.dataset == 'TieredImageNet':
                hdim = 768
                from model.networks.visformer import visformer_small
                self.encoder = visformer_small()
            else:
                hdim = 384
                from model.networks.visformer import visformer_tiny
                self.encoder = visformer_tiny()
        elif args.backbone_class == 'Res12_ori':
            hdim = 640
            from model.networks.res12_ori import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'SwinT':
            hdim = 768
            from model.networks.swin_transformer import SwinTransformer
            self.encoder = SwinTransformer()
        else:
            raise ValueError('')

    def split_instances(self, data):
        args = self.args
        if self.training:
            return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))
        else:
            return  (torch.Tensor(np.arange(args.eval_way*args.eval_shot)).long().view(1, args.eval_shot, args.eval_way), 
                     torch.Tensor(np.arange(args.eval_way*args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query, args.eval_way))

    def forward(self, x, gt_label, file_names,status = False,get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            instance_embs = self.encoder(x)
            proto = self.get_tsneFeat(instance_embs,gt_label,file_names,status)
            return proto
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x)
            num_inst = instance_embs.shape[0]
            # split support query set for few-shot data
            support_idx, query_idx = self.split_instances(x)
            if self.training:
                logits, logits_reg = self._forward(instance_embs, support_idx, query_idx,gt_label,file_names,status)
                return logits, logits_reg
            else:
                logits = self._forward(instance_embs, support_idx, query_idx,gt_label,file_names,status)
                return logits

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')
    def get_tsneFeat(self,x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')