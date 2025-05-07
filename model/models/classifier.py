import torch
import torch.nn as nn
import numpy as np
from model.utils import euclidean_metric
import torch.nn.functional as F
import json
import pandas as pd
workspace = "path/to/yourcode/PVSA"

class SemAlign(nn.Module):
    def __init__(self,in_features,out_features):
        super(SemAlign, self).__init__()
        self.fc1 = nn.Linear(in_features, 2048)
        self.fc2 = nn.Linear(2048, 640)
        self.fc2 = nn.Linear(640, out_features)    
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
    
        
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class Classifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            hdim = 64
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
            self.encoder = Wide_ResNet(28, 10, 0.5)
        elif args.backbone_class == 'Visformer':
            from model.networks.visformer import visformer_tiny
            self.encoder = visformer_tiny()
            hdim = 384
        elif args.backbone_class == 'SwinT':
            hdim = 768
            from model.networks.swin_transformer import SwinTransformer
            self.encoder = SwinTransformer()             
        else:
            raise ValueError('')
        self.hdim = hdim
        self.load_sem()
        self.fc_clip = nn.Linear(hdim,512)
        self.fc_local = nn.Linear(hdim,512)
        self.fc_classify = nn.Linear(hdim,args.num_class)
        if self.args.Adapter == 'Text':
            if self.args.multi_sem:
                self.SemAdapter_captions = Adapter(512)
                self.SemAdapter_class = Adapter(512)
                self.SemAdapter_entity = Adapter(512)
            else:
                self.SemAdapter = Adapter(512)
        elif self.args.Adapter == 'All':
            self.SemAdapter = Adapter(512)
            self.VisAdapter = Adapter(512)
        
    def get_semantic(self,image_features,file_names,label):
        
        file_names = np.array(file_names)
        if file_names.shape[0] ==2 :
            class_name,file_name = file_names[0,:],file_names[1,:]
        else:
            file_name = file_names

        if self.args.multi_sem:
            #idx = gt_label
            file_names = np.array(file_names)
            text_features_class = []
            text_features_captions = []
            text_features_entity = []
            for i in label:
                text_features_class.append(self.semantic[self.idx_to_name[self.to_csvidx_train[i]]].clone().detach())
                text_features_entity.append(self.semantic_opp[self.idx_to_name[self.to_csvidx_train[i]]].clone().detach())
            for i in file_names.flatten():
                text_features_captions.append(self.semantic_captions[i].clone().detach())            
                
            text_features_captions = torch.stack(text_features_captions).contiguous().view(image_features.shape[0],-1).cuda()
            text_features_captions = text_features_captions.to(torch.float32)
            text_features_captions.requires_grad = False
            text_features_captions = text_features_captions / text_features_captions.norm(dim=-1, keepdim=True)

            text_features_entity = torch.stack(text_features_entity).contiguous().view(image_features.shape[0],-1).cuda()
            text_features_entity = text_features_entity.to(torch.float32)
            text_features_entity.requires_grad = False
            text_features_entity = text_features_entity / text_features_entity.norm(dim=-1, keepdim=True)
            
            text_features_class = torch.stack(text_features_class).contiguous().view(image_features.shape[0],-1).cuda()
            text_features_class = text_features_class.to(torch.float32)
            text_features_class.requires_grad = False
            text_features_class = text_features_class / text_features_class.norm(dim=-1, keepdim=True)
            
            return text_features_captions,text_features_class,text_features_entity
        else:
            #idx = gt_label
            file_name = np.array(file_name)
            text_features_captions = []
            for i in file_name.flatten():
                text_features_captions.append(self.semantic_captions[i].clone().detach())                          
            text_features_captions = torch.stack(text_features_captions).contiguous().view(image_features.shape[0],-1).cuda()
            text_features_captions = text_features_captions.to(torch.float32)
            text_features_captions.requires_grad = False
            text_features_captions = text_features_captions / text_features_captions.norm(dim=-1, keepdim=True)
            return text_features_captions,text_features_captions,text_features_captions
            
    
    def load_sem(self):
        
        if self.args.dataset == 'MiniImageNet':

            semantic_captions = torch.load(workspace+'/semantic_class/miniimagenet_imagecaptions_all.pth')
            self.semantic_captions = {}
            for i in semantic_captions:
                self.semantic_captions[i] = semantic_captions[i].clone().detach().type(torch.float32)

            semantic_opp = torch.load(workspace+"/semantic_class/entity_fromGpt4o_pos.pth")
            #semantic_neg = torch.load(workspace+"/semantic_class/entity_fromGpt4o_neg.pth")
            self.semantic_opp ={}
            self.semantic_neg = {}
            for i in semantic_opp:
                self.semantic_opp[i] = semantic_opp[i].clone().detach().type(torch.float32)
                #self.semantic_neg[i] = semantic_neg[i].clone().detach().type(torch.float32)
            
            self.to_csvidx_train = pd.read_csv(workspace+"/model/dataloader/csv/"+self.args.dataset+"/train.csv",header=None,dtype=str).T.values.tolist()[0]
            self.to_csvidx_val = pd.read_csv(workspace+"/model/dataloader/csv/"+self.args.dataset+"/val.csv",header=None,dtype=str).T.values.tolist()[0]
            self.to_csvidx_test = pd.read_csv(workspace+"/model/dataloader/csv/"+self.args.dataset+"/test.csv",header=None,dtype=str).T.values.tolist()[0]
            self.map_dict = {
                'test':self.to_csvidx_test,
                'train':self.to_csvidx_train,
                'val':self.to_csvidx_val
            }
            
        elif self.args.dataset == 'CIFAR_FS':
            # entity 
            semantic_opp = torch.load(workspace+"/semantic/cifarfs_pos.pth")
            #semantic_neg = torch.load(workspace+"/semantic/cifarfs_neg.pth")
            self.semantic_opp ={}
            #self.semantic_neg = {}
            for i in semantic_opp:
                self.semantic_opp[i] = semantic_opp[i].clone().detach().type(torch.float32)
                #self.semantic_neg[i] = ssemantic_neg[i].clone().detach().type(torch.float32)
            # Captions 
            semantic_captions = torch.load(workspace+'/semantic/cifarfs_captions.pth')
            self.semantic_captions = {}
            for i in semantic_captions:
                self.semantic_captions[i] = semantic_captions[i].clone().detach().type(torch.float32)
        elif self.args.dataset == 'CUB':
            # Captions 
            semantic_captions = torch.load(workspace+'/semantic/cub_captions.pth')
            self.semantic_captions = {}
            for i in semantic_captions:
                self.semantic_captions[i] = semantic_captions[i].clone().detach().type(torch.float32)
        elif self.args.dataset == 'FC100':
            # Captions 
            semantic_captions = torch.load(workspace+'/semantic/fc100_captions.pth')
            self.semantic_captions = {}
            for i in semantic_captions:
                self.semantic_captions[i] = semantic_captions[i].clone().detach().type(torch.float32)
        elif self.args.dataset == 'TieredImageNet':
            # Captions 
            semantic_captions = torch.load(workspace+'/semantic/tiere_captions.pth')
            self.semantic_captions = {}
            for i in semantic_captions:
                self.semantic_captions[i] = semantic_captions[i].clone().detach().type(torch.float32)
            

    def forward(self, label,data,file_names,is_emb = False):
        out = self.encoder(data)
        
        out_clip = self.fc_clip(out)
        out_classify = self.fc_classify(out)
        
        local_shape = self.encoder.localout_forpretrain.shape
        bathc_size,channel,hw = local_shape[0],local_shape[1],local_shape[2]*local_shape[3]
        out_local = self.fc_local(self.encoder.localout_forpretrain.view(-1,self.hdim))
        out_local = out_local.view(hw,bathc_size,-1)
        
        if file_names != None:
            text_features_captions,text_features_class,text_features_entity = self.get_semantic(out,file_names,label)
        if self.args.multi_sem:
            if self.args.Adapter == 'Text':
                text_features_captions = self.SemAdapter_captions(text_features_captions)
                text_features_class = self.SemAdapter_class(text_features_class)
                text_features_entity = self.SemAdapter_entity(text_features_entity)
            elif self.args.Adapter == 'All':
                text_features_captions_adapter = self.SemAdapter(text_features_captions)
                text_features_captions = 0.8 * text_features_captions + 0.2 * text_features_captions_adapter
                out_clip_adapter = self.VisAdapter(out_clip)
                out_clip = 0.8 * out_clip + 0.2 * out_clip_adapter
        else:
            if self.args.Adapter == 'Text':
                text_features_captions = self.SemAdapter(text_features_captions)
                return out_clip,out_classify,out_local,(text_features_captions,text_features_captions,text_features_captions)

                    
        return out_clip,out_classify,out_local,(text_features_captions,text_features_class,text_features_entity)
    
    def forward_proto(self, data_shot, data_query, way = None):
        if way is None:
            way = self.args.num_class
        proto = self.encoder(data_shot)
        query = self.encoder(data_query)
        if self.args.use_clipfc:
            proto =self.fc_clip(proto)
            query = self.fc_clip(query)
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)

        
        logits_dist = euclidean_metric(query, proto)
        logits_sim = torch.mm(query, F.normalize(proto, p=2, dim=-1).t())
        return logits_dist, logits_sim
    
