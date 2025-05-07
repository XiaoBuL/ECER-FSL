import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
import json
from model.models.base2 import FewShotModel2

workspace = "path/to/yourcode/PVSA"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output

    
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



class SemAlign(nn.Module):
    def __init__(self,in_features,out_features):
        super(SemAlign, self).__init__()
        self.fc1 = nn.Linear(in_features, 4096)
        self.fc2 = nn.Linear(4096, out_features)  
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class MapSemAlign(nn.Module):
    def __init__(self, in_features, out_features,args):
        super(MapSemAlign, self).__init__()
        self.args = args
        self.linear = nn.Linear(in_features, out_features)
        self.fusion = SemAlign(in_features + out_features, out_features)

    def forward(self, x, feat):
        # x: [B, in_features]
        # feat: [B, out_features, h, w]
        # kshot: x [K*N,-1] feat
        # one shot: x[5,512] feat[5,160,21,21]
        # Linear transformation and normalization
        x = x.reshape(int(self.args.shot)*int(self.args.way),-1)# 5,5,512 -> 25,512
        x_transformed = self.linear(x)  # [B, out_features]
        x_transformed = x_transformed / (x_transformed.norm(dim=-1, keepdim=True) + 1e-8)

        # Reshape feat
        B, out_features, h, w = feat.size()
        feat = feat.view(B, out_features, -1)  # [B, out_features, h*w]

        # Ensure feat vectors are normalized
        feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-8)

        # Expand x_transformed for cosine similarity calculation
        x_transformed = x_transformed.unsqueeze(2)  # [B, out_features, 1] 5,160 -> 5,160,1
        x_transformed = x_transformed.expand(-1, -1, h * w)  # [B, out_features, h*w]

        # Compute cosine similarity
        sim = F.cosine_similarity(x_transformed, feat, dim=1)  # [B, h*w]

        # Compute weighted feature
        sim = sim.unsqueeze(1)  # [B, 1, h*w]
        weighted_feat = torch.bmm(feat, sim.permute(0, 2, 1)).squeeze(2)  # [B, out_features]

        # Combine and fuse
        combined = torch.cat([x, weighted_feat], dim=1)  # [B, in_features + out_features]
        output = self.fusion(combined)  # [B, out_features]

        return output


class MultiSem_Bfusion_Adapter(FewShotModel2):
    def __init__(self, args):
        super().__init__(args)
        if args.backbone_class == 'ConvNet':
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = 640
        elif args.backbone_class == 'Res18':
            hdim = 512
        elif args.backbone_class == 'WRN':
            hdim = 640
        elif args.backbone_class == 'Visformer':
            hdim = 384
        elif args.backbone_class == 'SwinT':
            hdim = 768
            from model.networks.swin_transformer import SwinTransformer
            self.encoder = SwinTransformer()
        else:
            raise ValueError('')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # class sem
        if self.args.dataset == 'MiniImageNet':
            semantic = torch.load('./semantic_class/{}_class.pth'.format(self.args.dataset))
            self.semantic = {}
            for key in semantic:
                self.semantic[key] = semantic[key].clone().detach().type(torch.float32)
 
            # entity  sem
            semantic_opp = torch.load(workspace+"/semantic_class/entity_fromGpt4o_pos.pth")
            self.semantic_opp ={}
            for i in semantic_opp:
                self.semantic_opp[i] = semantic_opp[i].clone().detach().type(torch.float32)

            
            self.to_csvidx_train = pd.read_csv(workspace+"/model/dataloader/csv/"+self.args.dataset+"/train.csv",header=None,dtype=str).T.values.tolist()[0]
            self.to_csvidx_val = pd.read_csv(workspace+"/model/dataloader/csv/"+self.args.dataset+"/val.csv",header=None,dtype=str).T.values.tolist()[0]
            self.to_csvidx_test = pd.read_csv(workspace+"/model/dataloader/csv/"+self.args.dataset+"/test.csv",header=None,dtype=str).T.values.tolist()[0]
            self.map_dict = {
                'test':self.to_csvidx_test,
                'train':self.to_csvidx_train,
                'val':self.to_csvidx_val
            }
            
            with open('/path/to/yourdataset/miniImageNet_folder_to_name.json', 'r', encoding='utf-8') as json_file:
                self.idx_to_name = json.load(json_file)
        elif self.args.dataset == 'CIFAR_FS':
            # class_sem 
            semantic = torch.load('./semantic_class/{}_class.pth'.format(self.args.dataset))
            self.semantic = {}
            for key in semantic:
                self.semantic[key] = semantic[key].clone().detach().type(torch.float32)
            # entity 
            semantic_opp = torch.load(workspace+"/semantic_class/cifarfs_pos.pth")
            self.semantic_opp ={}
            for i in semantic_opp:
                self.semantic_opp[i] = semantic_opp[i].clone().detach().type(torch.float32)

        elif self.args.dataset == 'CUB':
            # class_sem 
            semantic = torch.load('./semantic_class/{}_class.pth'.format(self.args.dataset))
            self.semantic = {}
            for key in semantic:
                self.semantic[key] = semantic[key].clone().detach().type(torch.float32)
            # entity 
            semantic_opp = torch.load(workspace+"/semantic_class/cub_pos.pth")
            self.semantic_opp ={}
            for i in semantic_opp:
                self.semantic_opp[i] = semantic_opp[i].clone().detach().type(torch.float32)

        elif self.args.dataset == 'FC100':
            # class_sem 
            semantic = torch.load('./semantic_class/{}_class.pth'.format(self.args.dataset))
            self.semantic = {}
            for key in semantic:
                self.semantic[key] = semantic[key].clone().detach().type(torch.float32)
            # entity 
            semantic_opp = torch.load(workspace+"/semantic_class/cifarfs_pos.pth")
            self.semantic_opp ={}
            for i in semantic_opp:
                self.semantic_opp[i] = semantic_opp[i].clone().detach().type(torch.float32)

        elif self.args.dataset == 'TieredImageNet':
            # class_sem 
            semantic = torch.load('./semantic_class/{}_class.pth'.format(self.args.dataset))
            self.semantic = {}
            for key in semantic:
                self.semantic[key] = semantic[key].clone().detach().type(torch.float32)
            # entity 
            semantic_opp = torch.load(workspace+"/semantic_class/tiere_pos.pth")
            self.semantic_opp ={}
            for i in semantic_opp:
                self.semantic_opp[i] = semantic_opp[i].clone().detach().type(torch.float32)

            
        self.s_dim = 512
        
        self.SemAlign_class = SemAlign(hdim+self.s_dim,hdim).to(device)
        self.SemAlign_entity = SemAlign(hdim+self.s_dim,hdim).to(device)

        self.SemAdapter = Adapter(512)

        self.layer_semalign2 = MapSemAlign(self.s_dim, hdim//4,self.args)
        self.layer_semalign3 = MapSemAlign(self.s_dim, hdim//2,self.args)
        self.layer_semalign2_3 = MapSemAlign(hdim//4, hdim//2,self.args)

        self.layer_semalign4 = MapSemAlign(self.s_dim, hdim,self.args)
        self.layer_semalign2_4 = MapSemAlign(hdim//2, hdim,self.args)
        self.layer_semalign3_4 = MapSemAlign(hdim//2, hdim,self.args)


        self.fusion_map = nn.Sequential(
            nn.Linear(hdim*3,4096),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(4096,hdim)
        )
       
    def _forward(self, instance_embs, support_idx, query_idx, gt_label,file_names,status=False):

        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))

        
        # get sem feat
        support_sem, sem_opp = self.get_semantic(gt_label,support_idx,support,np.array(file_names),status)
        support_sem = support_sem.reshape(-1,512)
        sem_opp = sem_opp.reshape(-1,512)
        
        # class_level
        support_sem = support_sem / support_sem.norm(dim=-1,keepdim=True)        
        # entity_level
        if self.args.ifentity:
            sem_opp = sem_opp / sem_opp.norm(dim=-1,keepdim=True)
        
        # textAdapter
        support_sem = self.SemAdapter(support_sem)

        # get proto
        proto = support.reshape(int(self.args.shot)*int(self.args.way),-1)
        proto = proto / proto.norm(dim=-1,keepdim=True)

        # fusion
        layer2out = self.encoder.layer2_out[support_idx.contiguous().view(-1)].contiguous() 
        layer2out = self.layer_semalign2(sem_opp.squeeze(0).squeeze(1), layer2out).unsqueeze(0)
        
        layer3out_feat = self.encoder.layer3_out[support_idx.contiguous().view(-1)].contiguous()
        layer3out = self.layer_semalign3(sem_opp.squeeze(0).squeeze(1), layer3out_feat).unsqueeze(0)
        layer2out = self.layer_semalign2_3(layer2out.squeeze(0), layer3out_feat).unsqueeze(0)

        layer4out_feat = self.encoder.layer4_out[support_idx.contiguous().view(-1)].contiguous()
        layer4out = self.layer_semalign4(sem_opp.squeeze(0).squeeze(1), layer4out_feat).unsqueeze(0)

        layer2out = self.layer_semalign2_4(layer2out.squeeze(0), layer4out_feat).unsqueeze(0)
        layer3out = self.layer_semalign3_4(layer3out.squeeze(0), layer4out_feat).unsqueeze(0)
        
        proto = proto + self.fusion_map(torch.cat((layer2out, layer3out, layer4out),-1)).reshape(int(self.args.shot)*int(self.args.way),-1)
        
        
        # fusion_class
        fusion_class = torch.cat((proto,support_sem),-1)
        fusion_class = self.SemAlign_class(fusion_class)
        # fusion_entity
        if self.args.ifentity:
            fusion_entity = torch.cat((proto,sem_opp),-1)
            fusion_entity = self.SemAlign_entity(fusion_entity)

       
        
        if self.args.ifentity:
            fusion_all = (fusion_class+fusion_entity).contiguous().view(support.shape[0],support.shape[1],support.shape[2],-1).cuda()
            proto = 0.2 * support.mean(dim=1) + 0.8 * fusion_all.mean(dim=1)
        else:
            proto = 0.2 * support.mean(dim=1) + 0.8 * fusion_class
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])
    
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
               
        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else:
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)

        if self.training:
            return logits, -1
        else:
            return logits 
    
    def get_semantic(self,gt_label,support_idx,support,file_names,status,fsl=True):
        if fsl:
            if self.args.dataset == 'MiniImageNet':
                dataset = ''
                if status:
                    dataset = 'test'
                elif self.training:
                    dataset = 'train'
                else:
                    dataset = 'val'
                idx = gt_label[support_idx][0]
                support_sem = []
                sem_opp = []
                to_csvidx = self.map_dict[dataset] 
                for i in idx:
                    for j in i:
                        support_sem.append(self.semantic[self.idx_to_name[to_csvidx[j]]])
                        sem_opp.append(self.semantic_opp[self.idx_to_name[to_csvidx[j]]])

                support_sem = torch.stack(support_sem).contiguous().view(support.shape[0],support.shape[1],support.shape[2],-1).cuda()
                sem_opp = torch.stack(sem_opp).contiguous().view(support.shape[0],support.shape[1],support.shape[2],-1).cuda()
                
                return support_sem,sem_opp 
            else:
                class_name,file_name = file_names[0,:],file_names[1:,]
                support_sem = []
                sem_opp = []

                for name in class_name[support_idx].flatten():
                    support_sem.append(self.semantic[name])
                    if self.args.ifentity:
                        if self.args.dataset == 'TieredImageNet':
                            entity_name = file_name[0][support_idx].flatten()[np.where(class_name == name)[0][0]][:9]
                            sem_opp.append(self.semantic_opp[entity_name])
                        else:
                            sem_opp.append(self.semantic_opp[name])
                support_sem = torch.stack(support_sem).contiguous().view(support.shape[0],support.shape[1],support.shape[2],-1).cuda()
                if self.args.ifentity:
                    sem_opp = torch.stack(sem_opp).contiguous().view(support.shape[0],support.shape[1],support.shape[2],-1).cuda()
               
                return support_sem,sem_opp
                
            

        


