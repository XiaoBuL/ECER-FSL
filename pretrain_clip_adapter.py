import argparse
import os
import os.path as osp
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.models.classifier import Classifier
from model.dataloader.samplers import CategoriesSampler
from model.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pandas as pd
import json
import numpy as np
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# pre-train model, compute validation acc after 500 epoches
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'TieredImageNet', 'CUB','CIFAR_FS','FC100'])    
    parser.add_argument('--backbone_class', type=str, default='Res12', choices=['ConvNet', 'Res12','Visformer'])
    parser.add_argument('--schedule', type=int, nargs='+', default=[350, 400, 440, 460, 480], help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--query', type=int, default=15)    
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--exp_num', type=str, default='0')
    parser.add_argument('--semantic', type=str, default='miniimagenet_imagecaptions_all.pth')
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--use_clipfc', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--Adapter', type=str, default='Text',choices=['Text','All','NoAdapter']) 
    parser.add_argument('--multi_sem', type=bool, default=False)
    parser.add_argument('--ifres', type=bool, default=False)
    args = parser.parse_args()
    args.orig_imsize = -1
    pprint(vars(args))
    set_seed(args.seed)
    save_path1 = '-'.join([args.dataset, args.backbone_class, 'Pre-clip-exp-',str(args.exp_num)])
    save_path2 = '_'.join([str(args.lr), str(args.gamma), str(args.schedule)])
    args.save_path = osp.join(save_path1, save_path2)
    if not osp.exists(save_path1):
        os.mkdir(save_path1)
    ensure_path(args.save_path)

    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet import tieredImageNet as Dataset
    elif args.dataset == 'FC100':
        from model.dataloader.fc100 import DatasetLoader as Dataset
    elif args.dataset == 'CIFAR_FS':
        from model.dataloader.cifar_fs import DatasetLoader as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    trainset = Dataset('train', args, augment=True)
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    args.num_class = trainset.num_class
    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, 200, valset.num_class, 1 + args.query) # test on 16-way 1-shot
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    args.way = valset.num_class
    args.shot = 1
    
    # construct model
    model = Classifier(args)
    if 'Conv' in  args.backbone_class:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    elif 'Res' in args.backbone_class:
        if args.optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    elif 'Vis' in args.backbone_class:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    else:
        raise ValueError('No Such Encoder')    
    criterion = torch.nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if args.ngpu  > 1:
            model.encoder = torch.nn.DataParallel(model.encoder, device_ids=list(range(args.ngpu)))
        
        model = model.cuda()
        criterion = criterion.cuda()
    
    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
    
    def save_checkpoint(is_best, filename='checkpoint.pth.tar'):
        state = {'epoch': epoch + 1,
                 'args': args,
                 'state_dict': model.state_dict(),
                 'trlog': trlog,
                 'val_acc_dist': trlog['max_acc_dist'],
                 'val_acc_sim': trlog['max_acc_sim'],
                 'optimizer' : optimizer.state_dict(),
                 'global_count': global_count}
        
        torch.save(state, osp.join(args.save_path, filename))
        if is_best:
            shutil.copyfile(osp.join(args.save_path, filename), osp.join(args.save_path, 'model_best.pth.tar'))
    
    if args.resume == True:
        # load checkpoint
        state = torch.load(osp.join(args.save_path, 'model_best.pth.tar'))
        init_epoch = state['epoch']
        resumed_state = state['state_dict']
        # resumed_state = {'module.'+k:v for k,v in resumed_state.items()}
        model.load_state_dict(resumed_state)
        trlog = state['trlog']
        optimizer.load_state_dict(state['optimizer'])
        initial_lr = optimizer.param_groups[0]['lr']
        global_count = state['global_count']
    else:
        init_epoch = 1
        trlog = {}
        trlog['args'] = vars(args)
        trlog['train_loss'] = []
        trlog['val_loss_dist'] = []
        trlog['val_loss_sim'] = []
        trlog['train_acc'] = []
        trlog['val_acc_sim'] = []
        trlog['val_acc_dist'] = []
        trlog['max_acc_dist'] = 0.0
        trlog['max_acc_dist_epoch'] = 0
        trlog['max_acc_sim'] = 0.0
        trlog['max_acc_sim_epoch'] = 0        
        initial_lr = args.lr
        global_count = 0
        
    
    timer = Timer()
    writer = SummaryWriter(logdir=args.save_path)
    for epoch in range(init_epoch, args.max_epoch + 1):
        # refine the step-size
        if epoch in args.schedule:
            initial_lr *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr
        
        model.train()
        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, label,file_names = batch[0], batch[1], batch[2]
                data = data.cuda()
                label = label.cuda()
                label = label.type(torch.LongTensor)
            else:
                data, label = batch
                label = label.type(torch.LongTensor)    
            
            image_features,logits,out_local,text_features = model(label,data,file_names)
            image_features = image_features.to(torch.float32)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            out_local = out_local.to(torch.float32)
            out_local = out_local / out_local.norm(dim=-1, keepdim=True)

            syn_label = torch.arange(image_features.shape[0]).clone().to('cuda')
            text_features_captions,_,_ = text_features[0],text_features[1],text_features[2]
            
            # local
            HW = out_local.shape[0]
            patch_loss = 0
            for patch in out_local:
                logits_patch =  torch.matmul(patch,text_features_captions.T)
                patch_loss_1 = F.cross_entropy(logits_patch, syn_label)
                patch_loss_2 = F.cross_entropy(logits_patch.permute(1,0), syn_label)
                patch_loss_ = (patch_loss_1 + patch_loss_2)/2
                patch_loss = patch_loss + patch_loss_
            patch_loss = patch_loss / HW
            
            # global
            logits_captions = torch.matmul(image_features,text_features_captions.T)
            loss_captions_1 = F.cross_entropy(logits_captions, syn_label)
            loss_captions_2 = F.cross_entropy(logits_captions.permute(1,0), syn_label)
            loss_clip = (loss_captions_1 + loss_captions_2)/2
            
            # ce loss
            logits = logits.cuda()
            label = label.cuda()
            loss_ce = criterion(logits, label)
            
            
            # total loss
            loss = loss_clip + loss_ce + patch_loss

            acc_ce = count_acc(logits,label)
            acc_captions = count_acc(logits_captions, syn_label)
            writer.add_scalar('data/loss', float(loss), global_count)
            writer.add_scalar('data/acc', float(acc_ce), global_count)
            if (i-1) % 100 == 0:
                print('epoch {}, train {}/{}, loss={:.4f} acc_captions={:.4f} acc_CE={:.4f}'.format(epoch, i, len(train_loader), loss.item(), acc_captions,acc_ce))
            tl.add(loss.item())
            ta.add(acc_ce)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tl = tl.item()
        ta = ta.item()

        # do not do validation in first 500 epoches
        if epoch > 300 or (epoch-1) % 5 == 0:
            model.eval()
            vl_dist = Averager()
            va_dist = Averager()
            vl_sim = Averager()
            va_sim = Averager()            
            print('[Dist] best epoch {}, current best val acc={:.4f}'.format(trlog['max_acc_dist_epoch'], trlog['max_acc_dist']))
            print('[Sim] best epoch {}, current best val acc={:.4f}'.format(trlog['max_acc_sim_epoch'], trlog['max_acc_sim']))
            # test performance with Few-Shot
            label = torch.arange(valset.num_class).repeat(args.query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)        
            with torch.no_grad():
                for i, batch in tqdm(enumerate(val_loader, 1)):
                    if torch.cuda.is_available():
                        #data, _ = [_.cuda() for _ in batch]
                        data, _,_ = batch[0], batch[1], batch[2]
                        data = data.cuda()
                        #label = label.cuda()
                        #label = label.type(torch.LongTensor)
                    else:
                        data, _ = batch
                    data_shot, data_query = data[:valset.num_class], data[valset.num_class:] # 16-way test
                    logits_dist, logits_sim = model.forward_proto(data_shot, data_query, valset.num_class)
                    loss_dist = F.cross_entropy(logits_dist, label)
                    acc_dist = count_acc(logits_dist, label)
                    loss_sim = F.cross_entropy(logits_sim, label)
                    acc_sim = count_acc(logits_sim, label)                    
                    vl_dist.add(loss_dist.item())
                    va_dist.add(acc_dist)
                    vl_sim.add(loss_sim.item())
                    va_sim.add(acc_sim)                    

            vl_dist = vl_dist.item()
            va_dist = va_dist.item()
            vl_sim = vl_sim.item()
            va_sim = va_sim.item()            
            writer.add_scalar('data/val_loss_dist', float(vl_dist), epoch)
            writer.add_scalar('data/val_acc_dist', float(va_dist), epoch)     
            writer.add_scalar('data/val_loss_sim', float(vl_sim), epoch)
            writer.add_scalar('data/val_acc_sim', float(va_sim), epoch)               
            print('epoch {}, val, loss_dist={:.4f} acc_dist={:.4f} loss_sim={:.4f} acc_sim={:.4f}'.format(epoch, vl_dist, va_dist, vl_sim, va_sim))
    
            if va_dist > trlog['max_acc_dist']:
                trlog['max_acc_dist'] = va_dist
                trlog['max_acc_dist_epoch'] = epoch
                save_model('max_acc_dist')
                save_checkpoint(True)
                
            if va_sim > trlog['max_acc_sim']:
                trlog['max_acc_sim'] = va_sim
                trlog['max_acc_sim_epoch'] = epoch
                save_model('max_acc_sim')
                save_checkpoint(True)            
    
            trlog['train_loss'].append(tl)
            trlog['train_acc'].append(ta)
            trlog['val_loss_dist'].append(vl_dist)
            trlog['val_acc_dist'].append(va_dist)
            trlog['val_loss_sim'].append(vl_sim)
            trlog['val_acc_sim'].append(va_sim)            
            save_model('epoch-last')
    
            print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
    writer.close()
    
    
    import pdb
    pdb.set_trace()