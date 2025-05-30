import time
import os.path as osp
import numpy as np
import logging
import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
from tensorboardX import SummaryWriter
from collections import deque
from tqdm import tqdm
import os
class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        
        logging.basicConfig(filename=osp.join(args.save_path,'training.log'), level=logging.INFO, 
                            format='%(asctime)s - %(message)s',filemode='a')
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux
    
    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        
        # start FSL training
        label, label_aux = self.prepare_label()
        print("Before epoch")
        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()

            start_tm = time.time()
            avg_loss = 0
            num =0
            for batch in self.train_loader:
                self.train_step += 1
                file_names = None
                if self.args.model_class == 'FEAT':
                    if torch.cuda.is_available():
                        data, gt_label = [_.cuda() for _ in batch]
                    else:
                        data, gt_label = batch[0], batch[1]
                else:
                    if torch.cuda.is_available():
                        data, gt_label,file_names = batch[0], batch[1], batch[2]
                        data = data.cuda()
                        gt_label = gt_label.cuda()
                    else:
                        data, gt_label,file_names = batch[0], batch[1]
               
                data_tm = time.time()
                #print("data_load_time:",data_tm-start_tm)
                self.dt.add(data_tm - start_tm)

                logits, _ = self.para_model(data,gt_label,file_names)


                loss = F.cross_entropy(logits, label)
                total_loss = F.cross_entropy(logits, label)
                    
                tl2.add(loss)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)

                tl1.add(total_loss.item())
                ta.add(acc)
                avg_loss += loss
                num += 1
                #print("total_loss:",total_loss.item()," train acc:",acc)
                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)    

                # refresh start_tm
                start_tm = time.time()
            print("avg_loss:",avg_loss/num)
            logging.info(f"Epoch {epoch}, Train Loss: {avg_loss/num:.4f}")
            avg_loss=0
            num=0
            self.lr_scheduler.step()
            self.try_evaluate(epoch)
            #self.save_training_log()
            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        print("current lr:{}".format(self.optimizer.param_groups[0]['lr']))
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                file_names = None
                if self.args.model_class == 'FEAT':
                    if torch.cuda.is_available():
                        data, gt_label = [_.cuda() for _ in batch]
                    else:
                        data, gt_label = batch[0], batch[1]
                else:
                    if torch.cuda.is_available():
                        data, gt_label,file_names = batch[0], batch[1], batch[2]
                        data = data.cuda()
                        gt_label = gt_label.cuda()
                    else:
                        data, gt_label,file_names = batch[0], batch[1]

                logits = self.model(data,gt_label,file_names)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        #logging.info(f"Epoch {epoch}, Train Loss: {avg_loss/num:.4f}")
        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation mode
        # self.model.load_state_dict(torch.load(osp.join("./checkpoints/MiniImageNet-semfew2-Res12-05w01s15q-Pre-DIS/40_0.5_lr1e-05mul10_step_T164.0T264.0_b0.01_bsz080-NoAug", 'old_max_acc.pth'))['params'])
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        record = np.zeros((10000, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        status = True
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                file_names = None
                if self.args.model_class == 'FEAT':
                    if torch.cuda.is_available():
                        data, gt_label = [_.cuda() for _ in batch]
                    else:
                        data, gt_label = batch[0], batch[1]
                else:
                    if torch.cuda.is_available():
                        data, gt_label,file_names = batch[0], batch[1], batch[2]
                        data = data.cuda()
                        gt_label = gt_label.cuda()
                    else:
                        data, gt_label,file_names = batch[0], batch[1]
                
                logits = self.model(data,gt_label,file_names,status)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl

        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))
        logging.info('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        logging.info('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))

        return vl, va, vap
    
    def final_record(self):
        # save the best performance in a txt file
        
        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))            
    
