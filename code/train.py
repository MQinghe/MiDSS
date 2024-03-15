import argparse
import logging
import os
import random
import shutil
import sys
from typing import Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from networks.unet_model import UNet
from dataloaders.dataloader import FundusSegmentation, ProstateSegmentation, MNMSSegmentation
import dataloaders.custom_transforms as tr
from utils import losses, metrics, ramps, util
from torch.cuda.amp import autocast, GradScaler
import contextlib
import matplotlib.pyplot as plt 

from torch.optim.lr_scheduler import LambdaLR
import math

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='prostate', choices=['fundus', 'prostate'])
parser.add_argument("--save_name", type=str, default="debug", help="experiment_name")
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument("--max_iterations", type=int, default=60000, help="maximum epoch number to train")
parser.add_argument('--num_eval_iter', type=int, default=500)
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.03, help="segmentation network learning rate")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument('--load',action='store_true')
parser.add_argument('--load_path',type=str,default='../model/lb1_ratio0.2/iter_6000.pth')
parser.add_argument("--threshold", type=float, default=0.95, help="confidence threshold for using pseudo-labels",)

parser.add_argument('--amp', type=int, default=1, help='use mixed precision training or not')

parser.add_argument("--label_bs", type=int, default=4, help="labeled_batch_size per gpu")
parser.add_argument("--unlabel_bs", type=int, default=4)
parser.add_argument("--test_bs", type=int, default=1)
parser.add_argument('--domain_num', type=int, default=6)
parser.add_argument('--lb_domain', type=int, default=1)
parser.add_argument('--lb_num', type=int, default=40)
# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency_type", type=str, default="mse", help="consistency_type")
parser.add_argument("--consistency", type=float, default=1.0, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")

parser.add_argument('--depth', type=int, default=28)
parser.add_argument('--widen_factor', type=int, default=2)
parser.add_argument('--leaky_slope', type=float, default=0.1)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument("--cutmix_prob", default=1.0, type=float)
parser.add_argument("--LB", default=0.01, type=float)
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def cycle(iterable: Iterable):
    """Make an iterator returning elements from the iterable.

    .. note::
        **DO NOT** use `itertools.cycle` on `DataLoader(shuffle=True)`.\n
        Because `itertools.cycle` saves a copy of each element, batches are shuffled only at the first epoch. \n
        See https://docs.python.org/3/library/itertools.html#itertools.cycle for more details.
    """
    while True:
        for x in iterable:
            yield x

def get_SGD(net, name='SGD', lr=0.1, momentum=0.9, \
                  weight_decay=5e-4, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    optim = getattr(torch.optim, name)
    
    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)
    
    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]
    
    optimizer = optim(per_param_args, lr=lr,
                    momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer
        
        
def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''
    
    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''
        
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr
    
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def extract_amp_spectrum(img_np):
    # trg_img is of dimention CxHxW (C = 3 for RGB image and 1 for slice)
    
    fft = np.fft.fft2( img_np, axes=(-2, -1) )
    amp_np, pha_np = np.abs(fft), np.angle(fft)

    return amp_np

def low_freq_mutate_np( amp_src, amp_trg, L=0.1, degree=1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    # print (b)
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    # ratio = random.randint(1,10)/10
    ratio = random.uniform(0, degree)

    a_src[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2] * (1-ratio) + a_trg[:,h1:h2,w1:w2] * ratio
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def source_to_target_freq( src_img, amp_trg, L=0.1, degree=1 ):
    # exchange magnitude
    # input: src_img, trg_img
    src_img = src_img #.transpose((2, 0, 1))
    src_img_np = src_img #.cpu().numpy()
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L, degree=degree)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg #.transpose(1, 2, 0)


if args.dataset == 'fundus':
    part = ['cup', 'disc']
    dataset = FundusSegmentation
elif args.dataset == 'prostate':
    part = ['base'] 
    dataset = ProstateSegmentation
n_part = len(part)
dice_calcu = {'fundus':metrics.dice_coeff_2label, 'prostate':metrics.dice_coeff}

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size).cuda()
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

@torch.no_grad()
def test(args, model, test_dataloader, epoch, writer, ema=True):
    model.eval()
    model_name = 'ema' if ema else 'stu'
    val_loss = 0.0
    val_dice = [0.0] * n_part
    domain_metrics = []
    domain_num = len(test_dataloader)
    if args.dataset == 'fundus':
        ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        softmax, sigmoid, multi = False, True, True
    elif args.dataset == 'prostate':
        ce_loss = CrossEntropyLoss(reduction='none')
        softmax, sigmoid, multi = True, False, False
    dice_loss = losses.DiceLossWithMask(2)
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        dc = -1
        domain_val_loss = 0.0
        domain_val_dice = [0.0] * n_part
        for batch_num,sample in enumerate(cur_dataloader):
            dc = sample['dc'][0].item()
            data = sample['image'].cuda()
            mask = sample['label'].cuda()
            if args.dataset == 'fundus':
                cup_mask = mask.eq(0).float()
                disc_mask = mask.le(128).float()
                mask = torch.cat((cup_mask.unsqueeze(1), disc_mask.unsqueeze(1)),dim=1)
            elif args.dataset == 'prostate':
                mask = mask.eq(0).long()
            output = model(data)
            loss_seg = ce_loss(output, mask).mean() + \
                        dice_loss(output, mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)

            
            if args.dataset == 'fundus':
                dice = dice_calcu[args.dataset](np.asarray(torch.sigmoid(output.cpu()))>=0.5,mask.clone().cpu())
            elif args.dataset == 'prostate':
                dice = dice_calcu[args.dataset](np.asarray(torch.max(torch.softmax(output.cpu(),dim=1), dim=1)[1]),mask.clone().cpu())
            
            domain_val_loss += loss_seg.item()
            for i in range(len(domain_val_dice)):
                domain_val_dice[i] += dice[i]

        domain_val_loss /= len(cur_dataloader)
        val_loss += domain_val_loss
        writer.add_scalar('{}_val/domain{}/loss'.format(model_name, dc), domain_val_loss, epoch)
        for i in range(len(domain_val_dice)):
            domain_val_dice[i] /= len(cur_dataloader)
            val_dice[i] += domain_val_dice[i]
        for n, p in enumerate(part):
            writer.add_scalar('{}_val/domain{}/val_{}_dice'.format(model_name, dc, p), domain_val_dice[n], epoch)
        text = 'domain%d epoch %d : loss : %f' % (dc, epoch, domain_val_loss)
        for n, p in enumerate(part):
            text += ' val_%s_dice: %f' % (p, domain_val_dice[n])
            if n != n_part-1:
                text += ','
        logging.info(text)
        
    model.train()
    val_loss /= domain_num
    writer.add_scalar('{}_val/loss'.format(model_name), val_loss, epoch)
    for i in range(len(val_dice)):
        val_dice[i] /= domain_num
    for n, p in enumerate(part):
        writer.add_scalar('{}_val/val_{}_dice'.format(model_name, p), val_dice[n], epoch)
    text = 'epoch %d : loss : %f' % (epoch, val_loss)
    for n, p in enumerate(part):
        text += ' val_%s_dice: %f' % (p, val_dice[n])
        if n != n_part-1:
            text += ','
    logging.info(text)
    return val_dice

def train(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    base_lr = args.base_lr

    if args.dataset == 'fundus':
        num_channels = 3
        patch_size = 256
        num_classes = 2
        args.label_bs = 4
        args.unlabel_bs = 4
        min_v, max_v = 0.5, 1.5
        fillcolor = 255
        args.max_iterations = 30000
        if args.domain_num >=4:
            args.domain_num = 4
    elif args.dataset == 'prostate':
        num_channels = 1
        patch_size = 384
        num_classes = 2
        args.label_bs = 4
        args.unlabel_bs = 4
        min_v, max_v = 0.1, 2
        fillcolor = 255
        args.max_iterations = 60000
        if args.domain_num >= 6:
            args.domain_num = 6

    max_iterations = args.max_iterations
    weak = transforms.Compose([tr.RandomScaleCrop(patch_size),
            tr.RandomScaleRotate(fillcolor=fillcolor),
            tr.RandomHorizontalFlip(),
            tr.elastic_transform(),
            ])
    
    strong = transforms.Compose([
            tr.Brightness(min_v, max_v),
            tr.Contrast(min_v, max_v),
            tr.GaussianBlur(kernel_size=int(0.1 * patch_size), num_channels=num_channels),
    ])

    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain_num = args.domain_num
    domain = list(range(1,domain_num+1))
    if args.dataset == 'fundus':
        domain_len = [50, 99, 320, 320]
    elif args.dataset == 'prostate':
        domain_len = [225, 305, 136, 373, 338, 133]
    lb_domain = args.lb_domain
    data_num = domain_len[lb_domain-1]
    lb_num = args.lb_num
    lb_idxs = list(range(lb_num))
    unlabeled_idxs = list(range(lb_num, data_num))
    test_dataset = []
    test_dataloader = []
    lb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=[lb_domain], 
                                                selected_idxs = lb_idxs, weak_transform=weak,normal_toTensor=normal_toTensor)
    ulb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=domain, 
                                                selected_idxs=unlabeled_idxs, weak_transform=weak, strong_tranform=strong,normal_toTensor=normal_toTensor)
    for i in range(1, domain_num+1):
        cur_dataset = dataset(base_dir=train_data_path, phase='test', splitid=-1, domain=[i], normal_toTensor=normal_toTensor)
        test_dataset.append(cur_dataset)
    lb_dataloader = cycle(DataLoader(lb_dataset, batch_size = args.label_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True))
    ulb_dataloader = cycle(DataLoader(ulb_dataset, batch_size = args.unlabel_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True))
    for i in range(0,domain_num):
        cur_dataloader = DataLoader(test_dataset[i], batch_size = args.test_bs, shuffle=False, num_workers=0, pin_memory=True)
        test_dataloader.append(cur_dataloader)

    def create_model(ema=False):
        # Network definition
        if args.model == 'unet':
            model = UNet(n_channels = num_channels, n_classes = num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model.cuda()

    model = create_model()
    ema_model = create_model(ema=True)

    iter_num = 0
    start_epoch = 0

    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # set to train
    if args.dataset == 'fundus':
        ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        softmax, sigmoid, multi = False, True, True
    elif args.dataset == 'prostate':
        ce_loss = CrossEntropyLoss(reduction='none')
        softmax, sigmoid, multi = True, False, False
    dice_loss = losses.DiceLossWithMask(num_classes)

    logging.info("{} iterations per epoch".format(args.num_eval_iter))

    max_epoch = max_iterations // args.num_eval_iter
    best_dice = [0.0] * n_part
    best_dice_iter = [-1] * n_part
    best_avg_dice = 0.0
    best_avg_dice_iter = -1
    dice_of_best_avg = [0.0] * n_part
    stu_best_dice = [0.0] * n_part
    stu_best_dice_iter = [-1] *n_part
    stu_best_avg_dice = 0.0
    stu_best_avg_dice_iter = -1
    stu_dice_of_best_avg = [0.0] * n_part

    iter_num = int(iter_num)
    threshold = args.threshold

    if args.load:
        path_str = '../model/prostate/dm{}/checkpoint.pth'.format(args.lb_domain)
        start_epoch, ema_model, model, optimizer, best_avg_dice, best_avg_dice_iter, stu_best_avg_dice, stu_best_avg_dice_iter = util.load_osmancheckpoint(
            path_str, ema_model, model, optimizer
        )
        iter_num = start_epoch*args.num_eval_iter
        logging.info('Models restored from epoch {}'.format(start_epoch))


    scaler = GradScaler()
    amp_cm = autocast if args.amp else contextlib.nullcontext

    for epoch_num in range(start_epoch, max_epoch):
        model.train()
        ema_model.train()
        p_bar = tqdm(range(args.num_eval_iter))
        p_bar.set_description(f'No. {epoch_num+1}')
        for i_batch in range(1, args.num_eval_iter+1):
            lb_sample = next(lb_dataloader)
            ulb_sample = next(ulb_dataloader)
            lb_x_w, lb_y = lb_sample['image'], lb_sample['label']
            ulb_x_w, ulb_x_s, ulb_y = ulb_sample['image'], ulb_sample['strong_aug'], ulb_sample['label']
            lb_dc, ulb_dc = lb_sample['dc'].cuda(), ulb_sample['dc'].cuda()
            
            move_transx = []
            for i in range(len(lb_x_w)):
                amp_trg = extract_amp_spectrum((ulb_x_w[i].numpy()+1)*127.5)
                img_freq = source_to_target_freq(((lb_x_w[i]+1)*127.5).numpy(), amp_trg, L=args.LB, degree=iter_num/max_iterations)
                img_freq = np.clip(img_freq, 0, 255).astype(np.float32)
                move_transx.append(img_freq)
            move_transx = torch.tensor(np.array(move_transx), dtype=torch.float32)
            move_transx = move_transx/127.5 -1
            move_transx = move_transx.cuda()

            lb_x_w, lb_y, ulb_x_w, ulb_x_s, ulb_y = lb_x_w.cuda(), lb_y.cuda(), ulb_x_w.cuda(), ulb_x_s.cuda(), ulb_y.cuda()
            if args.dataset == 'fundus':
                lb_cup_label = lb_y.eq(0).float() # == 0
                lb_disc_label = lb_y.le(128).float()  # <= 128
                lb_mask = torch.cat((lb_cup_label.unsqueeze(1), lb_disc_label.unsqueeze(1)),dim=1)
                ulb_cup_label = ulb_y.eq(0).float()
                ulb_disc_label = ulb_y.le(128).float()
                ulb_mask = torch.cat((ulb_cup_label.unsqueeze(1), ulb_disc_label.unsqueeze(1)),dim=1)
            elif args.dataset == 'prostate':
                lb_mask = lb_y.eq(0).long()
                ulb_mask = ulb_y.eq(0).long()

            with amp_cm():
                with torch.no_grad():
                    label_box = torch.stack([obtain_cutmix_box(img_size=patch_size, p=args.cutmix_prob) for i in range(len(ulb_x_s))], dim=0)
                    img_box = label_box.unsqueeze(1)
                    if args.dataset == 'fundus':
                        label_box = label_box.unsqueeze(1)
                    logits_ulb_x_w = ema_model(ulb_x_w)
                    ulb_x_w_ul = ulb_x_w * (1-img_box) + lb_x_w * img_box
                    logits_w_ul = ema_model(ulb_x_w_ul)
                    ulb_x_w_lu = lb_x_w * (1-img_box) + ulb_x_w * img_box
                    logits_w_lu = ema_model(ulb_x_w_lu)
                    if args.dataset == 'fundus':
                        prob = logits_ulb_x_w.sigmoid()
                        pseudo_label = prob.ge(0.5).float()
                        mask = prob.ge(threshold).float() + prob.le(1-threshold).float()
                        prob_w_ul = logits_w_ul.sigmoid()
                        pseudo_label_w_ul = prob_w_ul.ge(0.5).float()
                        mask_w_ul = prob_w_ul.ge(threshold).float() + prob_w_ul.le(1-threshold).float()
                        prob_w_lu = logits_w_lu.sigmoid()
                        pseudo_label_w_lu = prob_w_lu.ge(0.5).float()
                        mask_w_lu = prob_w_lu.ge(threshold).float() + prob_w_lu.le(1-threshold).float()
                    elif args.dataset == 'prostate':
                        prob_ulb_x_w = torch.softmax(logits_ulb_x_w, dim=1)
                        prob, pseudo_label = torch.max(prob_ulb_x_w, dim=1)
                        mask = (prob > threshold).unsqueeze(1).float()
                        prob_w_ul = torch.softmax(logits_w_ul, dim=1)
                        conf_w_ul, pseudo_label_w_ul = torch.max(prob_w_ul, dim=1)
                        mask_w_ul = (conf_w_ul > threshold).unsqueeze(1).float()
                        prob_w_lu = torch.softmax(logits_w_lu, dim=1)
                        conf_w_lu, pseudo_label_w_lu = torch.max(prob_w_lu, dim=1)
                        mask_w_lu = (conf_w_lu > threshold).unsqueeze(1).float()

                    mask_w = mask_w_ul * (1-img_box) + mask_w_lu * img_box
                    pseudo_label_w = (pseudo_label_w_ul * (1-label_box) + pseudo_label_w_lu * label_box).long()
                    if args.dataset == 'fundus':
                        pseudo_label_w = pseudo_label_w.float()
                        ensemble = (pseudo_label_w == pseudo_label).float() * mask
                    elif args.dataset == 'prostate':
                        ensemble = (pseudo_label_w == pseudo_label).unsqueeze(1).float() * mask
                    mask_w[ensemble == 0] = 0

                mask_ul, mask_lu = mask.clone(), mask.clone()
                ulb_x_s_ul = ulb_x_s * (1-img_box) + move_transx * img_box
                pseudo_label_ul = (pseudo_label * (1-label_box) + lb_mask * label_box).long()
                mask_ul[img_box.expand(mask_ul.shape) == 1] = 1
                ulb_x_s_lu = move_transx * (1-img_box) + ulb_x_s * img_box
                pseudo_label_lu = (lb_mask * (1-label_box) + pseudo_label * label_box).long()
                if args.dataset == 'fundus':
                    pseudo_label_ul = pseudo_label_ul.float()
                    pseudo_label_lu = pseudo_label_lu.float()
                mask_lu[img_box.expand(mask_lu.shape) == 0] = 1
                # outputs for model
                logits_lb_x_w = model(lb_x_w)
                logits_ulb_x_s_ul = model(ulb_x_s_ul)
                logits_ulb_x_s_lu = model(ulb_x_s_lu)
                logits_ulb_x_s = model(ulb_x_s)

                sup_loss = ce_loss(logits_lb_x_w, lb_mask).mean() + \
                            dice_loss(logits_lb_x_w, lb_mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)
                
                consistency_weight = get_current_consistency_weight(
                    iter_num // (args.max_iterations/args.consistency_rampup))

                unsup_loss_ul = (ce_loss(logits_ulb_x_s_ul, pseudo_label_ul) * mask_ul.squeeze(1)).mean() + \
                                dice_loss(logits_ulb_x_s_ul, pseudo_label_ul.unsqueeze(1), mask=mask_ul, softmax=softmax, sigmoid=sigmoid, multi=multi)

                unsup_loss_lu = (ce_loss(logits_ulb_x_s_lu, pseudo_label_lu) * mask_lu.squeeze(1)).mean() + \
                                dice_loss(logits_ulb_x_s_lu, pseudo_label_lu.unsqueeze(1), mask=mask_lu, softmax=softmax, sigmoid=sigmoid, multi=multi)

                unsup_loss_s = (ce_loss(logits_ulb_x_s, pseudo_label_w) * mask_w.squeeze(1)).mean() + \
                                dice_loss(logits_ulb_x_s, pseudo_label_w.unsqueeze(1), mask=mask_w, softmax=softmax, sigmoid=sigmoid, multi=multi)

                loss = sup_loss + consistency_weight * (unsup_loss_ul + unsup_loss_lu + consistency_weight * unsup_loss_s)

            optimizer.zero_grad()

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # update ema model
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('train/mask', mask.mean(), iter_num)
            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/loss', loss.item(), iter_num)
            writer.add_scalar('train/sup_loss', sup_loss.item(), iter_num)
            writer.add_scalar('train/unsup_loss_ul', unsup_loss_ul.item(), iter_num)
            writer.add_scalar('train/unsup_loss_lu', unsup_loss_lu.item(), iter_num)
            writer.add_scalar('train/unsup_loss_s', unsup_loss_s.item(), iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/bi_consistency_weight', consistency_weight**2, iter_num)
            if p_bar is not None:
                p_bar.update()

            if args.dataset == 'fundus':
                p_bar.set_description('iteration %d: loss:%.4f,sup_loss:%.4f, unsup_loss_ul:%f, unsup_loss_lu:%f, cons_w:%.4f,mask_ratio:%.4f' 
                                        % (iter_num, loss.item(), sup_loss.item(), unsup_loss_ul.item(), unsup_loss_lu.item(), consistency_weight, mask.mean()))
            elif args.dataset == 'prostate':
                p_bar.set_description('iteration %d : loss:%.3f, sup_loss:%.3f, unsup_loss_ul:%.3f, unsup_loss_lu:%.3f, unsup_loss_s:%.3f, cons_w:%.3f, mask_ratio:%.3f' 
                                    % (iter_num, loss.item(), sup_loss.item(), unsup_loss_ul.item(), unsup_loss_lu.item(), unsup_loss_s.item(), consistency_weight, mask.mean()))

        if p_bar is not None:
            p_bar.close()


        logging.info('test ema model')
        text = ''
        val_dice = test(args, ema_model, test_dataloader, epoch_num+1, writer)
        for n, p in enumerate(part):
            if val_dice[n] > best_dice[n]:
                best_dice[n] = val_dice[n]
                best_dice_iter[n] = iter_num
            text += 'val_%s_best_dice: %f at %d iter' % (p, best_dice[n], best_dice_iter[n])
            text += ', '
        if sum(val_dice) / len(val_dice) > best_avg_dice:
            best_avg_dice = sum(val_dice) / len(val_dice)
            best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                dice_of_best_avg[n] = val_dice[n]
        text += 'val_best_avg_dice: %f at %d iter' % (best_avg_dice, best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, dice_of_best_avg[n])
        logging.info(text)
        logging.info('test stu model')
        stu_val_dice = test(args, model, test_dataloader, epoch_num+1, writer, ema=False)
        text = ''
        for n, p in enumerate(part):
            if stu_val_dice[n] > stu_best_dice[n]:
                stu_best_dice[n] = stu_val_dice[n]
                stu_best_dice_iter[n] = iter_num
            text += 'stu_val_%s_best_dice: %f at %d iter' % (p, stu_best_dice[n], stu_best_dice_iter[n])
            text += ', '
        if sum(stu_val_dice) / len(stu_val_dice) > stu_best_avg_dice:
            stu_best_avg_dice = sum(stu_val_dice) / len(stu_val_dice)
            stu_best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                stu_dice_of_best_avg[n] = stu_val_dice[n]
            save_text = "{}_avg_dice_best_model.pth".format(args.model)
            save_best = os.path.join(snapshot_path, save_text)
            logging.info('save cur best avg model to {}'.format(save_best))
            torch.save(model.state_dict(), save_best)
        text += 'val_best_avg_dice: %f at %d iter' % (stu_best_avg_dice, stu_best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, stu_dice_of_best_avg[n])
        logging.info(text)
        text = 'checkpoint.pth'
        checkpoint_path = os.path.join(snapshot_path, text)
        util.save_osmancheckpoint(epoch_num+1, ema_model, model, optimizer, best_avg_dice, best_avg_dice_iter, stu_best_avg_dice, stu_best_avg_dice_iter, checkpoint_path)
        logging.info('save checkpoint to {}'.format(checkpoint_path))

        
    writer.close()


if __name__ == "__main__":
    snapshot_path = "../model/" + args.dataset + "/" + args.save_name + "/"
    if args.dataset == 'fundus':
        train_data_path='../../data/Fundus'
    elif args.dataset == 'prostate':
        train_data_path="../../data/ProstateSlice"
    elif args.dataset == 'MNMS':
        train_data_path="../../data/MNMS/mnms"

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    elif not args.overwrite:
        raise Exception('file {} is exist!'.format(snapshot_path))
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cmd = " ".join(["python"] + sys.argv)
    logging.info(cmd)
    logging.info(str(args))

    train(args, snapshot_path)
