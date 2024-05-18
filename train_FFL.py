# Change the dataset path
import bifpnc
import new_network

DATASET_PATH ='~/data'
import torch.nn.functional as F
import argparse
import json
import time
from datetime import datetime
import warnings
import os
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import random
from logger import SummaryLogger
import utils
import our_network


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

torch.backends.cudnn.allow_tf32 = False



parser = argparse.ArgumentParser(description='Quantization finetuning for CIFAR100')
parser.add_argument('--text', default='log.txt', type=str)
parser.add_argument('--exp_name', default='cifar100/FFL_res32', type=str)
parser.add_argument('--log_time', default='1', type=str)
parser.add_argument('--lr', default='0.1', type=float)
parser.add_argument('--resume_epoch', default='0', type=int)
parser.add_argument('--epoch', default='300', type=int)
parser.add_argument('--decay_epoch', default=[150, 225], nargs="*", type=int)
parser.add_argument('--w_decay', default='1e-4', type=float)
parser.add_argument('--cu_num', default='0', type=str)
parser.add_argument('--seed', default='1', type=str)
parser.add_argument('--load_pretrained', default='models/ResNet82.pth', type=str)
parser.add_argument('--save_model', default='ckpt.t7', type=str)
parser.add_argument('--n', type=int, default=20, help='Model depth.')
parser.add_argument('--consistency_rampup', '--consistency_rampup', default=80, type=float,
                    metavar='consistency_rampup', help='consistency_rampup ratio')
parser.add_argument('--num_channels', default=256, type=int)
parser.add_argument('--num_features', default=-1, type=int)
parser.add_argument('--alpha', default=1, type=float)
parser.add_argument('--beta', default=0.0, type=float)
parser.add_argument('--repeat', default=1, type=int)
parser.add_argument('--depth', default=2, type=int)
parser.add_argument('--width', default=2, type=int)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parser.parse_args()
print(args)


# MaualSeed ####################
torch.manual_seed(int(args.seed))

#### random Seed #####
# random.seed(random.randint(1, 10000))
# torch.manual_seed(args.manualSeed)
#####################


os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num

trainloader, valloader, testloader = utils.get_cifar10_dataloaders(64, 50)
num_classes = 10

#Other parameters
DEVICE = torch.device("cuda")
RESUME_EPOCH = args.resume_epoch
DECAY_EPOCH = args.decay_epoch
DECAY_EPOCH = [ep - RESUME_EPOCH for ep in DECAY_EPOCH]
FINAL_EPOCH = args.epoch
EXPERIMENT_NAME = args.exp_name
W_DECAY = args.w_decay
base_lr = args.lr
args.depth = [args.depth] * 3
def AT_loss( student, teacher):
    s_attention = F.normalize(student.pow(2).mean(1).view(student.size(0), -1))
    with torch.no_grad():
        t_attention = F.normalize(teacher.pow(2).mean(1).view(teacher.size(0), -1))
    return (s_attention - t_attention).pow(2).mean()

model = new_network.cifar_resnet(num_classes=num_classes,
            depth=args.n,)

# if len(args.load_pretrained) > 2 :
#     path = args.load_pretrained
#     state = torch.load(path)
#     utils.load_checkpoint(model, state)


# According to CIFAR
module1 = new_network.Fusion_module(2048, num_classes, 4)
module2 = new_network.Fusion_module(4096, num_classes, 4)
model.to(DEVICE)
module1.to(DEVICE)
module2.to(DEVICE)
if args.num_features == -1:
    args.num_features = len(model.network_channels)
args.network_channels = model.network_channels[-args.num_features:]
bifpn = bifpnc.BiFPNc(args.network_channels, num_classes, args)
bifpn.to(DEVICE)
# Loss and Optimizer
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=W_DECAY, nesterov=True)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
criterion_CE = nn.CrossEntropyLoss()
criterion_kl = utils.KLLoss().cuda()

optimizer_FM1 = optim.SGD(module1.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
scheduler_FM1 = optim.lr_scheduler.MultiStepLR(optimizer_FM1, milestones=DECAY_EPOCH, gamma=0.1)
optimizer_FM2 = optim.SGD(module2.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
scheduler_FM2 = optim.lr_scheduler.MultiStepLR(optimizer_FM2, milestones=DECAY_EPOCH, gamma=0.1)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return utils.sigmoid_rampup(epoch, args.consistency_rampup)

def eval(net,module1,module2, args,bifpn,test_flag=False):
    loader = valloader if not test_flag else testloader
    flag = 'Val.' if not test_flag else 'Test'

    epoch_start_time = time.time()
    net.eval()
    module1.eval()
    module2.eval()
    # top1 = AverageMeter()
    val_loss = 0

    correct_sub1 = 0
    correct_sub2 = 0
    correct_fused = 0

    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.cuda(), targets.cuda()
        at=utils.att(args,bifpn)
        feats1, feats2, output1, output2, fmap, fuse = model(inputs, False)
        feats1 = feats1[-args.num_features:]
        feats2 = feats2[-args.num_features:]
        bi_feats1, bi_outputs1 = bifpn(feats1, False)
        bi_feats2, bi_outputs2 = bifpn(feats2, False)
        ensemble_logit = (output1 + output2) / 2
        fused_logit = module1(fmap[0], fmap[1])
        bi_ensemble_logit = (bi_outputs1 + bi_outputs2) / 2
        # avgpool = nn.AvgPool2d(4)
        # bi_feats1[-1] = avgpool(bi_feats1[-1])
        # bi_feats1[-1] = bi_feats1[-1].view(bi_feats1[-1].size(0), -1)
        # bi_feats2[-1] = avgpool(bi_feats2[-1])
        # bi_feats2[-1] = bi_feats2[-1].view(bi_feats2[-1].size(0), -1)
        bi_fused_logit = module2(bi_feats1[-1], bi_feats2[-1])
        at_loss1 = 0.5 * (AT_loss(fuse[0], fuse[2]) + AT_loss(fuse[1], fuse[2]))
        at_loss2 = 0.5 * (AT_loss(fuse[3], fuse[5]) + AT_loss(fuse[4], fuse[5]))
        at_loss3 = 0.5 * (AT_loss(fuse[6], fuse[8]) + AT_loss(fuse[7], fuse[8]))
        at_loss4 = 0.5 * (AT_loss(fuse[9], fuse[11]) + AT_loss(fuse[10], fuse[11]))

        loss_model = criterion_CE(output1, targets) + criterion_CE(output2, targets) + criterion_CE(fused_logit,
                                                                                                    targets)

        loss_bifpn = criterion_CE(bi_outputs1, targets) + criterion_CE(bi_outputs2, targets) + criterion_CE(
            bi_fused_logit, targets)
        loss_model += at(output1, bi_outputs1, feats1, bi_feats2) + at(output2, bi_outputs2,feats2,bi_feats2)
        loss_biToFFL_fus = criterion_kl(bi_fused_logit, bi_ensemble_logit) + criterion_kl(fused_logit, bi_fused_logit) + criterion_kl(fused_logit,
                                                                                                        ensemble_logit) + criterion_kl(
            output1, fused_logit, ) + criterion_kl(output2, fused_logit)

        loss = loss_model + loss_bifpn + loss_biToFFL_fus + at_loss1 + at_loss2 + at_loss3 + at_loss4
        val_loss += loss.item()

        _, predicted_sub1 = torch.max(output1.data, 1)
        _, predicted_sub2 = torch.max(output2.data, 1)
        _, predicted_fused = torch.max(fused_logit.data, 1)

        total += targets.size(0)

        correct_sub1 += predicted_sub1.eq(targets.data).cpu().sum().float().item()
        correct_sub2 += predicted_sub2.eq(targets.data).cpu().sum().float().item()
        correct_fused += predicted_fused.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('%s \t Time Taken: %.2f sec' % (flag,time.time() - epoch_start_time))
    print('Loss: %.3f | Acc sub-1: %.3f%% | Acc sub-2: %.3f%% | Acc fused: %.3f%% |' % (
        val_loss / (b_idx + 1), 100. * correct_sub1 / total, 100. * correct_sub2 / total,
        100. * correct_fused / total))

    return val_loss / (b_idx + 1), correct_sub1 / total, correct_sub2 / total, correct_fused / total

def train(model,module1,module2, epoch,bifpn,args):
    epoch_start_time = time.time()
    print('\n EPOCH: %d' % epoch)
    model.train()
    module1.train()
    module2.train()

    train_loss = 0
    correct_sub1 = 0
    correct_sub2 = 0
    correct_fused = 0

    total = 0

    global optimizer
    global optimizer_FM1
    global optimizer_FM2
    at = utils.att(args, bifpn)

    consistency_weight = get_current_consistency_weight(epoch)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        optimizer_FM1.zero_grad()
        optimizer_FM2.zero_grad()
        #at=utils.att(args,bifpn)
        ###################################################################################
        feats1, feats2, output1, output2, fmap, fuse = model(x=inputs,preact=False)
        feats1 = feats1[-args.num_features:]
        feats2 = feats2[-args.num_features:]
        bi_feats1, bi_outputs1 = bifpn(feats1,preact=False)
        bi_feats2, bi_outputs2 = bifpn(feats2,preact=False)
        ensemble_logit = (output1 + output2) / 2
        fused_logit = module1(fmap[0], fmap[1])
        bi_ensemble_logit = (bi_outputs1 + bi_outputs2) / 2

        # avgpool=nn.AvgPool2d(4)
        # bi_feats1[-1] = avgpool(bi_feats1[-1])
        # bi_feats1[-1] = bi_feats1[-1].view(bi_feats1[-1].size(0), -1)
        # bi_feats2[-1] = avgpool(bi_feats2[-1])
        # bi_feats2[-1] = bi_feats2[-1].view(bi_feats2[-1].size(0), -1)
        bi_fused_logit = module2(bi_feats1[-1], bi_feats2[-1])

        at_loss1 = 0.5 * (AT_loss(fuse[0], fuse[2]) + AT_loss(fuse[1], fuse[2]))
        at_loss2 = 0.5 * (AT_loss(fuse[3], fuse[5]) + AT_loss(fuse[4], fuse[5]))
        at_loss3 = 0.5 * (AT_loss(fuse[6], fuse[8]) + AT_loss(fuse[7], fuse[8]))
        at_loss4 = 0.5 * (AT_loss(fuse[9], fuse[11]) + AT_loss(fuse[10], fuse[11]))

        loss_model = criterion_CE(output1, targets) + criterion_CE(output2, targets) + criterion_CE(fused_logit,targets)

        loss_bifpn = criterion_CE(bi_outputs1, targets) + criterion_CE(bi_outputs2, targets) + criterion_CE(bi_fused_logit, targets)
        loss_model += at(output1, bi_outputs1, feats1, bi_feats2) + at(output2, bi_outputs2,feats2,bi_feats2)
        loss_biToFFL_fus = consistency_weight*(criterion_kl(bi_fused_logit, bi_ensemble_logit) + criterion_kl(fused_logit, bi_fused_logit) + criterion_kl(fused_logit,
                                                                                                        ensemble_logit) + criterion_kl(
            output1, fused_logit, ) + criterion_kl(output2, fused_logit))

        loss = loss_model + loss_bifpn + loss_biToFFL_fus + at_loss1 + at_loss2 + at_loss3 + at_loss4

        loss.backward()
        optimizer.step()
        optimizer_FM1.step()
        optimizer_FM2.step()
        train_loss += loss.item()

        _, predicted_sub1 = torch.max(output1.data, 1)
        _, predicted_sub2 = torch.max(output2.data, 1)
        _, predicted_fused = torch.max(fused_logit.data, 1)

        total += targets.size(0)

        correct_sub1 += predicted_sub1.eq(targets.data).cpu().sum().float().item()
        correct_sub2 += predicted_sub2.eq(targets.data).cpu().sum().float().item()
        correct_fused += predicted_fused.eq(targets.data).cpu().sum().float().item()

        b_idx = batch_idx

    # batch_size = targets.size(0)
    # losses.update(loss.item(), batch_size)
    # acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
    # top1.update(acc1, batch_size)


    print('Train s1 \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc sub-1: %.3f%% | Acc sub-2: %.3f%% | Acc fused: %.3f%% |' % (
        train_loss / (b_idx + 1), 100. * correct_sub1 / total, 100. * correct_sub2 / total, 100. * correct_fused / total))

    return train_loss / (b_idx + 1), correct_fused / total


if __name__ == '__main__':
    time_log = datetime.now().strftime('%m-%d %H-%M')
    if int(args.log_time) :
        folder_name = 'FFL_{}'.format(time_log)


    path = os.path.join(EXPERIMENT_NAME, folder_name)
    if not os.path.exists('ckpt/' + path):
        os.makedirs('ckpt/' + path)
    if not os.path.exists('logs/' + path):
        os.makedirs('logs/' + path)

    # Save argparse arguments as logging
    with open('logs/{}/commandline_args.txt'.format(path), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # Instantiate logger
    logger = SummaryLogger(path)
    best_acc_s = 0
    best_acc_f = 0


    for epoch in range(RESUME_EPOCH, FINAL_EPOCH+1):
        f = open(os.path.join("logs/" + path, 'log.txt'), "a")

        ### Train ###
        train_loss, acc = train(model,module1,module2 ,epoch,bifpn,args)
        scheduler.step()
        scheduler_FM1.step()
        scheduler_FM2.step()

        ### Evaluate FFL  ###
        val_loss, accuracy_sub1, accuracy_sub2, accuracy_fused  = eval(model,module1,module2,args,bifpn, test_flag=True)
        best_acc_f = max(best_acc_f, accuracy_fused)
        best_acc_s = max(best_acc_s, accuracy_sub1, accuracy_sub2)
        best_acc_f = max(best_acc_f, accuracy_fused)
        best_acc_s = max(best_acc_s, accuracy_sub1, accuracy_sub2)
        print('the best submodel is %.2f %%' % (best_acc_s * 100))
        print('the best fuse model is %.2f %%' % (best_acc_f * 100))

        utils.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, True, 'ckpt/' + path, filename='Model_{}.pth'.format(epoch))

        utils.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': module1.state_dict(),
                    'optimizer' : optimizer_FM1.state_dict(),
                }, True, 'ckpt/' + path, filename='Module_{}.pth'.format(epoch))

        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': module2.state_dict(),
            'optimizer': optimizer_FM2.state_dict(),
        }, True, 'ckpt/' + path, filename='Module_{}.pth'.format(epoch))

        f.write('EPOCH {epoch} \t'
                'ACC_sub-1 : {acc_sub1:.4f} \t ACC_sub-2 : {acc_sub2:.4f}\t' 
                'ACC_fused : {acc_fused:.4f} \t \n'.format(
                    epoch=epoch, acc_sub1=accuracy_sub1, acc_sub2=accuracy_sub2, acc_fused=accuracy_fused)
                )
        f.close()

