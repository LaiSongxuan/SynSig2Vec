# -*- coding:utf-8 -*-
import os, pickle
import numpy 
import argparse
from collections import OrderedDict

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nutils
import torch.optim as optim
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# import dataset.datasetTrainAll_SF as dataset
import dataset.datasetTrainAll_SLN as dataset
# import dataset.datasetTrainAll_SLN_stroke as dataset
from network import Sig2Vec, LabelSmoothingLoss

parser = argparse.ArgumentParser(description='Online signature verification')
parser.add_argument('--train-shot-g', type=int, default=5, 
                    help='number of genuine samples per class per training batch(default: 5)') #Genuine samples only
parser.add_argument('--train-shot-f', type=int, default=10, 
                    help='number of forgery samples per class per training batch(default: 5)')
parser.add_argument('--train-tasks', type=int, default=4, 
                    help='number of tasks per batch')
parser.add_argument('--epochs', type=int, default=200, 
                    help='number of epochs to train (default: 200)')
parser.add_argument('--workers', type=int, default=0,
                    help='number of workers for DataLoader')
parser.add_argument('--seed', type=int, default=111, 
                    help='numpy random seed (default: 111)')
parser.add_argument('--save-interval', type=int, default=25, 
                    help='how many epochs to wait before saving the model.')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

args = parser.parse_args()
n_task = args.train_tasks
n_shot_g = args.train_shot_g
n_shot_f = args.train_shot_f

numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

sigDict = pickle.load(open("../data/MCYT_dev.pkl", "rb")) #, encoding='iso-8859-1'
dset = dataset.dataset(
                sigDict=sigDict,
                slnPath="../sigma_lognormal/params/mcyt_dev_full",
                slnLevel=0.4,
                taskSize=n_task, 
                taskNumGen=n_shot_g, 
                taskNumNeg=n_shot_f,
                numSynthesis=10,
                prefix="MCYT",
            )
del sigDict

sigDict = pickle.load(open("../data/BSID_dev.pkl", "rb")) #, encoding='iso-8859-1'
dset.addDatabase(
                sigDict=sigDict, 
                slnPath="../sigma_lognormal/params/bsid_dev_full",
                prefix="BSID"
            )
del sigDict

sigDict = pickle.load(open("../data/EBio1_dev.pkl", "rb")) #, encoding='iso-8859-1'
dset.addDatabase(
                sigDict=sigDict, 
                slnPath="../sigma_lognormal/params/ebio1_dev_full",
                prefix="EBio1"
            )
del sigDict

dset.computeStats() 

### Stroke-based synthesis
# sigDict = pickle.load(open("../data/MCYT_dev_pad.pkl", "rb")) #, encoding='iso-8859-1'
# dset = dataset.dataset(
#                 sigDict=sigDict,
#                 slnPath="../sigma_lognormal/params/mcyt_dev_stroke",
#                 slnLevel=0.4,
#                 taskSize=n_task, 
#                 taskNumGen=n_shot_g, 
#                 taskNumNeg=n_shot_f,
#                 numSynthesis=10,
#                 prefix="MCYT",
#             )
# del sigDict

# sigDict = pickle.load(open("../data/BSID_dev_pad.pkl", "rb")) #, encoding='iso-8859-1'
# dset.addDatabase(
#                 sigDict=sigDict, 
#                 slnPath="../sigma_lognormal/params/bsid_dev_stroke",
#                 prefix="BSID"
#             )
# del sigDict

# sigDict = pickle.load(open("../data/EBio1_dev_pad.pkl", "rb")) #, encoding='iso-8859-1'
# dset.addDatabase(
#                 sigDict=sigDict, 
#                 slnPath="../sigma_lognormal/params/ebio1_dev_stroke",
#                 prefix="EBio1"
#             )
# del sigDict

# dset.computeStats()

### Real handwritten signatures
# sigDict = pickle.load(open("../data/MCYT_dev.pkl", "rb")) #, encoding='iso-8859-1'
# dset = dataset.dataset(
#                     sigDict=sigDict,
#                     taskSize=n_task, 
#                     taskNumGen=n_shot_g, 
#                     taskNumNeg=n_shot_f,
#                 )
# del sigDict
# sigDict = pickle.load(open("../data/BSID_dev.pkl", "rb")) #, encoding='iso-8859-1'
# dset.addDatabase(sigDict)
# del sigDict
# sigDict = pickle.load(open("../data/EBio1_dev.pkl", "rb")) #, encoding='iso-8859-1'
# dset.addDatabase(sigDict)
# del sigDict
# sigDict = pickle.load(open("../data/EBio2_dev.pkl", "rb"))
# dset.addDatabase(sigDict)
# del sigDict

sampler = dataset.batchSampler(dset, loop=False)
dataLoader = DataLoader(dset, num_workers=args.workers, batch_sampler=sampler, collate_fn=dataset.collate_fn) 

model = Sig2Vec(n_in=dset.featDim,
                n_classes=len(dset), 
                n_task=n_task,
                n_shot_g=n_shot_g,
                n_shot_f=n_shot_f,
                APAlpha=6.)
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5, nesterov=True)

def adjust_learning_rate(optim, epoch):
    """Sets the learning rate schedule"""
    lr = args.lr * 1 if epoch < 150 else args.lr * 0.1
    for param_group in optim.param_groups:
        param_group['lr'] = lr

if not os.path.exists("./models/%d"%args.seed):
    os.mkdir("./models/%d"%args.seed)

for epoch in range(args.epochs):
    train_loss = 0
    train_lossCE = 0
    train_mAP = 0
    # adjust_learning_rate(optimizer, epoch)
    for idx, batch in enumerate(dataLoader):
        sig, lens, label = batch
        mask = model.getOutputMask(lens)

        sig = Variable(torch.from_numpy(sig)).cuda()
        mask = Variable(torch.from_numpy(mask)).cuda()
        lens = Variable(torch.from_numpy(lens)).cuda()
        label = Variable(torch.from_numpy(label)).cuda()

        model.zero_grad()
        output, output2 = model(sig, mask, lens)
        lossCE = model.smoothCEloss(output2, label, eps=0.10)
        loss, mAP = model.APLoss_DLM(output) #AP loss 
        # loss = model.siameseLoss(output)
        # loss, _, loss2 = model.tripletLoss(output, margin=0.1)
        
        (loss+lossCE).backward() #

        optimizer.step()
        
        train_mAP += mAP.item()
        train_loss += loss.item()
        train_lossCE += lossCE.item()

        if (idx) % 50 == 0:
            if idx==0:
                print (epoch, idx, train_loss, train_lossCE, train_mAP)
            else:
                print (epoch, idx, train_loss / 50, train_lossCE / 50, train_mAP / 50)
            train_loss = 0
            train_lossCE = 0
            train_mAP = 0

    if epoch % args.save_interval == 0:
       torch.save(model.state_dict(), "models/%d/epoch%d"%(args.seed, epoch))

torch.save(model.state_dict(), "models/%d/epochEnd"%args.seed)
