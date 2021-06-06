# -*- coding:utf-8 -*-
import os, pickle
import numpy 
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nutils
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader

import dataset.datasetTest_SF as dataset
from network import Sig2Vec

parser = argparse.ArgumentParser(description='Online signature verification')
parser.add_argument('--train-shot-g', type=int, default=5, #Training set in meta-training (meta-batch)
                    help='number of genuine samples per class per training batch(default: 5)') #Genuine samples only
parser.add_argument('--train-shot-f', type=int, default=5, #Test set in meta-training (meta-batch)
                    help='number of forgery samples per class per training batch(default: 5)')
parser.add_argument('--train-tasks', type=int, default=1, 
                    help='number of tasks per batch')
parser.add_argument('--workers', type=int, default=0,
                    help='number of workers for DataLoader')
parser.add_argument('--seed', type=int, default=111, 
                    help='numpy random seed (default: 1)')
parser.add_argument('--epoch', type=str, default="End", 
                    help='model from the i-th epoch for testing')

args = parser.parse_args()
n_task = args.train_tasks

n_shot_g = args.train_shot_g
n_shot_f = args.train_shot_f

numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

sigDict = pickle.load(open("../data/EBio1_eva_finger.pkl", "rb"), encoding="iso-8859-1")
dset = dataset.dataset(sigDict=sigDict)
sampler = dataset.batchSampler(dset)
dataLoader = DataLoader(dset, num_workers=args.workers, batch_sampler=sampler, collate_fn=dataset.collate_fn) 

weightDict = torch.load("models/%d/epoch%s"%(args.seed, args.epoch))
model = Sig2Vec(n_in=dset.featDim,
                n_classes=weightDict.get('cls.weight').shape[0], #268+230+30+46,
                n_task=n_task,
                n_shot_g=n_shot_g,
                n_shot_f=n_shot_f)
model.load_state_dict(weightDict)
model.cuda()
model.train(mode=False)
model.eval()

feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch
    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    lens = Variable(torch.from_numpy(lens)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, _ = model(sig, mask, lens)
    output = output.data.cpu().numpy()
    feats.append(output)

feats = numpy.concatenate(feats, axis=0)
labels = numpy.kron(numpy.arange(0, len(dset.testKeys)), numpy.ones(dset.accumNumNeg)) 
inds = numpy.ones(dset.accumNumNeg); inds[dset.numGen:] = 0
inds = numpy.kron(numpy.ones(len(dset.testKeys)), inds) 

if not os.path.exists("exps/seed%d"%args.seed):
    os.mkdir("exps/seed%d"%args.seed)

if not os.path.exists("exps/seed%d/ebio1_finger"%args.seed):
    os.mkdir("exps/seed%d/ebio1_finger"%args.seed)

numpy.save("exps/seed%d/ebio1_finger/feats_epoch%s.npy"%(args.seed, args.epoch), feats)
numpy.save("exps/seed%d/ebio1_finger/labels_epoch%s.npy"%(args.seed, args.epoch), labels)
numpy.save("exps/seed%d/ebio1_finger/inds_epoch%s.npy"%(args.seed, args.epoch), inds)
# numpy.save("exps/seed%d/ebio1_finger/mean_epoch%s.npy"%(args.seed, args.epoch), model.bn.running_mean.cpu().numpy())
# numpy.save("exps/seed%d/ebio1_finger/stdvar_epoch%s.npy"%(args.seed, args.epoch), model.bn.running_var.cpu().numpy()**0.5)

sigDict = pickle.load(open("../data/EBio1_dev_finger.pkl", "rb"), encoding="iso-8859-1")
dset = dataset.dataset(sigDict=sigDict)
sampler = dataset.batchSampler(dset)
dataLoader = DataLoader(dset, num_workers=args.workers, batch_sampler=sampler, collate_fn=dataset.collate_fn) 

feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch
    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    lens = Variable(torch.from_numpy(lens)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, _ = model(sig, mask, lens)
    output = output.data.cpu().numpy()
    feats.append(output)

feats = numpy.concatenate(feats, axis=0)
labels = numpy.kron(numpy.arange(0, len(dset.testKeys)), numpy.ones(dset.accumNumNeg)) 
inds = numpy.ones(dset.accumNumNeg); inds[dset.numGen:] = 0
inds = numpy.kron(numpy.ones(len(dset.testKeys)), inds) 

if not os.path.exists("exps/training/seed%d"%args.seed):
    os.mkdir("exps/training/seed%d"%args.seed)

if not os.path.exists("exps/training/seed%d/ebio1_finger"%args.seed):
    os.mkdir("exps/training/seed%d/ebio1_finger"%args.seed)

if not os.path.exists("exps/training/seed%d/ebio1_finger"%args.seed):
    os.mkdir("exps/training/seed%d/ebio1_finger"%args.seed)

numpy.save("exps/training/seed%d/ebio1_finger/feats_epoch%s.npy"%(args.seed, args.epoch), feats)


sigDict = pickle.load(open("../data/EBio2_eva_finger.pkl", "rb"), encoding="iso-8859-1")
dset = dataset.dataset(sigDict=sigDict)
sampler = dataset.batchSampler(dset)
dataLoader = DataLoader(dset, num_workers=args.workers, batch_sampler=sampler, collate_fn=dataset.collate_fn) 

feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch
    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    lens = Variable(torch.from_numpy(lens)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, _ = model(sig, mask, lens)
    output = output.data.cpu().numpy()
    feats.append(output)

feats = numpy.concatenate(feats, axis=0)
labels = numpy.kron(numpy.arange(0, len(dset.testKeys)), numpy.ones(dset.accumNumNeg)) 
inds = numpy.ones(dset.accumNumNeg); inds[dset.numGen:] = 0
inds = numpy.kron(numpy.ones(len(dset.testKeys)), inds) 

if not os.path.exists("exps/seed%d/ebio2_finger"%args.seed):
    os.mkdir("exps/seed%d/ebio2_finger"%args.seed)

numpy.save("exps/seed%d/ebio2_finger/feats_epoch%s.npy"%(args.seed, args.epoch), feats)
numpy.save("exps/seed%d/ebio2_finger/labels_epoch%s.npy"%(args.seed, args.epoch), labels)
numpy.save("exps/seed%d/ebio2_finger/inds_epoch%s.npy"%(args.seed, args.epoch), inds)

sigDict = pickle.load(open("../data/EBio2_dev_finger.pkl", "rb"), encoding="iso-8859-1")
dset = dataset.dataset(sigDict=sigDict)
sampler = dataset.batchSampler(dset)
dataLoader = DataLoader(dset, num_workers=args.workers, batch_sampler=sampler, collate_fn=dataset.collate_fn) 

feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch
    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    lens = Variable(torch.from_numpy(lens)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, _ = model(sig, mask, lens)
    output = output.data.cpu().numpy()
    feats.append(output)

feats = numpy.concatenate(feats, axis=0)
labels = numpy.kron(numpy.arange(0, len(dset.testKeys)), numpy.ones(dset.accumNumNeg)) 
inds = numpy.ones(dset.accumNumNeg); inds[dset.numGen:] = 0
inds = numpy.kron(numpy.ones(len(dset.testKeys)), inds) 

if not os.path.exists("exps/training/seed%d"%args.seed):
    os.mkdir("exps/training/seed%d"%args.seed)

if not os.path.exists("exps/training/seed%d/ebio2_finger"%args.seed):
    os.mkdir("exps/training/seed%d/ebio2_finger"%args.seed)

if not os.path.exists("exps/training/seed%d/ebio2_finger"%args.seed):
    os.mkdir("exps/training/seed%d/ebio2_finger"%args.seed)

numpy.save("exps/training/seed%d/ebio2_finger/feats_epoch%s.npy"%(args.seed, args.epoch), feats)
