# -*- coding:utf-8 -*-
import numpy, pdb
import os, sys, pickle 
from collections import defaultdict
from matplotlib import pyplot as plt

from slbox import synthesis_pressure
from scipy import interpolate

from .utils import *

def interpPressure(path, nfs=2, nPoints=None, interp="cubic"):
    assert isinstance(nfs, int)
    if nfs == 1 or path.shape[0] <= 1 or path.shape[0] == nPoints:
        return path
    '''Interpolate the path by to 1/(fs*nfs) Hz, fs is the base frequency.'''
    times = numpy.linspace(0, len(path)-1, num=len(path), endpoint=True)
    if nPoints is None:
        times_interp = numpy.linspace(0, len(path)-1, num=1+(len(path)-1)*nfs, endpoint=True)
    else:
        times_interp = numpy.linspace(0, len(path)-1, num=nPoints, endpoint=True)
    fp = interpolate.interp1d(times, path, kind=interp)
    path_interp = fp(times_interp)    
    return path_interp

class dataset(object):
    """docstring for dataset"""
    def __init__(self, sigDict, slnPath, slnLevel=0.4, taskSize=1, taskNumGen=5, taskNumNeg=5, numSynthesis=10, prefix="MCYT"):
        super(dataset, self).__init__()
        self.trainKeys = list(sigDict.keys())
        self.taskSize = taskSize
        self.taskNumGen = taskNumGen
        self.taskNumNeg = taskNumNeg
        self.numSynthesis = numSynthesis
        self.slnLevel = slnLevel

        # Features extracted from synthesized signatures based on the sigma lognormal parameters.
        print (">>>>> Synthesizing signatures... <<<<<")
        slnPath_G, slnPath_F = self.synthesis(sigDict, slnPath, prefix)
        print (">>>>> Done <<<<<")

        # print (">>>>> Generating GP samples... <<<<<")
        # self.gpnoise = self.gp(numSamples=2000)
        # print (">>>>> Done <<<<<")

        self.feats = []
        self.numGen = numpy.zeros(len(self.trainKeys), dtype=numpy.int32)
        self.numSlnG = numpy.zeros(len(self.trainKeys), dtype=numpy.int32)
        self.numSlnN = numpy.zeros(len(self.trainKeys), dtype=numpy.int32)
        # The dataset is user-by-user arranged in the following format: anchors, genuines, forgeries, anchors, ...
        print (">>>>> Extracting features... <<<<<")
        for idx, key in enumerate(self.trainKeys):
            sys.stdout.write(">>>>> User key: %d <<<<<\r"%key)
            sys.stdout.flush()
            featExt(sigDict[key][True], self.feats)
            featExt(slnPath_G[key], self.feats)
            featExt(slnPath_F[key], self.feats)
            self.numGen[idx] = len(sigDict[key][True])
            self.numSlnG[idx] = len(sigDict[key][True]) * numSynthesis
            self.numSlnN[idx] = len(sigDict[key][True]) * numSynthesis
        print (">>>>> Done <<<<<")
        self.featDim = self.feats[0].shape[1]

    def synthesis(self, sigDict, slnPath, prefix, cache=True):
        if os.path.exists("./cache/%s_synStrokes_%d_%.1f.pkl"%(prefix, self.numSynthesis, self.slnLevel)):
            # PATH_G, PATH_F = pickle.load(open("./cache/%s_synStrokes_%d_%.1f.pkl"%(prefix, self.numSynthesis, self.slnLevel), 'rb'), encoding='iso-8859-1')
            PATH_G, PATH_F = pickle.load(open("./cache/%s_synStrokes_%d_%.1f.pkl"%(prefix, self.numSynthesis, self.slnLevel), 'rb'))
        else:
            PATH_G = defaultdict(list)
            PATH_F = defaultdict(list)
            for key in sigDict.keys():
                for idx in range(len(sigDict[key][True])):
                    Ps = numpy.load(os.path.join(slnPath, "%d/Pmatrix_G%d_%d.npy"%(key, key, idx)), allow_pickle=True)
                    Rs = numpy.load(os.path.join(slnPath, "%d/residual_G%d_%d.npy"%(key, key, idx)), allow_pickle=True)
                    PRs = numpy.load(os.path.join(slnPath, "%d/pressure_G%d_%d.npy"%(key, key, idx)), allow_pickle=True)
                    # Generate synthesized signatures for each genuine signature
                    for i, r in enumerate(Rs):
                        if (r.shape[0]+1)//2 != PRs[i].shape[0]:
                            print ((r.shape[0]+1)//2, PRs[i].shape[0])
                            break
                    for i in range(self.numSynthesis):
                        PATH_G[key].append(synthesis_pressure.synthesis(Ps, Rs, PRs, forgery=False, dtype="float32", padPenUp=True, const=self.slnLevel))
                        PATH_F[key].append(synthesis_pressure.synthesis(Ps, Rs, PRs, forgery=True, dtype="float32", padPenUp=True, const=self.slnLevel))
                # print ("Synthesizing user %d..."%(key))
                sys.stdout.write("Synthesizing user %d...\r"%(key))
                sys.stdout.flush()
            if cache:
                pickle.dump([PATH_G, PATH_F], open("./cache/%s_synStrokes_%d_%.1f.pkl"%(prefix, self.numSynthesis, self.slnLevel), 'wb'))
        return PATH_G, PATH_F

    def addDatabase(self, sigDict, slnPath, prefix, cache=True):
        newKeys = list(sigDict.keys())
        self.trainKeys = numpy.concatenate((self.trainKeys, newKeys))
        N = len(self.feats)

        print (">>>>> Synthesizing signatures... <<<<<")
        slnPath_G, slnPath_F = self.synthesis(sigDict, slnPath, prefix)
        print (">>>>> Done <<<<<")

        numGen = numpy.zeros(len(newKeys), dtype=numpy.int32)
        numSlnG = numpy.zeros(len(newKeys), dtype=numpy.int32)
        numSlnN = numpy.zeros(len(newKeys), dtype=numpy.int32)
        print (">>>>> Extracting features... <<<<<")
        for i, key in enumerate(newKeys):
            # print (">>>>>User key:", key, "<<<<<")
            sys.stdout.write(">>>>> User key: %d <<<<<\r"%key)
            sys.stdout.flush()
            featExt(sigDict[key][True], self.feats)
            featExt(slnPath_G[key], self.feats)
            featExt(slnPath_F[key], self.feats)
            numGen[i] = len(sigDict[key][True])
            numSlnG[i] = len(sigDict[key][True]) * self.numSynthesis
            numSlnN[i] = len(sigDict[key][True]) * self.numSynthesis
        print (">>>>> Done <<<<<")
        self.numGen = numpy.concatenate((self.numGen, numGen))
        self.numSlnG = numpy.concatenate((self.numSlnG, numSlnG))
        self.numSlnN = numpy.concatenate((self.numSlnN, numSlnN))

    def computeStats(self):
        self.accumNum2 = numpy.cumsum(self.numGen + self.numSlnG + self.numSlnN) 
        self.accumNum = numpy.roll(self.accumNum2, 1); self.accumNum[0] = 0 

        self.lens = numpy.zeros(len(self.feats), dtype=numpy.float32)
        self.userLens = numpy.zeros(len(self.numGen), dtype=numpy.float32)
        for i, f in enumerate(self.feats):
            userIdx = numpy.sum(i>=self.accumNum2)
            sampleIdx = i - self.accumNum[userIdx]
            if sampleIdx < self.numGen[userIdx]:
                self.userLens[userIdx] += f.shape[0]
            self.lens[i] = f.shape[0]
        self.maxLen = int(numpy.max(self.lens))
        self.userLens = 1. / numpy.sqrt(self.userLens)
        self.userLens = self.userLens / numpy.sum(self.userLens)
        
    def __getitem__(self, index):        
        sig = self.feats[index] #[numpy.random.randint(3):]
        # sig = pathDrop(sig)
        sigLen = sig.shape[0] # sigLen = self.lens[index]
        sigLabel = numpy.sum(index>=self.accumNum2) # Note that keys start from 1, while labels start from 0. 

        return sig, sigLen, sigLabel

    def __len__(self):
        return len(self.trainKeys) 

class batchSampler(object):
    """docstring for sampler"""
    def __init__(self, dataset, loop=False):
        super(batchSampler, self).__init__()
        self.taskSize = dataset.taskSize
        self.index = numpy.arange(0, len(dataset.trainKeys), dtype=numpy.int32)
        self.taskNumGen = dataset.taskNumGen
        self.taskNumNeg = dataset.taskNumNeg
        self.numGen = dataset.numGen
        self.numSlnG = dataset.numSlnG
        self.numSlnN = dataset.numSlnN
        self.numSynthesis = dataset.numSynthesis
        self.accumNum = dataset.accumNum
        self.accumNum2 = dataset.accumNum2
        self.numIters = len(dataset)
        self.loop = loop
        self.prob = dataset.userLens

    def __iter__(self):
        batch = []
        numpy.random.shuffle(self.index)
        for i in range(self.numIters):
            if self.loop:
                idxs = self.index[numpy.arange(i, i+self.taskSize)%len(self.index)]
            else:
                idxs = numpy.random.choice(self.index, size=self.taskSize, replace=False) #, p=self.prob
            for idx in idxs:
                anchor = numpy.random.choice(self.numGen[idx], size=1, replace=False)
                gen = numpy.random.choice(self.numSynthesis, size=self.taskNumGen, replace=False) 
                neg = numpy.random.choice(self.numSynthesis, size=self.taskNumNeg, replace=False) 
                batch.append(anchor + self.accumNum[idx])
                batch.append(gen + (self.accumNum[idx] + self.numGen[idx] + anchor * self.numSynthesis))
                batch.append(neg + (self.accumNum[idx] + self.numGen[idx] + self.numSlnG[idx] + anchor * self.numSynthesis))
                # rand = numpy.random.choice(self.accumNum2[-1]-self.accumNum2[idx]+self.accumNum[idx], size=self.taskNumNeg//2, replace=False)
                # rand = (rand + self.accumNum2[idx]) % self.accumNum2[-1]
                # batch.append(rand)
            batch = numpy.concatenate(batch, axis=0).astype(numpy.int32)
            yield batch
            batch = []

    def __len__(self):
        return self.numIters

def collate_fn(batch):
    ''' `batch` is a list of tuple where 1-st element is the signature, 2-nd element is its length and 3-rd element is the label.
    '''
    batchSize = len(batch)
    sig = [item[0] for item in batch]
    sigLen = numpy.array([item[1] for item in batch], dtype=numpy.float32)
    sigLabel = numpy.array([item[2] for item in batch], dtype=numpy.int64)
    maxLen = int(numpy.max(sigLen))

    # print (sigLabel)
    # plt.plot(sig[0][:, -2], sig[0][:, -1], color="k") #marker="o", markersize=3
    # for i in range(1, 6):
    #     plt.plot(sig[i][:, -2] + (i - 1) * 2, sig[i][:, -1] + 2.0, color="k") #marker="o", markersize=3
    # for i in range(6, 11):
    #     plt.plot(sig[i][:, -2] + (i - 6) * 2, sig[i][:, -1] + 4.0, color="k") #marker="o", markersize=3
    # for i in range(11, 16):
    #     plt.plot(sig[i][:, -2] + (i - 6) * 2, sig[i][:, -1] + 6.0, color="k") #marker="o", markersize=3
    # plt.show()

    sigPadded = numpy.zeros((batchSize, maxLen, sig[0].shape[1]), dtype=numpy.float32)
    for idx, s in enumerate(sig):
            sigPadded[idx,:s.shape[0]] = s

    return sigPadded, sigLen, sigLabel



