#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy 
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

numpy.set_printoptions(threshold=1e6)
numpy.random.seed(13)

def getEER(FAR, FRR):
    a = FRR <= FAR
    s = numpy.sum(a)
    a[-s-1] = 1
    a[-s+1:] = 0
    FRR = FRR[a]
    FAR = FAR[a] 
    a = [[FRR[1]-FRR[0], FAR[0]-FAR[1]], [-1, 1]]
    b = [(FRR[1]-FRR[0])*FAR[0]-(FAR[1]-FAR[0])*FRR[0], 0]
    return numpy.linalg.solve(a, b)

def scoreScatter(gen, forg):
    ax = plt.subplot()
    ax.scatter(gen[:,2],gen[:,1], color='r')
    ax.scatter(forg[:,2],forg[:,1], color='k', marker="*")
    # ax.set_xlim((0.0, 2.5))
    # ax.set_ylim((0.0, 2.5))
    ax.set_xlabel("dmin")
    ax.set_ylabel("dmean")
    ax.grid("on")
    k = (numpy.sum(gen[:,1] / gen[:,0]) + numpy.sum(forg[:,1] / forg[:,0])) / (gen.shape[0] + forg.shape[0])
    x = numpy.linspace(0, 0.3, 1000)  
    y = -x / k + 0.27
    plt.plot(x, y, 'k')
    plt.title("DISTANCE")
    plt.show()

def selectTemplate(distMatrix):
    refNum = distMatrix.shape[0]
    if refNum == 1:
        return None, 1, 1, 1, 1, 1
    # distMatrix = distMatrix + distMatrix.transpose()
    '''index of the template signature'''
    idx = numpy.argmin(numpy.sum(distMatrix, axis=1) / (refNum - 1))
    dvar = numpy.sqrt((numpy.sum(distMatrix**2) / refNum / (refNum - 1)- (numpy.sum(distMatrix) / refNum / (refNum - 1))**2))
    '''pair-wise distance'''
    dmean = numpy.sum(distMatrix) / refNum / (refNum - 1) 
    '''distance of reference signatures to the template signature dtmp'''
    dtmp = numpy.sum(distMatrix[:, idx]) / (refNum - 1)
    '''distance of reference signatures to their farthest neighbor dmax'''
    dmax = numpy.mean(numpy.max(distMatrix, axis=1))
    '''distance of reference signatures to their nearest neighbor dmin'''
    distMatrix[range(refNum), range(refNum)] = float("inf")
    distMatrix[distMatrix==0] = float("inf")
    dmin = numpy.mean(numpy.min(distMatrix, axis=1))
    
    return idx, dtmp**0.5, dmax**0.5, dmin**0.5, dmean**0.5, dvar

'''Settings for the verifier. Note that in the preprocessing stage, the 
signatrues are sorted according to the filenames, therefore we directly 
index the samples using integers correspondting to the sorted results.
Different subsets may have slightly different indexing integers.'''
epoch = "End"
N_train = 1 # 1 for 1vs1 and 4 for 4vs1
if N_train == 4:
    N_loops = 1
elif N_train == 1:
    N_loops = 4
else:
    raise ValueError

ROC_FAR = 0
ROC_FRR = 0
TOTAL_P = 0
TOTAL_N = 0

# MCYT
N_user = 100
EER_G = []; EER_L = []

for seed in [111, 222, 333, 444, 555]:
    epoch = str(epoch)
    feats = numpy.load("./exps/seed%d/mcyt/feats_epoch%s.npy"%(seed, epoch))
    feats = feats.reshape(feats.shape[0], -1, 32)
    feats = feats / (numpy.sum(feats**2, axis=-1, keepdims=True)**0.5 + 1e-8)
    feats = feats.reshape(feats.shape[0], -1)
    feats = feats / (numpy.sum(feats**2, axis=1, keepdims=True)**0.5 + 1e-8)
    labels = numpy.load("./exps/seed%d/mcyt/labels_epoch%s.npy"%(seed, epoch))
    inds = numpy.load("./exps/seed%d/mcyt/inds_epoch%s.npy"%(seed, epoch))

    for rand in range(N_loops):
        EERs = []; datum_p = []; datum_n = []
        for k in numpy.arange(0, N_user): 
            idxUser = numpy.where(labels==k)[0]
            dset = feats[idxUser]; ind = inds[idxUser]

            idxGen = numpy.where(ind)[0]
            # The first four genuine signatures are used as templates.
            idxTemp = idxGen[0:4] 
            if N_train == 1:
                temp = dset[idxTemp[rand:rand+1]]
            else:
                temp = dset[idxTemp]
            # The rest genuine signatures and skilled forgeries are used for testing.
            test = dset[list(set(range(len(ind))) - set(idxTemp))]
            testInd = ind[list(set(range(len(ind))) - set(idxTemp))]

            distTemp = numpy.sum((temp[:,None,:] - temp[None,:,:])**2, axis=2)
            dist =  numpy.sum((test[:,None,:] - temp[None,:,:])**2, axis=2)

            # Intra-writer statistics for score normalization. We only use the min & mean scores.
            idx, dtmp, dmax, dmin, dmean, dvar = selectTemplate(distTemp)
            distMax = numpy.max(dist, axis=1)[:,None] / dmax
            distMin = numpy.min(dist, axis=1)[:,None] / dmin
            distMean = numpy.mean(dist, axis=1)[:,None] / dmean

            dist = numpy.concatenate((distMax, distMin, distMean), axis=1) / 10. 

            datum_p.append(dist[numpy.where(testInd)[0]])
            datum_n.append(dist[numpy.where(1-testInd)[0]])

            # '''Local threshold'''
            user_p = datum_p[-1]
            user_n = datum_n[-1]
            th = numpy.arange(0, 5, 0.001)[None,:]
            FRR = 1. - numpy.sum(numpy.sum(user_p[:,1:] * [1, 1/1], axis=1)[:,None] - th <= 0, axis=0) / float(user_p[:,1:].shape[0])
            FAR = 1. - numpy.sum(numpy.sum(user_n[:,1:] * [1, 1/1], axis=1)[:,None] - th >= 0, axis=0) / float(user_n[:,1:].shape[0])
            EERs.append(getEER(FAR, FRR)[0] * 100)

        EER_L.append(numpy.mean(EERs))

        datum_p = numpy.concatenate(datum_p, axis=0)
        datum_n = numpy.concatenate(datum_n, axis=0)
        '''Global threshold'''
        k = 1 #Simply set to 1.
        th = numpy.arange(0, 5, 0.001)[None,:]
        # scoreScatter(datum_p, datum_n)
        FRR = 1. - numpy.sum(numpy.sum(datum_p[:,1:] * [1, 1/k], axis=1)[:,None] - th <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - numpy.sum(numpy.sum(datum_n[:,1:] * [1, 1/k], axis=1)[:,None] - th >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 0.2 / N_loops * datum_n.shape[0] 
        ROC_FRR += FRR * 0.2 / N_loops * datum_p.shape[0]
    
print (epoch, "~")
print ("Global threshold:", numpy.mean(EER_G), "Local threshold:", numpy.mean(EER_L))
TOTAL_P += datum_p.shape[0] 
TOTAL_N += datum_n.shape[0] 

# BiosecurID
N_user = 132
EER_G = []; EER_L = []

for seed in [111, 222, 333, 444, 555]:
    epoch = str(epoch)
    feats = numpy.load("./exps/seed%d/bsid/feats_epoch%s.npy"%(seed, epoch))
    feats = feats.reshape(feats.shape[0], -1, 32)
    feats = feats / (numpy.sum(feats**2, axis=-1, keepdims=True)**0.5 + 1e-8)
    feats = feats.reshape(feats.shape[0], -1)
    feats = feats / (numpy.sum(feats**2, axis=1, keepdims=True)**0.5 + 1e-8)
    labels = numpy.load("./exps/seed%d/bsid/labels_epoch%s.npy"%(seed, epoch))
    inds = numpy.load("./exps/seed%d/bsid/inds_epoch%s.npy"%(seed, epoch))

    for rand in range(N_loops):
        EERs = []; datum_p = []; datum_n = []
        for k in numpy.arange(0, N_user): 
            idxUser = numpy.where(labels==k)[0]
            dset = feats[idxUser]; ind = inds[idxUser]

            idxGen = numpy.where(ind)[0]
            # The first four genuine signatures are used as templates.
            idxTemp = idxGen[0:4] 
            if N_train == 1:
                temp = dset[idxTemp[rand:rand+1]]
            else:
                temp = dset[idxTemp]
            # The rest genuine signatures and skilled forgeries are used for testing.
            test = dset[list(set(range(len(ind))) - set(idxTemp))]
            testInd = ind[list(set(range(len(ind))) - set(idxTemp))]

            distTemp = numpy.sum((temp[:,None,:] - temp[None,:,:])**2, axis=2)
            dist =  numpy.sum((test[:,None,:] - temp[None,:,:])**2, axis=2)

            # Intra-writer statistics for score normalization. We only use the min & mean scores.
            idx, dtmp, dmax, dmin, dmean, dvar = selectTemplate(distTemp)
            distMax = numpy.max(dist, axis=1)[:,None] / dmax
            distMin = numpy.min(dist, axis=1)[:,None] / dmin
            distMean = numpy.mean(dist, axis=1)[:,None] / dmean

            dist = numpy.concatenate((distMax, distMin, distMean), axis=1) / 10. 

            datum_p.append(dist[numpy.where(testInd)[0]])
            datum_n.append(dist[numpy.where(1-testInd)[0]])

            # '''Local threshold'''
            user_p = datum_p[-1]
            user_n = datum_n[-1]
            th = numpy.arange(0, 5, 0.001)[None,:]
            # scoreScatter(user_p, user_n)
            FRR = 1. - numpy.sum(numpy.sum(user_p[:,1:] * [1, 1/1], axis=1)[:,None] - th <= 0, axis=0) / float(user_p[:,1:].shape[0])
            FAR = 1. - numpy.sum(numpy.sum(user_n[:,1:] * [1, 1/1], axis=1)[:,None] - th >= 0, axis=0) / float(user_n[:,1:].shape[0])
            EERs.append(getEER(FAR, FRR)[0] * 100)
            # print (k, EERs[-1])
            
        EER_L.append(numpy.mean(EERs))

        datum_p = numpy.concatenate(datum_p, axis=0)
        datum_n = numpy.concatenate(datum_n, axis=0)
        '''Global threshold'''
        k = 1 #Simply set to 1.
        th = numpy.arange(0, 5, 0.001)[None,:]
        # scoreScatter(datum_p, datum_n)
        FRR = 1. - numpy.sum(numpy.sum(datum_p[:,1:] * [1, 1/k], axis=1)[:,None] - th <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - numpy.sum(numpy.sum(datum_n[:,1:] * [1, 1/k], axis=1)[:,None] - th >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 0.2 / N_loops * datum_n.shape[0]
        ROC_FRR += FRR * 0.2 / N_loops * datum_p.shape[0]
    
print (epoch, "~")
print ("Global threshold:", numpy.mean(EER_G), "Local threshold:", numpy.mean(EER_L))
TOTAL_P += datum_p.shape[0]
TOTAL_N += datum_n.shape[0]

# Biosecure DS2
N_user = 140
EER_G = []; EER_L = []

for seed in [111, 222, 333, 444, 555]:
    epoch = str(epoch)
    feats = numpy.load("./exps/seed%d/bsds2/feats_epoch%s.npy"%(seed, epoch))
    feats = feats.reshape(feats.shape[0], -1, 32)
    feats = feats / (numpy.sum(feats**2, axis=-1, keepdims=True)**0.5 + 1e-8)
    feats = feats.reshape(feats.shape[0], -1)
    feats = feats / (numpy.sum(feats**2, axis=1, keepdims=True)**0.5 + 1e-8)
    labels = numpy.load("./exps/seed%d/bsds2/labels_epoch%s.npy"%(seed, epoch))
    inds = numpy.load("./exps/seed%d/bsds2/inds_epoch%s.npy"%(seed, epoch))

    for rand in range(N_loops):
        EERs = []; datum_p = []; datum_n = []
        for k in numpy.arange(0, N_user): 
            idxUser = numpy.where(labels==k)[0]
            dset = feats[idxUser]; ind = inds[idxUser]

            # For BSDB2, 4 templates in session 1 + session2. 
            # Indexes correspond to the naming convention of the signature files.
            idxs = numpy.concatenate([numpy.array([0,1,2,3]), numpy.arange(15, 50)])
            dset = dset[idxs]
            ind = ind[idxs]

            idxGen = numpy.where(ind)[0]
            # The first four genuine signatures are used as templates.
            idxTemp = idxGen[0:4] 
            if N_train == 1:
                temp = dset[idxTemp[rand:rand+1]]
            else:
                temp = dset[idxTemp]
            # The rest genuine signatures and skilled forgeries are used for testing.
            test = dset[list(set(range(len(ind))) - set(idxTemp))]
            testInd = ind[list(set(range(len(ind))) - set(idxTemp))]

            distTemp = numpy.sum((temp[:,None,:] - temp[None,:,:])**2, axis=2)
            dist =  numpy.sum((test[:,None,:] - temp[None,:,:])**2, axis=2)

            # Intra-writer statistics for score normalization. We only use the min & mean scores.
            idx, dtmp, dmax, dmin, dmean, dvar = selectTemplate(distTemp)
            distMax = numpy.max(dist, axis=1)[:,None] / dmax
            distMin = numpy.min(dist, axis=1)[:,None] / dmin
            distMean = numpy.mean(dist, axis=1)[:,None] / dmean

            dist = numpy.concatenate((distMax, distMin, distMean), axis=1) / 10. 

            datum_p.append(dist[numpy.where(testInd)[0]])
            datum_n.append(dist[numpy.where(1-testInd)[0]])

            # '''Local threshold'''
            user_p = datum_p[-1]
            user_n = datum_n[-1]
            th = numpy.arange(0, 5, 0.001)[None,:]
            # scoreScatter(user_p, user_n)
            FRR = 1. - numpy.sum(numpy.sum(user_p[:,1:] * [1, 1/1], axis=1)[:,None] - th <= 0, axis=0) / float(user_p[:,1:].shape[0])
            FAR = 1. - numpy.sum(numpy.sum(user_n[:,1:] * [1, 1/1], axis=1)[:,None] - th >= 0, axis=0) / float(user_n[:,1:].shape[0])
            EERs.append(getEER(FAR, FRR)[0] * 100)
            # print (k, EERs[-1])
            
        EER_L.append(numpy.mean(EERs))

        datum_p = numpy.concatenate(datum_p, axis=0)
        datum_n = numpy.concatenate(datum_n, axis=0)
        '''Global threshold'''
        k = 1 #Simply set to 1.
        th = numpy.arange(0, 5, 0.001)[None,:]
        # scoreScatter(datum_p, datum_n)
        FRR = 1. - numpy.sum(numpy.sum(datum_p[:,1:] * [1, 1/k], axis=1)[:,None] - th <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - numpy.sum(numpy.sum(datum_n[:,1:] * [1, 1/k], axis=1)[:,None] - th >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 0.2 / N_loops * datum_n.shape[0]
        ROC_FRR += FRR * 0.2 / N_loops * datum_p.shape[0]
    
print (epoch, "~")
print ("Global threshold:", numpy.mean(EER_G), "Local threshold:", numpy.mean(EER_L))
TOTAL_P += datum_p.shape[0]
TOTAL_N += datum_n.shape[0]

# e-BioSign DS2
N_user = 35
EER_G = []; EER_L = []

for seed in [111, 222, 333, 444, 555]:
    epoch = str(epoch)
    feats = numpy.load("./exps/seed%d/ebio2/feats_epoch%s.npy"%(seed, epoch))
    feats = feats.reshape(feats.shape[0], -1, 32)
    feats = feats / (numpy.sum(feats**2, axis=-1, keepdims=True)**0.5 + 1e-8)
    feats = feats.reshape(feats.shape[0], -1)
    feats = feats / (numpy.sum(feats**2, axis=1, keepdims=True)**0.5 + 1e-8)
    labels = numpy.load("./exps/seed%d/ebio2/labels_epoch%s.npy"%(seed, epoch))
    inds = numpy.load("./exps/seed%d/ebio2/inds_epoch%s.npy"%(seed, epoch))

    for rand in range(N_loops):
        EERs = []; datum_p = []; datum_n = []
        for k in numpy.arange(0, N_user): 
            idxUser = numpy.where(labels==k)[0]
            dset = feats[idxUser]; ind = inds[idxUser]

            idxGen = numpy.where(ind)[0]
            # The first four genuine signatures are used as templates.
            idxTemp = idxGen[0:4] 
            if N_train == 1:
                temp = dset[idxTemp[rand:rand+1]]
            else:
                temp = dset[idxTemp]
            # The rest genuine signatures and skilled forgeries are used for testing.
            test = dset[list(set(range(len(ind))) - set(idxTemp))]
            testInd = ind[list(set(range(len(ind))) - set(idxTemp))]

            distTemp = numpy.sum((temp[:,None,:] - temp[None,:,:])**2, axis=2)
            dist =  numpy.sum((test[:,None,:] - temp[None,:,:])**2, axis=2)

            # Intra-writer statistics for score normalization. We only use the min & mean scores.
            idx, dtmp, dmax, dmin, dmean, dvar = selectTemplate(distTemp)
            distMax = numpy.max(dist, axis=1)[:,None] / dmax
            distMin = numpy.min(dist, axis=1)[:,None] / dmin
            distMean = numpy.mean(dist, axis=1)[:,None] / dmean

            dist = numpy.concatenate((distMax, distMin, distMean), axis=1) / 10. 

            datum_p.append(dist[numpy.where(testInd)[0]])
            datum_n.append(dist[numpy.where(1-testInd)[0]])

            # '''Local threshold'''
            user_p = datum_p[-1]
            user_n = datum_n[-1]
            th = numpy.arange(0, 5, 0.001)[None,:]
            # scoreScatter(user_p, user_n)
            FRR = 1. - numpy.sum(numpy.sum(user_p[:,1:] * [1, 1/1], axis=1)[:,None] - th <= 0, axis=0) / float(user_p[:,1:].shape[0])
            FAR = 1. - numpy.sum(numpy.sum(user_n[:,1:] * [1, 1/1], axis=1)[:,None] - th >= 0, axis=0) / float(user_n[:,1:].shape[0])
            EERs.append(getEER(FAR, FRR)[0] * 100)
            # print (k, EERs[-1])
            
        EER_L.append(numpy.mean(EERs))

        datum_p = numpy.concatenate(datum_p, axis=0)
        datum_n = numpy.concatenate(datum_n, axis=0)
        '''Global threshold'''
        k = 1 #Simply set to 1.
        th = numpy.arange(0, 5, 0.001)[None,:]
        # scoreScatter(datum_p, datum_n)
        FRR = 1. - numpy.sum(numpy.sum(datum_p[:,1:] * [1, 1/k], axis=1)[:,None] - th <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - numpy.sum(numpy.sum(datum_n[:,1:] * [1, 1/k], axis=1)[:,None] - th >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 0.2 / N_loops * datum_n.shape[0]
        ROC_FRR += FRR * 0.2 / N_loops * datum_p.shape[0]
    
print (epoch, "~")
print ("Global threshold:", numpy.mean(EER_G), "Local threshold:", numpy.mean(EER_L))
TOTAL_P += datum_p.shape[0]
TOTAL_N += datum_n.shape[0]

# e-BioSign DS1, w1
N_user = 35
device = 0
EER_G = []; EER_L = []

for seed in [111, 222, 333, 444, 555]:
    epoch = str(epoch)
    feats = numpy.load("./exps/seed%d/ebio1/feats_epoch%s.npy"%(seed, epoch))
    feats = feats.reshape(feats.shape[0], -1, 32)
    feats = feats / (numpy.sum(feats**2, axis=-1, keepdims=True)**0.5 + 1e-8)
    feats = feats.reshape(feats.shape[0], -1)
    feats = feats / (numpy.sum(feats**2, axis=1, keepdims=True)**0.5 + 1e-8)
    labels = numpy.load("./exps/seed%d/ebio1/labels_epoch%s.npy"%(seed, epoch))
    inds = numpy.load("./exps/seed%d/ebio1/inds_epoch%s.npy"%(seed, epoch))

    for rand in range(N_loops):
        EERs = []; datum_p = []; datum_n = []
        for k in numpy.arange(0, N_user): 
            idxUser = numpy.where(labels==k)[0]
            dset = feats[idxUser]; ind = inds[idxUser]
            
            deviceIndP = numpy.array([0,1,10,11,20,21,30,31]) + device * 2
            deviceIndN = numpy.array([40,41,42,55,56,57]) + device * 3
            dset = numpy.concatenate((dset[deviceIndP], dset[deviceIndN]), axis=0)
            ind = numpy.concatenate((ind[deviceIndP], ind[deviceIndN]), axis=0)

            idxGen = numpy.where(ind)[0]
            # The first four genuine signatures are used as templates.
            idxTemp = idxGen[0:4] 
            if N_train == 1:
                temp = dset[idxTemp[rand:rand+1]]
            else:
                temp = dset[idxTemp]
            # The rest genuine signatures and skilled forgeries are used for testing.
            test = dset[list(set(range(len(ind))) - set(idxTemp))]
            testInd = ind[list(set(range(len(ind))) - set(idxTemp))]

            distTemp = numpy.sum((temp[:,None,:] - temp[None,:,:])**2, axis=2)
            dist =  numpy.sum((test[:,None,:] - temp[None,:,:])**2, axis=2)

            # Intra-writer statistics for score normalization. We only use the min & mean scores.
            idx, dtmp, dmax, dmin, dmean, dvar = selectTemplate(distTemp)
            distMax = numpy.max(dist, axis=1)[:,None] / dmax
            distMin = numpy.min(dist, axis=1)[:,None] / dmin
            distMean = numpy.mean(dist, axis=1)[:,None] / dmean

            dist = numpy.concatenate((distMax, distMin, distMean), axis=1) / 10. 

            datum_p.append(dist[numpy.where(testInd)[0]])
            datum_n.append(dist[numpy.where(1-testInd)[0]])

            # '''Local threshold'''
            user_p = datum_p[-1]
            user_n = datum_n[-1]
            th = numpy.arange(0, 5, 0.001)[None,:]
            # scoreScatter(user_p, user_n)
            FRR = 1. - numpy.sum(numpy.sum(user_p[:,1:] * [1, 1/1], axis=1)[:,None] - th <= 0, axis=0) / float(user_p[:,1:].shape[0])
            FAR = 1. - numpy.sum(numpy.sum(user_n[:,1:] * [1, 1/1], axis=1)[:,None] - th >= 0, axis=0) / float(user_n[:,1:].shape[0])
            EERs.append(getEER(FAR, FRR)[0] * 100)
            # print (k, EERs[-1])
            
        EER_L.append(numpy.mean(EERs))

        datum_p = numpy.concatenate(datum_p, axis=0)
        datum_n = numpy.concatenate(datum_n, axis=0)
        '''Global threshold'''
        k = 1 #Simply set to 1.
        th = numpy.arange(0, 5, 0.001)[None,:]
        # scoreScatter(datum_p, datum_n)
        FRR = 1. - numpy.sum(numpy.sum(datum_p[:,1:] * [1, 1/k], axis=1)[:,None] - th <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - numpy.sum(numpy.sum(datum_n[:,1:] * [1, 1/k], axis=1)[:,None] - th >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 0.2 / N_loops * datum_n.shape[0]
        ROC_FRR += FRR * 0.2 / N_loops * datum_p.shape[0]
    
print (epoch, "~")
print ("Global threshold:", numpy.mean(EER_G), "Local threshold:", numpy.mean(EER_L))
TOTAL_P += datum_p.shape[0]
TOTAL_N += datum_n.shape[0]

# e-BioSign DS1, w2
N_user = 35
device = 1
EER_G = []; EER_L = []

for seed in [111, 222, 333, 444, 555]:
    epoch = str(epoch)
    feats = numpy.load("./exps/seed%d/ebio1/feats_epoch%s.npy"%(seed, epoch))
    feats = feats.reshape(feats.shape[0], -1, 32)
    feats = feats / (numpy.sum(feats**2, axis=-1, keepdims=True)**0.5 + 1e-8)
    feats = feats.reshape(feats.shape[0], -1)
    feats = feats / (numpy.sum(feats**2, axis=1, keepdims=True)**0.5 + 1e-8)
    labels = numpy.load("./exps/seed%d/ebio1/labels_epoch%s.npy"%(seed, epoch))
    inds = numpy.load("./exps/seed%d/ebio1/inds_epoch%s.npy"%(seed, epoch))

    for rand in range(N_loops):
        EERs = []; datum_p = []; datum_n = []
        for k in numpy.arange(0, N_user): 
            idxUser = numpy.where(labels==k)[0]
            dset = feats[idxUser]; ind = inds[idxUser]
            
            deviceIndP = numpy.array([0,1,10,11,20,21,30,31]) + device * 2
            deviceIndN = numpy.array([40,41,42,55,56,57]) + device * 3
            dset = numpy.concatenate((dset[deviceIndP], dset[deviceIndN]), axis=0)
            ind = numpy.concatenate((ind[deviceIndP], ind[deviceIndN]), axis=0)

            idxGen = numpy.where(ind)[0]
            # The first four genuine signatures are used as templates.
            idxTemp = idxGen[0:4] 
            if N_train == 1:
                temp = dset[idxTemp[rand:rand+1]]
            else:
                temp = dset[idxTemp]
            # The rest genuine signatures and skilled forgeries are used for testing.
            test = dset[list(set(range(len(ind))) - set(idxTemp))]
            testInd = ind[list(set(range(len(ind))) - set(idxTemp))]

            distTemp = numpy.sum((temp[:,None,:] - temp[None,:,:])**2, axis=2)
            dist =  numpy.sum((test[:,None,:] - temp[None,:,:])**2, axis=2)

            # Intra-writer statistics for score normalization. We only use the min & mean scores.
            idx, dtmp, dmax, dmin, dmean, dvar = selectTemplate(distTemp)
            distMax = numpy.max(dist, axis=1)[:,None] / dmax
            distMin = numpy.min(dist, axis=1)[:,None] / dmin
            distMean = numpy.mean(dist, axis=1)[:,None] / dmean

            dist = numpy.concatenate((distMax, distMin, distMean), axis=1) / 10. 

            datum_p.append(dist[numpy.where(testInd)[0]])
            datum_n.append(dist[numpy.where(1-testInd)[0]])

            # '''Local threshold'''
            user_p = datum_p[-1]
            user_n = datum_n[-1]
            th = numpy.arange(0, 5, 0.001)[None,:]
            # scoreScatter(user_p, user_n)
            FRR = 1. - numpy.sum(numpy.sum(user_p[:,1:] * [1, 1/1], axis=1)[:,None] - th <= 0, axis=0) / float(user_p[:,1:].shape[0])
            FAR = 1. - numpy.sum(numpy.sum(user_n[:,1:] * [1, 1/1], axis=1)[:,None] - th >= 0, axis=0) / float(user_n[:,1:].shape[0])
            EERs.append(getEER(FAR, FRR)[0] * 100)
            # print (k, EERs[-1])
            
        EER_L.append(numpy.mean(EERs))

        datum_p = numpy.concatenate(datum_p, axis=0)
        datum_n = numpy.concatenate(datum_n, axis=0)
        '''Global threshold'''
        k = 1 #Simply set to 1.
        th = numpy.arange(0, 5, 0.001)[None,:]
        # scoreScatter(datum_p, datum_n)
        FRR = 1. - numpy.sum(numpy.sum(datum_p[:,1:] * [1, 1/k], axis=1)[:,None] - th <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - numpy.sum(numpy.sum(datum_n[:,1:] * [1, 1/k], axis=1)[:,None] - th >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 0.2 / N_loops * datum_n.shape[0]
        ROC_FRR += FRR * 0.2 / N_loops * datum_p.shape[0]
    
print (epoch, "~")
print ("Global threshold:", numpy.mean(EER_G), "Local threshold:", numpy.mean(EER_L))
TOTAL_P += datum_p.shape[0]
TOTAL_N += datum_n.shape[0]

# e-BioSign DS1, w3
N_user = 35
device = 2
EER_G = []; EER_L = []

for seed in [111, 222, 333, 444, 555]:
    epoch = str(epoch)
    feats = numpy.load("./exps/seed%d/ebio1/feats_epoch%s.npy"%(seed, epoch))
    feats = feats.reshape(feats.shape[0], -1, 32)
    feats = feats / (numpy.sum(feats**2, axis=-1, keepdims=True)**0.5 + 1e-8)
    feats = feats.reshape(feats.shape[0], -1)
    feats = feats / (numpy.sum(feats**2, axis=1, keepdims=True)**0.5 + 1e-8)
    labels = numpy.load("./exps/seed%d/ebio1/labels_epoch%s.npy"%(seed, epoch))
    inds = numpy.load("./exps/seed%d/ebio1/inds_epoch%s.npy"%(seed, epoch))

    for rand in range(N_loops):
        EERs = []; datum_p = []; datum_n = []
        for k in numpy.arange(0, N_user): 
            idxUser = numpy.where(labels==k)[0]
            dset = feats[idxUser]; ind = inds[idxUser]
            
            deviceIndP = numpy.array([0,1,10,11,20,21,30,31]) + device * 2
            deviceIndN = numpy.array([40,41,42,55,56,57]) + device * 3
            dset = numpy.concatenate((dset[deviceIndP], dset[deviceIndN]), axis=0)
            ind = numpy.concatenate((ind[deviceIndP], ind[deviceIndN]), axis=0)

            idxGen = numpy.where(ind)[0]
            # The first four genuine signatures are used as templates.
            idxTemp = idxGen[0:4] 
            if N_train == 1:
                temp = dset[idxTemp[rand:rand+1]]
            else:
                temp = dset[idxTemp]
            # The rest genuine signatures and skilled forgeries are used for testing.
            test = dset[list(set(range(len(ind))) - set(idxTemp))]
            testInd = ind[list(set(range(len(ind))) - set(idxTemp))]

            distTemp = numpy.sum((temp[:,None,:] - temp[None,:,:])**2, axis=2)
            dist =  numpy.sum((test[:,None,:] - temp[None,:,:])**2, axis=2)

            # Intra-writer statistics for score normalization. We only use the min & mean scores.
            idx, dtmp, dmax, dmin, dmean, dvar = selectTemplate(distTemp)
            distMax = numpy.max(dist, axis=1)[:,None] / dmax
            distMin = numpy.min(dist, axis=1)[:,None] / dmin
            distMean = numpy.mean(dist, axis=1)[:,None] / dmean

            dist = numpy.concatenate((distMax, distMin, distMean), axis=1) / 10. 

            datum_p.append(dist[numpy.where(testInd)[0]])
            datum_n.append(dist[numpy.where(1-testInd)[0]])

            # '''Local threshold'''
            user_p = datum_p[-1]
            user_n = datum_n[-1]
            th = numpy.arange(0, 5, 0.001)[None,:]
            # scoreScatter(user_p, user_n)
            FRR = 1. - numpy.sum(numpy.sum(user_p[:,1:] * [1, 1/1], axis=1)[:,None] - th <= 0, axis=0) / float(user_p[:,1:].shape[0])
            FAR = 1. - numpy.sum(numpy.sum(user_n[:,1:] * [1, 1/1], axis=1)[:,None] - th >= 0, axis=0) / float(user_n[:,1:].shape[0])
            EERs.append(getEER(FAR, FRR)[0] * 100)
            # print (k, EERs[-1])
            
        EER_L.append(numpy.mean(EERs))

        datum_p = numpy.concatenate(datum_p, axis=0)
        datum_n = numpy.concatenate(datum_n, axis=0)
        '''Global threshold'''
        k = 1 #Simply set to 1.
        th = numpy.arange(0, 5, 0.001)[None,:]
        # scoreScatter(datum_p, datum_n)
        FRR = 1. - numpy.sum(numpy.sum(datum_p[:,1:] * [1, 1/k], axis=1)[:,None] - th <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - numpy.sum(numpy.sum(datum_n[:,1:] * [1, 1/k], axis=1)[:,None] - th >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 0.2 / N_loops * datum_n.shape[0]
        ROC_FRR += FRR * 0.2 / N_loops * datum_p.shape[0]
    
print (epoch, "~")
print ("Global threshold:", numpy.mean(EER_G), "Local threshold:", numpy.mean(EER_L))
TOTAL_P += datum_p.shape[0]
TOTAL_N += datum_n.shape[0]

# e-BioSign DS1, w4
N_user = 35
device = 3
EER_G = []; EER_L = []

for seed in [111, 222, 333, 444, 555]:
    epoch = str(epoch)
    feats = numpy.load("./exps/seed%d/ebio1/feats_epoch%s.npy"%(seed, epoch))
    feats = feats.reshape(feats.shape[0], -1, 32)
    feats = feats / (numpy.sum(feats**2, axis=-1, keepdims=True)**0.5 + 1e-8)
    feats = feats.reshape(feats.shape[0], -1)
    feats = feats / (numpy.sum(feats**2, axis=1, keepdims=True)**0.5 + 1e-8)
    labels = numpy.load("./exps/seed%d/ebio1/labels_epoch%s.npy"%(seed, epoch))
    inds = numpy.load("./exps/seed%d/ebio1/inds_epoch%s.npy"%(seed, epoch))

    for rand in range(N_loops):
        EERs = []; datum_p = []; datum_n = []
        for k in numpy.arange(0, N_user): 
            idxUser = numpy.where(labels==k)[0]
            dset = feats[idxUser]; ind = inds[idxUser]
            
            deviceIndP = numpy.array([0,1,10,11,20,21,30,31]) + device * 2
            deviceIndN = numpy.array([40,41,42,55,56,57]) + device * 3
            dset = numpy.concatenate((dset[deviceIndP], dset[deviceIndN]), axis=0)
            ind = numpy.concatenate((ind[deviceIndP], ind[deviceIndN]), axis=0)

            idxGen = numpy.where(ind)[0]
            # The first four genuine signatures are used as templates.
            idxTemp = idxGen[0:4] 
            if N_train == 1:
                temp = dset[idxTemp[rand:rand+1]]
            else:
                temp = dset[idxTemp]
            # The rest genuine signatures and skilled forgeries are used for testing.
            test = dset[list(set(range(len(ind))) - set(idxTemp))]
            testInd = ind[list(set(range(len(ind))) - set(idxTemp))]

            distTemp = numpy.sum((temp[:,None,:] - temp[None,:,:])**2, axis=2)
            dist =  numpy.sum((test[:,None,:] - temp[None,:,:])**2, axis=2)

            # Intra-writer statistics for score normalization. We only use the min & mean scores.
            idx, dtmp, dmax, dmin, dmean, dvar = selectTemplate(distTemp)
            distMax = numpy.max(dist, axis=1)[:,None] / dmax
            distMin = numpy.min(dist, axis=1)[:,None] / dmin
            distMean = numpy.mean(dist, axis=1)[:,None] / dmean

            dist = numpy.concatenate((distMax, distMin, distMean), axis=1) / 10. 

            datum_p.append(dist[numpy.where(testInd)[0]])
            datum_n.append(dist[numpy.where(1-testInd)[0]])

            # '''Local threshold'''
            user_p = datum_p[-1]
            user_n = datum_n[-1]
            th = numpy.arange(0, 5, 0.001)[None,:]
            # scoreScatter(user_p, user_n)
            FRR = 1. - numpy.sum(numpy.sum(user_p[:,1:] * [1, 1/1], axis=1)[:,None] - th <= 0, axis=0) / float(user_p[:,1:].shape[0])
            FAR = 1. - numpy.sum(numpy.sum(user_n[:,1:] * [1, 1/1], axis=1)[:,None] - th >= 0, axis=0) / float(user_n[:,1:].shape[0])
            EERs.append(getEER(FAR, FRR)[0] * 100)
            # print (k, EERs[-1])
            
        EER_L.append(numpy.mean(EERs))

        datum_p = numpy.concatenate(datum_p, axis=0)
        datum_n = numpy.concatenate(datum_n, axis=0)
        '''Global threshold'''
        k = 1 #Simply set to 1.
        th = numpy.arange(0, 5, 0.001)[None,:]
        # scoreScatter(datum_p, datum_n)
        FRR = 1. - numpy.sum(numpy.sum(datum_p[:,1:] * [1, 1/k], axis=1)[:,None] - th <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - numpy.sum(numpy.sum(datum_n[:,1:] * [1, 1/k], axis=1)[:,None] - th >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 0.2 / N_loops * datum_n.shape[0]
        ROC_FRR += FRR * 0.2 / N_loops * datum_p.shape[0]
    
print (epoch, "~")
print ("Global threshold:", numpy.mean(EER_G), "Local threshold:", numpy.mean(EER_L))
TOTAL_P += datum_p.shape[0]
TOTAL_N += datum_n.shape[0]

# e-BioSign DS1, w5
N_user = 35
device = 4
EER_G = []; EER_L = []

for seed in [111, 222, 333, 444, 555]:
    epoch = str(epoch)
    feats = numpy.load("./exps/seed%d/ebio1/feats_epoch%s.npy"%(seed, epoch))
    feats = feats.reshape(feats.shape[0], -1, 32)
    feats = feats / (numpy.sum(feats**2, axis=-1, keepdims=True)**0.5 + 1e-8)
    feats = feats.reshape(feats.shape[0], -1)
    feats = feats / (numpy.sum(feats**2, axis=1, keepdims=True)**0.5 + 1e-8)
    labels = numpy.load("./exps/seed%d/ebio1/labels_epoch%s.npy"%(seed, epoch))
    inds = numpy.load("./exps/seed%d/ebio1/inds_epoch%s.npy"%(seed, epoch))

    for rand in range(N_loops):
        EERs = []; datum_p = []; datum_n = []
        for k in numpy.arange(0, N_user): 
            idxUser = numpy.where(labels==k)[0]
            dset = feats[idxUser]; ind = inds[idxUser]
            
            deviceIndP = numpy.array([0,1,10,11,20,21,30,31]) + device * 2
            deviceIndN = numpy.array([40,41,42,55,56,57]) + device * 3
            dset = numpy.concatenate((dset[deviceIndP], dset[deviceIndN]), axis=0)
            ind = numpy.concatenate((ind[deviceIndP], ind[deviceIndN]), axis=0)

            idxGen = numpy.where(ind)[0]
            # The first four genuine signatures are used as templates.
            idxTemp = idxGen[0:4] 
            if N_train == 1:
                temp = dset[idxTemp[rand:rand+1]]
            else:
                temp = dset[idxTemp]
            # The rest genuine signatures and skilled forgeries are used for testing.
            test = dset[list(set(range(len(ind))) - set(idxTemp))]
            testInd = ind[list(set(range(len(ind))) - set(idxTemp))]

            distTemp = numpy.sum((temp[:,None,:] - temp[None,:,:])**2, axis=2)
            dist =  numpy.sum((test[:,None,:] - temp[None,:,:])**2, axis=2)

            # Intra-writer statistics for score normalization. We only use the min & mean scores.
            idx, dtmp, dmax, dmin, dmean, dvar = selectTemplate(distTemp)
            distMax = numpy.max(dist, axis=1)[:,None] / dmax
            distMin = numpy.min(dist, axis=1)[:,None] / dmin
            distMean = numpy.mean(dist, axis=1)[:,None] / dmean

            dist = numpy.concatenate((distMax, distMin, distMean), axis=1) / 10. 

            datum_p.append(dist[numpy.where(testInd)[0]])
            datum_n.append(dist[numpy.where(1-testInd)[0]])

            # '''Local threshold'''
            user_p = datum_p[-1]
            user_n = datum_n[-1]
            th = numpy.arange(0, 5, 0.001)[None,:]
            # scoreScatter(user_p, user_n)
            FRR = 1. - numpy.sum(numpy.sum(user_p[:,1:] * [1, 1/1], axis=1)[:,None] - th <= 0, axis=0) / float(user_p[:,1:].shape[0])
            FAR = 1. - numpy.sum(numpy.sum(user_n[:,1:] * [1, 1/1], axis=1)[:,None] - th >= 0, axis=0) / float(user_n[:,1:].shape[0])
            EERs.append(getEER(FAR, FRR)[0] * 100)
            # print (k, EERs[-1])
            
        EER_L.append(numpy.mean(EERs))

        datum_p = numpy.concatenate(datum_p, axis=0)
        datum_n = numpy.concatenate(datum_n, axis=0)
        '''Global threshold'''
        k = 1 #Simply set to 1.
        th = numpy.arange(0, 5, 0.001)[None,:]
        # scoreScatter(datum_p, datum_n)
        FRR = 1. - numpy.sum(numpy.sum(datum_p[:,1:] * [1, 1/k], axis=1)[:,None] - th <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - numpy.sum(numpy.sum(datum_n[:,1:] * [1, 1/k], axis=1)[:,None] - th >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 0.2 / N_loops * datum_n.shape[0]
        ROC_FRR += FRR * 0.2 / N_loops * datum_p.shape[0]
    
print (epoch, "~")
print ("Global threshold:", numpy.mean(EER_G), "Local threshold:", numpy.mean(EER_L))
TOTAL_P += datum_p.shape[0]
TOTAL_N += datum_n.shape[0]

print(getEER(ROC_FAR*1.0/TOTAL_N, ROC_FRR*1.0/TOTAL_P)[0] * 100)
plt.plot(ROC_FAR*1.0/TOTAL_N, ROC_FRR*1.0/TOTAL_P)
plt.show()

