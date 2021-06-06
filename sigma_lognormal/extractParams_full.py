# -*- coding:utf-8 -*-
import os
import numpy
import math
import pickle
import matplotlib.pyplot as plt
from scipy import signal, convolve
from scipy.interpolate import CubicSpline

from slbox import functions, optimization
from slbox.extractionFirstMode import pathOps, pad
from slbox.extractionFirstMode import paramExtraction, paramReconstruction

def getResidualPath(P, time, path, dt=0.01):
    PATH = pathOps(time, path, smoothing=False, pad=pad, zeroInit=False)
    for param in P:
        PATH.subtractStroke(param)
    R = PATH.finalPath().copy()
    del PATH

    return R      

def paramRefinement(P, AUC, time, path, speed_x, speed_y, dt=0.01):
    # mean = numpy.mean(AUC)
    # std = numpy.std(AUC)
    # P = P[AUC>(mean-std)]
    speed = numpy.concatenate((speed_x[:,None], speed_y[:,None]), axis=1)
    P = optimization.globalOptim(time, path, speed, P, dt=dt)
    
    return P

def computeSNR(P, time, path, speed_x, speed_y):
    reconPathX, reconPathY, reconVelocX, reconVelocY = paramReconstruction(P, time)
    reconVeloc = numpy.sqrt(reconVelocX**2 + reconVelocY**2)
    velocity = numpy.sqrt(speed_x**2 + speed_y**2)
    
    reconPathX = reconPathX - numpy.mean(reconPathX)
    reconPathY = reconPathY - numpy.mean(reconPathY)
    path = path - numpy.mean(path, axis=0)
    N = numpy.sum((numpy.square(reconVelocX - speed_x) + numpy.square(reconVelocY - speed_y)))
    S = numpy.sum((numpy.square(speed_x) + numpy.square(speed_y)))
    SNRv[count] = 10. * math.log10(S / (N + 1e-8))
    N = numpy.sum(numpy.square(reconVeloc - velocity))
    S = numpy.sum(numpy.square(velocity))
    SNRv2[count] = 10. * math.log10(S / (N + 1e-8))
    N = numpy.sum((numpy.square(reconPathX - path[:,0]) + numpy.square(reconPathY - path[:,1])))
    S = numpy.sum((numpy.square(path[:,0]) + numpy.square(path[:,1])))
    SNRt[count] = 10. * math.log10(S / (N + 1e-8))
    numLN[count] = len(P)

    # plt.plot(velocity, c='r')
    # plt.plot(reconVeloc, c='b')
    # plt.show()
    # plt.plot(path[:,0], path[:,1], c='r', marker="o") #, marker="o"
    # plt.plot(reconPathX, reconPathY, c='b', marker="o") #, marker="o"
    # plt.show()

    return SNRv[count], SNRv2[count], SNRt[count]

def show(path1, path2):
    plt.plot(path1[:,0], path1[:,1], c="b", marker="*")
    plt.plot(path2[:,0], path2[:,1]+0.8, c="r", marker="*")
    plt.show()

users = pickle.load(open('../Data/MCYT_dev.pkl', 'rb'))
saveDirPrefix = "./params/mcyt_dev_full/"

# users = pickle.load(open('../Data/BSID_dev.pkl', 'rb'))
# saveDirPrefix = "./params/bsid_dev_full/"

# users = pickle.load(open('../Data/Ebio1_dev.pkl', 'rb'))
# saveDirPrefix = "./params/Ebio1_dev_full/"

# users = pickle.load(open('../Data/Ebio2_dev.pkl', 'rb'))
# saveDirPrefix = "./params/Ebio2_dev_full/"

if not os.path.exists(saveDirPrefix):
    os.mkdir(saveDirPrefix)

keys = list(users.keys())
print(keys)

global SNRv; SNRv = numpy.zeros((len(keys)*len(users[keys[0]][True])), dtype=numpy.float32)
global SNRv2; SNRv2 = numpy.zeros((len(keys)*len(users[keys[0]][True])), dtype=numpy.float32)
global SNRt; SNRt = numpy.zeros((len(keys)*len(users[keys[0]][True])), dtype=numpy.float32)
global numLN; numLN = numpy.zeros((len(keys)*len(users[keys[0]][True])), dtype=numpy.int32)
global count; count = 0

for key in keys: #
    print ("Key: ", key)
    for idx, path in enumerate(users[key][True]):
        print ("Sample: ", idx)
        path = functions.cubicSplineInterp(path, nfs=2)
        time = numpy.arange(len(path), dtype=numpy.float64) * 0.005
        P, AUC, R, vMax, targetPath, speed_x, speed_y = paramExtraction(time, path, 
                                    lAmbda=15., 
                                    smoothing=True, 
                                    zeroInit=False,
                                    saveParam=False, 
                                    dt=0.005,
                                    seqMode=False,
                                    localOptim=False)
        snrv, snrv2, snrt = computeSNR(P, time, targetPath, speed_x, speed_y)

        if snrv < 15 or snrt < 12:
            P_res, AUC_res, R, _, _, _, _ = paramExtraction(time, R, 
                                        lAmbda=25., 
                                        smoothing=False, 
                                        zeroInit=True,
                                        saveParam=False, 
                                        dt=0.005, 
                                        vm=vMax, 
                                        seqMode=True) 
            P = numpy.concatenate((P, P_res), axis=0)
            AUC = numpy.concatenate((AUC, AUC_res), axis=0)
            AUC = AUC[numpy.argsort(P[:,1])] #ascending t0
            P = P[numpy.argsort(P[:,1])] #ascending t0
            snrv, snrv2, snrt = computeSNR(P, time, targetPath, speed_x, speed_y)

        if snrv < 18 or snrt < 18:
            ''' Optimization of the sigma-lognormal parameters that are not good enough. 
            It is slow, but leads to better synthesized signatures.'''
            print ("Before optimization: ", snrv, snrv2, snrt, P.shape)
            P_new = paramRefinement(P, AUC, time, targetPath, speed_x, speed_y, dt=0.005)
            snrv_new, snrv2_new, snrt_new = computeSNR(P_new, time, targetPath, speed_x, speed_y)
            if snrt_new >= snrt:
                snrv = snrv_new; snrv2 = snrv2_new; snrt = snrt_new; P = P_new
                P = P[numpy.argsort(P[:,1])] #ascending t0
            else:
                print ("~~~")
            R = getResidualPath(P, time, path, dt=0.005)
        else:
            R = getResidualPath(P, time, path, dt=0.005)
        
        print ("------------", snrv, snrv2, snrt, P.shape)

        reconPathX, reconPathY, _, _ = paramReconstruction(P, time, dt=0.005)
        reconPathX += R[0, 0]
        reconPathY += R[0, 1]
        reconPath = numpy.concatenate([reconPathX[:,None], reconPathY[:,None]], axis=1)
        # show(reconPath, path)

        saveDir = os.path.join(saveDirPrefix, str(key))
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
        numpy.save(os.path.join(saveDir, "Pmatrix_G%d_%d.npy"%(key, idx)), P)
        numpy.save(os.path.join(saveDir, "residual_G%d_%d.npy"%(key, idx)), R)

        count += 1

    print (count)

print (numpy.mean(numLN))
print (numpy.mean(SNRv))
print (numpy.mean(SNRv2))
print (numpy.mean(SNRt))
import pdb
pdb.set_trace()

