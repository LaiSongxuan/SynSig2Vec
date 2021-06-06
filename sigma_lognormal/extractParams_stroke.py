# -*- coding:utf-8 -*-
import os
import numpy
import math
import pickle
import matplotlib.pyplot as plt
from scipy import signal, convolve
from scipy.interpolate import CubicSpline

from slbox import functions, optimization, synthesis
from slbox.extractionFirstMode import pathOps, pad, velocKernel
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

def computeSNR(P, time, path, speed_x, speed_y, plot=False):
    reconPathX, reconPathY, reconVelocX, reconVelocY = paramReconstruction(P, time, dt=0.005, plot=plot)
    reconVeloc = numpy.sqrt(reconVelocX**2 + reconVelocY**2)
    velocity = numpy.sqrt(speed_x**2 + speed_y**2)

    reconPathX = reconPathX - numpy.mean(reconPathX)
    reconPathY = reconPathY - numpy.mean(reconPathY)
    path = path - numpy.mean(path, axis=0)
    N = numpy.sum((numpy.square(reconVelocX - speed_x) + numpy.square(reconVelocY - speed_y)))
    S = numpy.sum((numpy.square(speed_x) + numpy.square(speed_y)))
    # SNRv[count] = 10. * math.log10(S / (N + 1e-8))
    snrv = 10. * math.log10(S / (N + 1e-8))
    N = numpy.sum(numpy.square(reconVeloc - velocity))
    S = numpy.sum(numpy.square(velocity))
    # SNRv2[count] = 10. * math.log10(S / (N + 1e-8))
    snrv2 = 10. * math.log10(S / (N + 1e-8))
    N = numpy.sum((numpy.square(reconPathX - path[:,0]) + numpy.square(reconPathY - path[:,1])))
    S = numpy.sum((numpy.square(path[:,0]) + numpy.square(path[:,1])))
    # SNRt[count] = 10. * math.log10(S / (N + 1e-8))
    # numLN[count] = len(P)
    snrt = 10. * math.log10(S / (N + 1e-8))

    # if plot:
    #     axes = plt.subplot(211)
    #     axes.plot(velocity, c='k', linestyle="-", label="original speed")
    #     axes.plot(reconVeloc, c='b', linestyle="-", label="recovered speed")
    #     axes.set_xticks(range(0, len(velocity), 20))
    #     axes.set_yticks(numpy.arange(0, max(velocity)+0.005, 0.005))
    #     axes.set_xticklabels(numpy.arange(0, len(velocity), 20)/200.)
    #     axes.set_yticklabels(['%.2f' % w for w in numpy.arange(0, max(velocity)+0.005, 0.005)*40])
    #     axes.set_xlabel("time (s)")
    #     axes.set_ylabel("normalized speed")
    #     axes.legend(loc=2) #fontsize="small", 
    #     axes = plt.subplot(212)
    #     axes.plot(path[:,0], path[:,1], c='k', linestyle="-", label="original trajectory") #, marker="o"
    #     axes.plot(reconPathX, reconPathY, c='b', linestyle="-", label="recovered trajectory") #, marker="*"
    #     axes.set_xlabel("coordinate x")
    #     axes.set_ylabel("coordinate y")
    #     axes.set_yticks([-0.05, 0.00, 0.05, 0.10])
    #     axes.set_yticklabels(['%.2f' % w for w in [0.05, 0.10, 0.15, 0.20]])
    #     axes.legend(loc=2) #fontsize="small", 
    #     plt.show()
  
    return snrv, snrv2, snrt

def show(path1, path2):
    plt.plot(path1[:,0], path1[:,1], c="b", marker="*")
    plt.plot(path2[:,0], path2[:,1], c="r", marker="*")
    plt.show()

def getVmax(path):
    path = functions.cubicSplineInterp(path, nfs=2)
    path[:,0] = functions.butter_lowpass_filter(path[:,0], highcut=10, fs=100*2, order=3)
    path[:,1] = functions.butter_lowpass_filter(path[:,1], highcut=10, fs=100*2, order=3)
    # path = numpy.pad(path, pad_width=((pad,pad),(0,0)), mode="edge")
    speed_x = convolve(path[:,0], velocKernel, mode="valid")
    speed_y = convolve(path[:,1], velocKernel, mode="valid")
    velocity = numpy.sqrt(speed_x**2 + speed_y**2)
    vm = numpy.max(velocity) 
    return vm

users = pickle.load(open('../Data/MCYT_dev_pad.pkl', 'rb'))
saveDirPrefix = "./params/mcyt_dev_stroke/"

# users = pickle.load(open('../Data/BSID_dev_pad.pkl', 'rb'))
# saveDirPrefix = "./params/bsid_dev_stroke/"

# users = pickle.load(open('../Data/Ebio1_dev_pad.pkl', 'rb'))
# saveDirPrefix = "./params/Ebio1_dev_stroke/"

# users = pickle.load(open('../Data/Ebio2_dev_pad.pkl', 'rb'))
# saveDirPrefix = "./params/Ebio2_dev_stroke/"

if not os.path.exists(saveDirPrefix):
    os.mkdir(saveDirPrefix)

keys = list(users.keys())
print(keys)

global SNRt; SNRt = numpy.zeros((len(keys)*len(users[keys[0]][True])), dtype=numpy.float32)
global numLN; numLN = numpy.zeros((len(keys)*len(users[keys[0]][True])), dtype=numpy.int32)
global count; count = 0

for key in keys: #
    print ("Key: ", key)
    for idx, path in enumerate(users[key][True]):
        pressure = path[:, -1]
        inds = numpy.where(pressure)[0]
        rowIdx = numpy.where((inds[1:] - inds[0:-1]) != 1)[0] + 1
        rowIdx = numpy.pad(rowIdx, (1, 1), mode="constant")
        rowIdx[-1] = inds.shape[0] 
        path = path[:, [0, 1]]
        vMax = getVmax(path)        
        Ps = [] # sigma-lognormal parameters 
        Rs = [] # residual paths
        PRs = [] # pressure 
        Ss = [] # pen-downs

        print ("Sample: ", idx)
        for i in range(0,len(rowIdx)-1):
            segLen = rowIdx[i+1]-rowIdx[i]
            stroke = path[inds[rowIdx[i]]:inds[rowIdx[i]]+segLen, [0, 1]]
            stroke = functions.cubicSplineInterp(stroke, nfs=2)
            Ss.append(stroke)
        
            if stroke.shape[0] < 24:
                Ps.append(numpy.zeros((0, 6), dtype=numpy.float64))
                Rs.append(stroke)
                PRs.append(pressure[inds[rowIdx[i]]:inds[rowIdx[i]]+segLen])
                print ("------------", 0, 0, 0, (0, 6))
                continue
            
            if numpy.sum(stroke[1:] - stroke[0:-1]) == 0:
                Ps.append(numpy.zeros((0, 6), dtype=numpy.float64))
                Rs.append(stroke)
                PRs.append(pressure[inds[rowIdx[i]]:inds[rowIdx[i]]+segLen])
                print ("------------", 0, 0, 0, (0, 6))
                continue

            time = numpy.arange(len(stroke), dtype=numpy.float64) * 0.005

            P, AUC, R, vMax, targetPath, speed_x, speed_y = paramExtraction(time, stroke, 
                                        lAmbda=15., 
                                        smoothing=True, 
                                        zeroInit=False,
                                        saveParam=False, 
                                        dt=0.005,
                                        vm=vMax,
                                        seqMode=False)
            snrv, snrv2, snrt = computeSNR(P, time, targetPath, speed_x, speed_y)

            if snrv < 17 or snrt < 17:
                P_res, AUC_res, R, _, _, _, _ = paramExtraction(time, R, 
                                            lAmbda=30., 
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

            if 0 < snrv < 20 or 0 < snrt < 20 and P.shape[0] > 1:
                print ("------------ Before optimization: ", snrv, snrv2, snrt, P.shape)
                P = paramRefinement(P, AUC, time, targetPath, speed_x, speed_y, dt=0.005)
                R = getResidualPath(P, time, stroke, dt=0.005)
                P = P[numpy.argsort(P[:,1])] #ascending t0
                snrv, snrv2, snrt = computeSNR(P, time, targetPath, speed_x, speed_y)
            else:
                R = getResidualPath(P, time, stroke, dt=0.005)
            
            print ("------------", snrv, snrv2, snrt, P.shape)

            Ps.append(P)
            Rs.append(R)
            PRs.append(pressure[inds[rowIdx[i]]:inds[rowIdx[i]]+segLen])

        numParams = 0
        for P in Ps:
            numParams += P.shape[0]

        reconPath = []
        for i, P in enumerate(Ps):
            if P.shape[0] == 0:
                reconPathX = Rs[i][:, 0] 
                reconPathY = Rs[i][:, 1] 
            else:
                time = numpy.arange(len(Rs[i]), dtype=numpy.float64) * 0.005
                reconPathX, reconPathY, _, _ = paramReconstruction(P, time, dt=0.005)
                reconPathX += Rs[i][0,0] 
                reconPathY += Rs[i][0,1] 
            reconPath.append(numpy.concatenate([reconPathX[:,None], reconPathY[:,None]], axis=1))
            # show(reconPath[-1], Ss[i])
        reconPath = numpy.concatenate(reconPath, axis=0)

        reconPath = reconPath - numpy.mean(reconPath, axis=0, keepdims=True)
        path = numpy.concatenate(Ss, axis=0)
        path = path - numpy.mean(path, axis=0, keepdims=True)
        # show(reconPath, path)
        
        saveDir = os.path.join(saveDirPrefix, str(key))
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
        numpy.save(os.path.join(saveDir, "Pmatrix_G%d_%d.npy"%(key, idx)), Ps)
        numpy.save(os.path.join(saveDir, "residual_G%d_%d.npy"%(key, idx)), Rs)
        numpy.save(os.path.join(saveDir, "pressure_G%d_%d.npy"%(key, idx)), PRs)

        # Ps2 = numpy.load(os.path.join(saveDir, "Pmatrix_G%d_%d.npy"%(key, idx)), allow_pickle=True)
        # Rs2 = numpy.load(os.path.join(saveDir, "residual_G%d_%d.npy"%(key, idx)), allow_pickle=True)
        # PRs2 = numpy.load(os.path.join(saveDir, "pressure_G%d_%d.npy"%(key, idx)), allow_pickle=True)

        N = numpy.sum((numpy.square(reconPath[:,0] - path[:,0]) + numpy.square(reconPath[:,1] - path[:,1])))
        S = numpy.sum((numpy.square(path[:,0]) + numpy.square(path[:,1])))
        snrt = 10. * math.log10(S / (N + 1e-8))
        SNRt[count] = snrt
        numLN[count] = numParams
        count += 1

print (numpy.mean(numLN))
print (numpy.mean(SNRt))
import pdb
pdb.set_trace()

