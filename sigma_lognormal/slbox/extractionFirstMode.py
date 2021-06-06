# -*- coding:utf-8 -*-
import numpy, pdb
import math, os
import matplotlib.pyplot as plt
from scipy import convolve, signal, ndimage

from . import functions
from . import RX0
from . import angleEst
from . import optimization

global Debug; Debug = False
global verbose; verbose = False
# On coordinates
global velocKernel; velocKernel = [0.2,0.1,0,-0.1,-0.2] 
# global velocKernel; velocKernel = [0.5,0,-0.5] #Lead to noisy velocity. 
# On velocity
global accelKernel; accelKernel = [0.5,0,-0.5]
# Padding to the velocity
global pad; pad = len(velocKernel) // 2
# A sampling rate of 100 Hz. Suggested sampling rates: 100 Hz or 200 Hz.
global deltaT; deltaT = 0.01 
global nfs; nfs = 1

'''
Ref: 1. Development of a Sigma-Lognormal representation on on-line signatures (Majar)
     2. Towards an Automatic On-Line Signature Verifier Using Only One Reference Per Signer
     3. iDeLog: Iterative Dual Spatial and Kinematic Extraction of Sigma-Lognormal Parameters
'''

def criterionMode1():
    return True

def scanInitials():
    return True

def calculateSNRvxy(time, path, param, tinf1idx, tinf2idx):
    vxn = convolve(path[tinf1idx-pad:tinf2idx+pad+1, 0], velocKernel, mode="valid")
    vyn = convolve(path[tinf1idx-pad:tinf2idx+pad+1, 1], velocKernel, mode="valid")
    velocity_n = numpy.sqrt(vxn**2 + vyn**2)

    D, t0, mu, sigma, theta_s, theta_e = param
    ln = functions.lognormal(mu, sigma, loc=t0)
    T = time[tinf1idx-pad:tinf2idx-pad+1]
    velocity_a = ln.eval(T) * D
    vxa, vya = angleEst.estimateVxy(velocity_a, T, *param)

    N = numpy.sum((numpy.square(vxn - vxa) + numpy.square(vyn - vya)))
    S = numpy.sum((numpy.square(vxn) + numpy.square(vyn)))
    return 10. * math.log10(S / (N + 1e-8)), S, N

def calculateSNRvt(time, path, param, tinf1idx, tinf2idx):
    vxn = convolve(path[tinf1idx-pad:tinf2idx+pad+1, 0], velocKernel, mode="valid")
    vyn = convolve(path[tinf1idx-pad:tinf2idx+pad+1, 1], velocKernel, mode="valid")
    velocity_n = numpy.sqrt(vxn**2 + vyn**2)

    D, t0, mu, sigma, theta_s, theta_e = param
    ln = functions.lognormal(mu, sigma, loc=t0)
    T = time[tinf1idx-pad:tinf2idx-pad+1]
    velocity_a = ln.eval(T) * D
    
    N = numpy.sum(numpy.square(velocity_a - velocity_n))
    S = numpy.sum(numpy.square(velocity_n))
    return 10 * math.log10(S / (N + 1e-8)), S, N

class pathOps(object):
    def __init__(self, time, path, smoothing, pad=2, zeroInit=True):
        super(pathOps, self).__init__()
        self.pad = pad
        self.len = path.shape[0]
        self.STARTIDX = pad
        self.ENDIDX = path.shape[0] - 1 + pad
        self.zeroInit = zeroInit
        self.time, self.path = self.init(time, path, smoothing, pad)
        self.speed_x = convolve(self.path[:,0], velocKernel, mode="valid")
        self.speed_y = convolve(self.path[:,1], velocKernel, mode="valid")
        self.velocity = numpy.sqrt(self.speed_x**2 + self.speed_y**2)

    def init(self, time, path, smoothing, pad):
        # Padding or smoothing first may be slight different. 
        if smoothing: #Note: inplace modification!
            pathX = functions.butter_lowpass_filter(path[:,0], highcut=10, fs=100*nfs, order=3)
            pathY = functions.butter_lowpass_filter(path[:,1], highcut=10, fs=100*nfs, order=3)
            path = numpy.concatenate((pathX[:,None], pathY[:,None]), axis=1)
        if self.zeroInit:
            # Pad the ends with zeros according to the used convolutional kernel.
            path = numpy.pad(path, pad_width=((2*pad,2*pad),(0,0)), mode="edge")
            time = numpy.pad(time, pad_width=((pad,pad),), mode="edge")
            # Extrapolation
            time[0:pad] = 2*time[pad]-time[2*pad:pad:-1]
            time[-pad:] = 2*time[-pad-1]-time[-pad-2:-2*pad-2:-1]
            self.ENDIDX += 2 * pad
        else:
            path = numpy.pad(path, pad_width=((pad,pad),(0,0)), mode="edge")
        return time, path

    def subtractStroke(self, param):
        D, t0, mu, sigma, theta_s, theta_e = param
        ln = functions.lognormal(mu, sigma, loc=t0)
        pos = numpy.where(self.time>=t0)[0][0] #Assume t0 < self.time[-1]
        reconVeloc = ln.eval(self.time[pos:]) * D 
        lx, ly = angleEst.estimateLxy(self.time[pos:], *param, shift=[0, 0], dt=deltaT)
        reconPath = numpy.concatenate((lx[:,None], ly[:,None]), axis=1)
        
        pad = self.pad
        self.path[0:pad+pos] = self.path[0:pad+pos] - reconPath[0:1] #Acutally zeros
        self.path[pad+pos:-pad] = self.path[pad+pos:-pad] - reconPath
        self.path[-pad:] = self.path[-pad:] - reconPath[-1:]
        
    def addStroke(self, param):
        D, t0, mu, sigma, theta_s, theta_e = param
        ln = functions.lognormal(mu, sigma, loc=t0)
        pos = numpy.where(self.time>=t0)[0][0] #Assume t0 < self.time[-1]
        reconVeloc = ln.eval(self.time[pos:]) * D 
        lx, ly = angleEst.estimateLxy(self.time[pos:], *param, shift=[0, 0], dt=deltaT)
        reconPath = numpy.concatenate((lx[:,None], ly[:,None]), axis=1)
        
        pad = self.pad
        self.path[0:pad+pos] = self.path[0:pad+pos] + reconPath[0:1] #Acutally zeros
        self.path[pad+pos:-pad] = self.path[pad+pos:-pad] + reconPath
        self.path[-pad:] = self.path[-pad:] + reconPath[-1:]
        
    def resetPath(self, path):
        if path.shape[0] == self.path.shape[0]:
            self.path = path
        else:
            raise ValueError("Inappropriate path for reset!")

    def resetVelocity(self):
        self.speed_x = convolve(self.path[:,0], velocKernel, mode="valid")
        self.speed_y = convolve(self.path[:,1], velocKernel, mode="valid")
        self.velocity = numpy.sqrt(self.speed_x**2 + self.speed_y**2)

    def finalPath(self):
        if self.zeroInit:
            return self.path[2*self.pad:-2*self.pad]
        return self.path[self.pad:-self.pad]

def _extractLN(pathOps, t3prevIdx, t1prevIdx, thresh, win, seqMode=False, localOptim=False):
    '''
    A sliding window method for the sigma-lognormal parameter extraction. Identify strokes + RX0.
    A smaller win is not necessarily faster due to more frequent addStroke and subtractStroke operations.
    Generally win in [50, 100] works well.
    '''
    time = pathOps.time
    path = pathOps.path
    pad = pathOps.pad
    ENDIDX = pathOps.ENDIDX
    startIdx = t1prevIdx-pad  
    endIdx = min(t1prevIdx+win+pad, path.shape[0])
    speed_x = convolve(path[startIdx:endIdx, 0], velocKernel, mode="valid") #length win+2*pad, 2*pad boarder points
    speed_y = convolve(path[startIdx:endIdx, 1], velocKernel, mode="valid") #length win+2*pad, 2*pad boarder points
    velocity = numpy.sqrt(speed_x**2 + speed_y**2)

    acceleration = convolve(numpy.pad(velocity, pad_width=1, mode="edge"), accelKernel, mode="valid")
    velocity = numpy.pad(velocity, pad_width=(0,1), mode="constant") #Pad another zero at the end for maximum comparison
    acceleration = numpy.pad(acceleration, pad_width=(0,1), mode="constant") #Pad another zero at the end just to alignwith the velocity signal
    
    if Debug:
        pathOps.resetVelocity()
        plt.plot(pathOps.velocity)
        plt.title("Full velocity.")
        plt.show()
        plt.plot(velocity)
        plt.plot(acceleration)
        plt.title("The current velocity profile to be estimated")
        plt.show()

    if seqMode:
        ''' This setup detects the lognormal in sequence.'''
        t3Idx = t3prevIdx - t1prevIdx 
    else:
        ''' This setup allows that the next lognormal component can be put before the previous one.'''
        t3Idx = 0
    while(t3Idx < len(velocity) - 1):
        # velocity[-1] = 0 #For warpped maximum comparison.
        if (velocity[t3Idx] >= velocity[t3Idx+1]) and (velocity[t3Idx] > velocity[t3Idx-1]): 
            vm = velocity[t3Idx]
            if vm <  thresh or t3Idx == 0: 
                if verbose: print ("Invalid local maximum point at %d!"%t3Idx)
                t3Idx += 2
                continue
            t1Idx = 0
            t5Idx = len(velocity) - 2        
            # velocity[-1] = numpy.inf #For warpped minimum comparison.
            for t1Idx in range(t3Idx - 1, -1, -1):
                v1 = velocity[t1Idx]
                if ((v1 < velocity[t1Idx+1]) and (v1 <= velocity[t1Idx-1])) or v1 < 0.012 * vm:
                    break
            for t5Idx in range(t3Idx + 1, len(velocity) - 1):
                v5 = velocity[t5Idx]
                if ((v5 <= velocity[t5Idx+1]) and (v5 < velocity[t5Idx-1])) or v5 < 0.012 * vm:
                    break
            if t5Idx == t3Idx: #==len(velocity)-2
                if t3Idx+t1prevIdx != ENDIDX: #A lift at the end may be part of the next lognormal.
                    return None, None, [t1Idx+t1prevIdx, t1Idx+t1prevIdx, t3Idx+t1prevIdx, t3Idx+t1prevIdx]
                else: #If it comes to the end, just ignore this lift.
                    return None, None, [ENDIDX]*4
            if t5Idx - t1Idx < 4*nfs: 
                if verbose: print ("Noisy spike at %d!"%t3Idx)
                t3Idx = t5Idx + 1
                continue
            t2Idx = t1Idx
            for t2Idx in range(t3Idx - 1, t1Idx - 1, -1):
                if (acceleration[t2Idx] > acceleration[t2Idx+1]) and (acceleration[t2Idx] >= acceleration[t2Idx-1]):
                    break
            t4Idx = t5Idx        
            for t4Idx in range(t3Idx + 1, t5Idx + 1):
                if (acceleration[t4Idx] <= acceleration[t4Idx+1]) and (acceleration[t4Idx] < acceleration[t4Idx-1]):
                    break
            ''' Visualize the characteristic points for debugging
            '''
            if Debug:
                plt.plot(time[startIdx:endIdx-2*pad], velocity[:-1])
                plt.scatter(time[startIdx+t1Idx], velocity[t1Idx], c='b', marker="*")
                plt.scatter(time[startIdx+t2Idx], velocity[t2Idx], c='g', marker="*")
                plt.scatter(time[startIdx+t3Idx], velocity[t3Idx], c='r', marker="*")
                plt.scatter(time[startIdx+t4Idx], velocity[t4Idx], c='g', marker="*")
                plt.scatter(time[startIdx+t5Idx], velocity[t5Idx], c='b', marker="*")
                plt.title("Characteristic points.")
                plt.show()

            ''' Full path, should add the corresponding offsets
            '''
            _path = path[t2Idx+t1prevIdx:t4Idx+t1prevIdx+1]
            _time = time[t2Idx+t1prevIdx-pad:t4Idx+t1prevIdx-pad+1] #-pad: alignment with the padded path
            ''' Local path, the first point corresponding the t1prevIdx-th point of the full path
            '''
            _speed_x = speed_x[t2Idx:t4Idx+1]
            _speed_y = speed_y[t2Idx:t4Idx+1]
            _speed = numpy.concatenate((_speed_x[:,None], _speed_y[:,None]), axis=1)
            _veloc = velocity[t2Idx:t4Idx+1]
            
            auc = numpy.sum(velocity[t1Idx:t5Idx+1])-0.5*velocity[t1Idx]-0.5*velocity[t5Idx]

            idxs = [0, t3Idx - t2Idx, t4Idx - t2Idx]
            param, t, v = RX0.RX0(_veloc, _time, idxs, constrainParam=True)
            D, t0, mu, sigma = param

            theta_s, theta_e, _ = angleEst.estimateThetaSE2(_speed, _time, idxs, D, t0, mu, sigma)
            param = param + [theta_s, theta_e]
            if localOptim:
                if not (idxs[-1] < min(2*nfs, 4)): 
                    _param = optimization.localOptim(_time, _path, _speed, t[1], v[1], idxs, param, dt=deltaT)
                    # Minimum variance 0.1, maximum variance 0.55, to prevent disgusting local optima
                    if _param[3] > 0.1 and _param[3] < 0.55: 
                        param = _param
                else:
                    if verbose: print ("Warning! This is maybe a noisy spike!")

            ''' Visualize the reconstruted velocity profile for debugging
            '''
            if Debug:
                D, t0, mu, sigma, theta_s, theta_e = param
                ln = functions.lognormal(mu, sigma, loc=t0)
                reconVeloc = ln.eval(_time) * D 
                plt.plot(_time, reconVeloc, c='g', marker="o")
                plt.plot(_time, _veloc, c='r', marker="o")
                plt.title("Original velocity profile (red) and the reconstruted one (green).")
                plt.show()
                print ([t1Idx+t1prevIdx, t2Idx+t1prevIdx, t3Idx+t1prevIdx, t4Idx+t1prevIdx])
            return param, auc, [t1Idx+t1prevIdx, t2Idx+t1prevIdx, t3Idx+t1prevIdx, t4Idx+t1prevIdx]
        else:
            t3Idx += 1

    return None, None, [t1prevIdx+len(velocity)-2]*4

def paramExtraction(time, path, lAmbda=15., 
            smoothing=True, 
            localOptim=False,
            globalOptim=False,
            zeroInit=True, 
            seqMode=False, 
            saveParam=False, 
            saveDir="", 
            suffix="", 
            vm=None, 
            win=60, 
            dt=None):
    if dt:
        if not (dt == 0.01 or dt == 0.005):
            raise ValueError("dt should be 0.01 or 0.005.")
        global deltaT; deltaT = dt
        global nfs; nfs = int(0.01 / deltaT + 0.5)
    # Make the velocity profile start and end with zeros. May lead to sharp and overlapped components due to a sudden drop in velocity.
    # The velocity can be further smoothed using a low-pass filter
    PATH = pathOps(time, path, smoothing, pad, zeroInit)

    _path = PATH.path.astype(time.dtype, copy=True)
    _speed_x = PATH.speed_x.astype(time.dtype, copy=True)
    _speed_y = PATH.speed_y.astype(time.dtype, copy=True)

    P = [[0,0,0,0,0,0]] * 300
    I = 0
    J = -1
    AUC = [0] * 300
    t1IdxPrev = PATH.pad
    t3IdxPrev = PATH.pad
    if not vm:
        vm = numpy.max(PATH.velocity) 
    thresh = vm / lAmbda
    SNRmin = 20
    Imax = 2

    while(scanInitials()):
        param, auc, idxs = _extractLN(PATH, t3IdxPrev, t1IdxPrev, thresh, win, seqMode, localOptim)
        if param:
            J = 0
            P[J] = param
            AUC[J] = auc
            break
        elif idxs[0] == PATH.ENDIDX:
            break
        else:
            t3IdxPrev = idxs[2]; t1IdxPrev = idxs[0]
            if t3IdxPrev - t1IdxPrev > win / 2:
                win = win + 20
                if verbose: print ("Increasing window.")
    t1Idx = idxs[0]; t3Idx = idxs[2]
    while(idxs[0] < PATH.ENDIDX):
        if verbose: print ("SubtractStroke J.")
        PATH.subtractStroke(P[J])
        if verbose: print ("Estimating J+1...")
        # Note: The next lognormal component candidate is searched staring from t1Idx, i.e., the start point of the previous component.
        # While it works much of the time, it may get stuck when the previous component is baddly optimized, leading to J+1 = J ~= 0 or
        # an oscillation.
        param, auc, idxs = _extractLN(PATH, t3Idx, t1Idx, thresh, win, seqMode, localOptim)
        if param:
            P[J+1] = param
            AUC[J+1] = auc
        elif idxs[0] == PATH.ENDIDX:
            break
        else: 
            # Two possibles situations: 1. No peak in the current window. 2. A lift in the end.
            t3Idx = idxs[2]; t1Idx = idxs[0]
            if t3Idx - t1Idx > win / 2:
                win = win + 20
                if verbose: print ("Increasing window.")
            if verbose: print ("AddStroke J.")
            PATH.addStroke(P[J])
            continue
        if verbose: print ("AddStroke J.")
        PATH.addStroke(P[J])
        if verbose: print ("SubtractStroke J+1.")
        PATH.subtractStroke(P[J+1])
        if verbose: print ("Estimating J...")
        param, auc, idxs = _extractLN(PATH, t3IdxPrev, t1IdxPrev, thresh, win, seqMode, localOptim)
        if param:
            P[J] = param
            AUC[J] = auc
            # Do not update the indexes t1Idx and t3Idx as it may leave some components and lead to bugs:
            # the original component is explained away and the new J-th component is indeed located after 
            # the (J+1)-th component. XXXXX t3Idx = idxs[2]; t1Idx = idxs[0] XXXXX
        else: 
            #The (J+1)-th component explains away the J-th component
            print ("++++++++++++++++")
            PATH.addStroke(P[J+1])
            P[J] = P[J+1]
            P[J+1] = [0,0,0,0,0,0]
            AUC[J] = AUC[J+1]
            AUC[J+1] = 0
            t3IdxPrev = t3Idx; t1IdxPrev = t1Idx #Update the start point.
            continue
        snr, _, noise = calculateSNRvxy(PATH.time, PATH.path, P[J], idxs[1], idxs[3])
        if verbose: print ("~~~~~~~~~~~~SNR: %f~~~~~~~~~~~~"%snr)
        if (snr >= SNRmin) or (I >= Imax):
            if verbose: print ("AddStroke J+1.")
            PATH.addStroke(P[J+1])
            if verbose: print ("SubtractStroke J.")
            PATH.subtractStroke(P[J])
            if verbose: print ("Estimating J+1...")
            param, auc, idxs = _extractLN(PATH, t3Idx, t1Idx, thresh, win, seqMode, localOptim) 
            if param:
                P[J+1] = param
                AUC[J+1] = auc
            elif idxs[0] == PATH.ENDIDX:
                break
            else: #The refined J-th component explains away the (J+1)-th component
                t3Idx = idxs[2]; t1Idx = idxs[0]
                if t3Idx - t1Idx > win / 2:
                    win = win + 20
                    if verbose: print ("Increasing window.")
                if verbose: print ("AddStroke J.")
                PATH.addStroke(P[J]) 
                continue
            if verbose: print ("Moving to the next mode, i.e. J <= J+1. Reset I.")
            J += 1; I = 0
            t3IdxPrev = t3Idx; t1IdxPrev = t1Idx
            t1Idx = idxs[0]; t3Idx = idxs[2]
        else:
            if snr < 1:
                print ("~~~~~~~~~~~~~~")
            if verbose: print ("I <= I+1. AddStroke J+1.")
            PATH.addStroke(P[J+1])
            I += 1

    if J == -1:
        P = numpy.zeros((0,6), dtype=time.dtype)
        AUC = numpy.zeros((0,), dtype=time.dtype)
    else:
        P = numpy.array(P[0:J+1], dtype=time.dtype)
        AUC = numpy.array(AUC[0:J+1], dtype=time.dtype)
        AUC = AUC[numpy.argsort(P[:,1])]
        #ascending t0. Or sort by the peak index. Ideally the peak index should be in accordance with t0.
        P = P[numpy.argsort(P[:,1])] 
  
    if globalOptim:
        speed = numpy.concatenate((PATH.speed_x[:,None], PATH.speed_y[:,None]), axis=1)
        PATH.resetPath(_path)
        print ("Global optimization...")
        P = optimization.globalOptim(PATH.time, PATH.path[PATH.pad:-PATH.pad], speed, P, dt=deltaT)
        for J in range(len(P)):
            PATH.subtractStroke(P[J])

    if saveParam:
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
        numpy.save(os.path.join(saveDir, "Pmatrix%s.npy"%suffix), P)
        numpy.save(os.path.join(saveDir, "residual%s.npy"%suffix), PATH.finalPath())

    _path = _path[PATH.pad:-PATH.pad]
    if zeroInit:
        _speed_x = _speed_x[PATH.pad:-PATH.pad]
        _speed_y = _speed_y[PATH.pad:-PATH.pad]
        _path = _path[PATH.pad:-PATH.pad]

    R = PATH.finalPath().astype(time.dtype, copy=True) #Copy of the array
    del PATH
    return P, AUC, R, vm, _path, _speed_x, _speed_y

def paramReconstruction(params, time, dt=None, plot=False):
    if dt:
        if not (dt == 0.01 or dt == 0.005):
            raise ValueError("dt should be 0.01 or 0.005.")
        global deltaT; deltaT = dt
        global nfs; nfs = int(0.01 / deltaT + 0.5)
    reconVelocX = numpy.zeros(len(time), dtype=time.dtype)
    reconVelocY = numpy.zeros(len(time), dtype=time.dtype)
    pathx = pathy = numpy.zeros(len(time), dtype=time.dtype)
    # targets = numpy.zeros((len(params)+1, 2), dtype=time.dtype)
    # if plot:
    #     fig = plt.figure()
    #     axes = plt.subplot(211)
    for idx, param in enumerate(params):
        D, t0, mu, sigma, theta_s, theta_e = param
        ln = functions.lognormal(mu, sigma, loc=t0)
        _veloc = ln.eval(time) * D 
        _velocx, _velocy = angleEst.estimateVxy(_veloc, time, *param)
        reconVelocX += _velocx
        reconVelocY += _velocy
        lx, ly = angleEst.estimateLxy(time, *param, dt=deltaT, shift=[0, 0])
        pathx = pathx + lx 
        pathy = pathy + ly 
        # targets[idx, 0] = targets[idx-1, 0] + lx[-1]
        # targets[idx, 1] = targets[idx-1, 1] + ly[-1]
        # if plot:
        #     axes.plot(_veloc, c='g', linestyle="--")
    # if plot:
    #     axes.plot(_veloc, c='g', linestyle="--", label="virtual strokes")

    return pathx, pathy, reconVelocX, reconVelocY

