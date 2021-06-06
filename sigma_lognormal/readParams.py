# -*- coding:utf-8 -*-
import pdb
import os
import numpy
import pickle
import matplotlib.pyplot as plt

from scipy import interpolate, signal, convolve
from slbox import functions, optimization
from slbox.extractionFirstMode import paramExtraction, paramReconstruction

samplingMethod = 0

# D, t0, mu, sigma, theta_s, theta_e 
if samplingMethod == 0:
    '''
    Parameter dependent character level deformation. Determined via visual Turing tests. 
    Deformation level: <0.25: intra-variability; 0.25~0.75: equivalent to including samples from new writers; 
                        0.75~1: unnatural but legible; >1: illegible. 
    Ref: A sigma-lognormal model-based approach to generating large synthetic online handwriting sample databases.
    '''
    admissibleRange = numpy.array([[-0.1000, 0.1000],
                                   [-0.0850, 0.0825],
                                   [-0.3775/1.5, 0.3950/1.5],
                                   # [-0.3775/2.0, 0.3950/2.0],
                                   [-0.2875, 0.3250],
                                   [-0.0150, 0.0300],
                                   [-0.0150, 0.0275]]) 
else:
    '''
    Deformation of D, mu and sigma are parameter dependent. Deformation of t0, theta_s and theta_e are parameter independent. 
    Ref: Dynamic Signature Verification System Based on One Real Signature
    '''
    admissibleRange = numpy.array([0.1000, 0.0250, 0.1000, 0.1000, 0.1000, 0.1000])

def padAvergeSpeed(crt, indicator=2):
    '''Pad the pen-up with average speed lines.
    '''
    rowIdx = numpy.where(crt[:, indicator]==0)[0] #Start point of one stroke
    rowIdx = numpy.pad(rowIdx, (0, 1), mode="constant")
    rowIdx[-1] = crt.shape[0] 
    DIST = []
    count = []
    ### Compute the average speed
    for i in range(len(rowIdx)-1):
        segment = crt[rowIdx[i]:rowIdx[i+1], [0, 1]]
        if segment.shape[0] == 1:
            continue
        inc = segment[1:] - segment[0:-1]
        dist = numpy.sqrt(numpy.sum(inc**2, axis=1))
        DIST.append(numpy.sum(dist))
        count.append(dist.shape[0])
    if len(count) > 0:
        aveInc = sum(DIST) / sum(count)
    crt_new = crt
    count = 0
    crt_new[0, indicator] = 1.01 
    # Skip the first point.
    for i in range(1, crt.shape[0]):
        if crt[i, indicator] == 0: #button_status, start point indicator.
            # Last point of the last line and first point of current line
            data = crt[i-1:i+1, [0, 1]]
            dist = (sum((data[1] - data[0])**2))**0.5
            l = int(dist / aveInc) + 1
            if l <= 2:
                # penUp = numpy.zeros((1, crt.shape[1])); penUp[0, 0:2] = numpy.mean(data, axis=0)
                # crt_new = numpy.concatenate((crt_new[0:i+count], penUp, crt_new[i+count:]))
                # count = count + 1
                # ## If the above three lines are commented, this stroke is then regarded connected to the previous stroke, 
                # ## because there is no zero pressure point in between.
                crt_new[i+count, indicator] = 1.01 
                continue
            interv = numpy.array([0 + t * 10 for t in range(l)])
            intervT = interv - interv[0]
            fX = interpolate.interp1d(intervT[[0,-1]], data[:, 0], kind='slinear')
            fY = interpolate.interp1d(intervT[[0,-1]], data[:, 1], kind='slinear')
            intervX = fX(intervT)[1:-1]
            intervY = fY(intervT)[1:-1]
            intervT = interv[1:-1]
            penUp = numpy.zeros((l-2, crt.shape[1]))
            penUp[:, 0] = intervX
            penUp[:, 1] = intervY
            # print "Intervpolation time steps:", intervT
            crt_new = numpy.concatenate((crt_new[0:i+count, :], penUp, crt_new[i+count:, :]))
            count = count + l - 2
            crt_new[i+count, indicator] = 1.01 
    return crt_new

def centerNormSize(crt, coords=[0, 1]):
    assert len(coords)==2
    pps = crt[:, coords] # x & y coordinates
    
    m = numpy.min(pps, axis=0)
    M = numpy.max(pps, axis=0)
    pps = (pps - (M + m) / 2.0) / numpy.max(M - m) 
    # pps = (pps - (M + m) / 2.0) / (M - m) 
    crt[:, coords] = pps
    return crt

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

def sinusoidalTransform(path):
    MIN = numpy.min(path, axis=0)
    size = numpy.max(path, axis=0) - MIN
    M = size[0]; N = size[1]
    twoPi = 2 * numpy.pi
    alphaA = numpy.random.rand(2) * 0.15 - 0.075
    alphaW = numpy.random.rand(2) * 0.5 + 0.5 #(0.5, 1)
    alphaP = numpy.random.rand(2) * twoPi #(0, twoPi)
    Ax = M * 1.0 * alphaA[0]
    Ay = N * 1.0 * alphaA[1]
    Wx = twoPi / M * alphaW[0]
    Wy = twoPi / N * alphaW[1]
    path[:, 0] += Ax * numpy.sin((path[:,0] - MIN[0]) * Wx + alphaP[0])
    path[:, 1] += Ay * numpy.sin((path[:,1] - MIN[1]) * Wy + alphaP[1])
    return path

# numpy.random.seed(1)
users = pickle.load(open('../data/MCYT_dev.pkl', 'r'))

for key in users.keys()[0:]:
    print (key)
    saveDir = "./params/mcyt_dev_full/%d/"%key
    # numSamples = len(users[key][True])
    for idx, path in enumerate(users[key][True]):
        pressure = path[:,2]
        Ps = numpy.load(os.path.join(saveDir, "Pmatrix_G%d_%d.npy"%(key, idx)), allow_pickle=True)
        Rs = numpy.load(os.path.join(saveDir, "residual_G%d_%d.npy"%(key, idx)), allow_pickle=True)
        if not isinstance(Ps, list): #full path
            Ps = [Ps]
        if not isinstance(Rs, list): #full path
            Rs = [Rs]
        
        randTs = [] #Cached for alignment of the reconstructed path and its residual.
        randMUs = [] #Cached for alignment of the reconstructed path and its residual.
        for ip, P in enumerate(Ps):
            if P.shape[0] == 0:
                randTs.append(0)
                randMUs.append(0)
                continue
            if samplingMethod == 0:
                # Component-wise or stroke-wise?
                # rand = (numpy.random.rand(4) * 0.25 + 0.4) * admissibleRange[[0,1,2,3], numpy.random.randint(2, size=4)] 
                # randTs.append(rand[1])
                rand = (numpy.random.rand(P.shape[0], 4) * 0.25 + 0.0) * admissibleRange[[0,1,2,3], numpy.random.randint(2, size=(P.shape[0], 4))] 
                rand[:, 1] = rand[0, 1]
                randTs.append(rand[0, 1])
                # print (numpy.exp(P[:,2]+1.414*P[:,3]) - numpy.exp(P[:,2]-1.414*P[:,3]))

                ### 1
                randIdx = numpy.random.randint(0, P.shape[0])
                dt_mu = numpy.exp(P[randIdx, 2] * (rand[randIdx, 2] + 1)) - numpy.exp(P[randIdx, 2])
                rand[:, 2] = numpy.log(numpy.maximum(dt_mu + numpy.exp(P[:, 2]), 1e-3)) / P[:, 2] - 1
                randMUs.append(rand[:, 2])
                ### 2
                # rand[:, 2] = rand[0, 2]
                # randMUs.append(rand[0, 2])
                P[:,[0,1,2,3]] = P[:,[0,1,2,3]] * (rand + 1) #[None,:]
                ### 1
                rand = 1.0 * numpy.random.rand(2) + 1.0 * numpy.random.rand(P.shape[0], 2) - 1
                rand[1:, 0] = rand[0:-1, 1] 
                ### 2
                # rand = numpy.random.rand(2) * 2 - 1 
                # rand[1] = (rand[0] + rand[1]) / 2
                P[:,[4,5]] = P[:,[4,5]] + (rand * 0.0000)[None,:] 
            else:
                rand = numpy.random.randn(6)
                randTs.append(rand[1])
                rand[rand<-2] = -2 
                rand[rand>+2] = +2 #Clip at twice the standard variation
                P[:,[0,2,3]] = P[:,[0,2,3]] * (rand[[0,2,3]] * admissibleRange[[0,2,3]] + 1)[None,:]
                P[:,[1,4,5]] = P[:,[1,4,5]] + (rand[[1,4,5]] * admissibleRange[[1,4,5]])[None,:]

        reconPath = []
        reconPath2 = []
        inds = [0]
        for ip, P in enumerate(Ps):
            if P.shape[0] == 0:
                reconPathX = Rs[ip][:, 0]
                reconPathY = Rs[ip][:, 1]
            else:
                R = Rs[ip]
                r = randTs[ip]
                ''' A rough first alignment assume mu and sigma are the same.
                '''
                if samplingMethod == 0:
                    dt_t0 = numpy.min(P[:, 1]) * abs(r)
                    ind_start_init = 0 
                    ind_end_init = R.shape[0] * (r + 1) #Expansion or contraction.
                    if dt_t0 < 0:
                        #Update start point of the parameterized curve.
                        ind_start_init -= numpy.sign(r) * abs(dt_t0) / 0.005 
                else:
                    dR = abs(admissibleRange[1] * r) / 0.005 #Shifts of all strokes
                    ind_start_init = 0 + numpy.sign(r) * dR
                    ind_end_init = R.shape[0] + numpy.sign(r) * dR
                ''' A rough second alignment considering the changes of mu and sigma. 
                    Equation: log(t-t0) = ru +- sigma, t = exp(ru +- sigma) + t0
                '''
                dt_mu = numpy.median(numpy.exp(P[:, 2]) - numpy.exp(P[:, 2] / (randMUs[ip] + 1)))
                ind_start = ind_start_init + numpy.sign(dt_mu) * abs(dt_mu) / 0.005 
                ind_end = ind_end_init + ind_start - ind_start_init
                ind_start = numpy.sign(ind_start) * int(abs(ind_start) + 0.5); ind_end = int(ind_end + 0.5); 
                time = numpy.arange(ind_start, ind_end, dtype="float32") * 0.005 
                R = functions.cubicSplineInterp(R, nPoints=time.shape[0], interp="cubic") 
                ''' Reconstruct the path and add the residual.
                '''
                reconPathX, reconPathY, reconVelocX, reconVelocY = paramReconstruction(P, time, dt=0.005)
                reconPathX2 = reconPathX.copy()
                reconPathY2 = reconPathY.copy()
                reconPathX += R[:, 0]  #Original start points.
                reconPathY += R[:, 1] 
            reconPath.append(numpy.concatenate([reconPathX[0::2,None], reconPathY[0::2,None]], axis=1))
            reconPath2.append(numpy.concatenate([reconPathX2[0::2,None], reconPathY2[0::2,None]], axis=1))
            if ip < len(Ps) - 1:
                inds.append(reconPath[-1].shape[0])

        reconPath = numpy.concatenate(reconPath, axis=0)
        reconPath2 = numpy.concatenate(reconPath2, axis=0)
        # angle = (numpy.random.rand(1) * 2 - 1) * 0.1
        # cos = numpy.cos(angle); sin = numpy.sin(angle)
        # reconPath = numpy.dot(reconPath, numpy.array([[cos, sin], [-sin, cos]], dtype=numpy.float32))

        # Pad the pen-ups
        if len(inds) > 1:
            inds = numpy.cumsum(inds)
            indicators = numpy.ones((reconPath.shape[0], 1), dtype="float32")
            indicators[inds] = 0
            reconPath = numpy.concatenate((reconPath, indicators), axis=1)
            reconPath = padAvergeSpeed(reconPath)[:, [0, 1]]

        pressure = interpPressure(pressure, nPoints=reconPath.shape[0], interp="linear") 

        # pos = numpy.where(pressure>0)[0]
        plt.plot(reconPath[pos,0], reconPath[pos,1]+0.7, c="r", marker="o")
        plt.plot(reconPath[:,0], reconPath[:,1]+0.7, c="b")
        plt.plot(path[:,0], path[:,1], c="b") #numpy.where(path[:, 3])[0]
        # idx = numpy.where(numpy.sum(path[0:-1, 0:2] == path[1:, 0:2], axis=1)==2)[0]
        # plt.scatter(path[idx, 0], path[idx, 1], marker="o")
        # dx = numpy.convolve(path[:, 0], [0.5,0,-0.5], mode='same'); dx[0] = dx[1]; dx[-1] = dx[-2]
        # dy = numpy.convolve(path[:, 1], [0.5,0,-0.5], mode='same'); dy[0] = dy[1]; dy[-1] = dy[-2]
        # v = numpy.sqrt(dx**2+dy**2)
        # plt.plot(dx, color='r')
        # plt.plot(dy, color='g')
        # plt.plot(v, color='b')
        # reconDx = numpy.convolve(reconPath[:, 0], [0.5,0,-0.5], mode='same'); reconDx[0] = reconDx[1]; reconDx[-1] = reconDx[-2]
        # reconDy = numpy.convolve(reconPath[:, 1], [0.5,0,-0.5], mode='same'); reconDy[0] = reconDy[1]; reconDy[-1] = reconDy[-2]
        # reconV = numpy.sqrt(reconDx**2+reconDy**2)
        # plt.plot(reconDx+0.5, "--", color='r')
        # plt.plot(reconDy+0.5, "--", color='g')
        # plt.plot(reconV+0.5, "--", color='b')

        plt.show()



