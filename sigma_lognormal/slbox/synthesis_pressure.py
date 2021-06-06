# -*- coding:utf-8 -*-
import pdb
import numpy
import copy, time

from . import functions
from scipy import interpolate
from .extractionFirstMode import paramReconstruction

'''
Ref: 1. A sigma-lognormal model-based approach to generating large synthetic online handwriting sample databases.
     2. Dynamic Signature Verification System Based on One Real Signature
'''

admissibleRange = numpy.array([[-0.1000, 0.1000],
                               [-0.0850, 0.0825],
                               [-0.3775/1.5, 0.3950/1.5],
                               [-0.2875, 0.3250],
                               [-0.0150, 0.0300],
                               [-0.0150, 0.0275]])

def padAvergeSpeed(crt, dtype="float32", indicator=2):
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
        # count.append(dist.shape[0])
        count.append(numpy.count_nonzero(dist))
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
                # No padding
                crt_new[i+count, indicator] = 1.01 
                continue
            interv = numpy.array([0 + t * 10 for t in range(l)])
            intervT = interv - interv[0]
            fX = interpolate.interp1d(intervT[[0,-1]], data[:, 0], kind='slinear')
            fY = interpolate.interp1d(intervT[[0,-1]], data[:, 1], kind='slinear')
            intervX = fX(intervT)[1:-1]
            intervY = fY(intervT)[1:-1]
            intervT = interv[1:-1]
            penUp = numpy.zeros((l-2, crt.shape[1]), dtype=dtype)
            penUp[:, 0] = intervX
            penUp[:, 1] = intervY
            # print "Intervpolation time steps:", intervT
            crt_new = numpy.concatenate((crt_new[0:i+count, :], penUp, crt_new[i+count:, :]))
            count = count + l - 2
            crt_new[i+count, indicator] = 1.01 #crt_new[i+1+count, indicator] ##### Restore the pressure value
    return crt_new

def sinusoidalTransform(path, amp=0.05):
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

def synthesis(Ps, Rs, PRs, forgery=True, dtype="float32", padPenUp=True, const=0.4):
    ''' Ps: list of numpy arrays, with each array (maybe be empty) the lognormal parameters for one stroke.
        Rs: residuals of the sigma lognormal fitting.
        PRs: pressure information
        forgery: decide the distortion level to the parameters. 

        Output: Synthesized signatures sampled at 100Hz (while the parameters are extracted at 200Hz).
    '''
    Ps = copy.deepcopy(Ps)
    const = const if forgery else 0
    randTs = []
    randMUs = [] #Cached for alignment of the reconstructed path and its residual.
    for ip, P in enumerate(Ps):
        if P.shape[0] == 0:
            randTs.append(0)
            randMUs.append(0)
            continue
        # randRange = admissibleRange[[0,1,2,3], numpy.random.randint(2, size=(P.shape[0], 4))] 
        # randRange[:, 2] += numpy.sign(randRange[:, 2]) * numpy.random.rand(P.shape[0]) * 0.03
        if forgery:
            rand = (numpy.random.rand(P.shape[0], 4) * 0.25 + const) * admissibleRange[[0,1,2,3], numpy.random.randint(2, size=(P.shape[0], 4))] 
        else:
            rand = (numpy.random.rand(P.shape[0], 4) * 0.25) * admissibleRange[[0,1,2,3], numpy.random.randint(2, size=(P.shape[0], 4))] 
        rand[:, 1] = rand[0, 1]
        randTs.append(rand[0, 1])
        ### 1
        randIdx = numpy.random.randint(0, P.shape[0])
        dt_mu = numpy.exp(P[randIdx, 2] * (rand[randIdx, 2] + 1)) - numpy.exp(P[randIdx, 2])
        rand[:, 2] = numpy.log(numpy.maximum(dt_mu + numpy.exp(P[:, 2]), 1e-3)) / P[:, 2] - 1
        randMUs.append(rand[:, 2])
        ### 2
        # rand[:, 2] = rand[0, 2]
        # randMUs.append(rand[0, 2])
        P[:,[0,1,2,3]] = P[:,[0,1,2,3]] * (rand + 1) #[None,:] 
        # Parameter-dependent variations for theta_s and theta_e seem inappropriate. Use an independent approach here.  
        ### 1
        if numpy.random.randint(2):
            rand = 1.0 * numpy.random.rand(2) + 1.0 * numpy.random.rand(P.shape[0], 2) - 1
            rand[1:, 0] = rand[0:-1, 1] 
        ### 2
        else:
            rand = numpy.random.rand(2) * 2 - 1 
            rand[1] = (rand[0] + rand[1]) / 2
        P[:,[4,5]] = P[:,[4,5]] + (rand * (0.1 + int(forgery) * 0.1))#[None,:] 
        
    reconPath = []
    reconPressure = []
    inds = [0]
    for ip, P in enumerate(Ps):
        if P.shape[0] == 0:
            reconPathX = Rs[ip][:, 0].astype(dtype) 
            reconPathY = Rs[ip][:, 1].astype(dtype)
            reconPR = PRs[ip]
        else:
            ''' A rough first alignment assume mu and sigma are the same.
            '''
            r = randTs[ip]
            dt = numpy.min(P[:, 1]) * abs(r)
            ind_start_init = 0 
            ind_end_init = Rs[ip].shape[0] * (r + 1) #Expansion or contraction.
            if dt < 0: 
                #Left extrapolation, right truncation
                ind_start_init -= numpy.sign(r) * abs(dt) / 0.005
            ''' A rough second alignment considering the changes of mu (and sigma?). 
                Equation: log(t-t0) = ru +- sigma, t = exp(ru +- sigma) + t0
            '''            
            dt_mu = numpy.median(numpy.exp(P[:, 2]) - numpy.exp(P[:, 2] / (randMUs[ip] + 1)))
            ind_start = ind_start_init + numpy.sign(dt_mu) * abs(dt_mu) / 0.005 
            ind_end = ind_end_init + ind_start - ind_start_init
            ind_start = numpy.sign(ind_start) * int(abs(ind_start) + 0.5); ind_end = int(ind_end + 0.5); 
            T = numpy.arange(ind_start, ind_end, dtype=dtype) * 0.005 
            R = functions.cubicSplineInterp(Rs[ip], nPoints=T.shape[0], interp="cubic") #"cubic"
            reconPR = interpPressure(PRs[ip], nPoints=(T.shape[0]+1)//2, interp="cubic").astype(dtype) #"cubic"
            ''' Reconstruct the path and add the residual.
            '''
            reconPathX, reconPathY, reconVelocX, reconVelocY = paramReconstruction(P, T, dt=0.005)
            reconPathX += R[:, 0]  
            reconPathY += R[:, 1] 
        reconPath.append(numpy.concatenate([reconPathX[0::2,None], reconPathY[0::2,None]], axis=1))
        reconPressure.append(reconPR)
        if ip < len(Ps) - 1: #Leave the last stroke (since we don't need padding after it)
            inds.append(reconPath[-1].shape[0])

    reconPath = numpy.concatenate(reconPath, axis=0)
    reconPressure = numpy.concatenate(reconPressure, axis=0)
    reconPath = numpy.concatenate([reconPath, reconPressure[:,None]], axis=1)
    if padPenUp:
        inds = numpy.cumsum(inds)
        indicators = numpy.ones((reconPath.shape[0], 1), dtype=dtype)
        indicators[inds] = 0
        reconPath = numpy.concatenate((reconPath, indicators), axis=1)
        reconPath = padAvergeSpeed(reconPath, dtype, indicator=3)[:,[0,1,2]]

    # sinusoidalTransform(reconPath) #No improvement

    return reconPath





