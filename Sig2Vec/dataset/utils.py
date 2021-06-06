# -*- coding:utf-8 -*-
import numpy
from scipy import signal

def pathDrop(path, low=0.05, high=0.075):
    l = path.shape[0]
    r = (high - low) * numpy.random.random_sample() + low
    ll = int(r * l)
    idx = numpy.random.choice(numpy.arange(1, l), ll, replace=False)
    path = numpy.delete(path, idx, axis=0)
    # path[idx] = 0
    return path

def diff(x):
    dx = numpy.convolve(x, [0.5,0,-0.5], mode='same'); dx[0] = dx[1]; dx[-1] = dx[-2]
    # dx = numpy.convolve(x, [0.2,0.1,0,-0.1,-0.2], mode='same'); dx[0] = dx[1] = dx[2]; dx[-1] = dx[-2] = dx[-3]
    return dx

def diffTheta(x):
    dx = numpy.zeros_like(x)
    dx[1:-1] = x[2:] - x[0:-2]; dx[-1] = dx[-2]; dx[0] = dx[1]
    temp = numpy.where(numpy.abs(dx)>numpy.pi)
    dx[temp] -= numpy.sign(dx[temp]) * 2 * numpy.pi
    dx *= 0.5
    return dx

def sinusoidalTransform(path):
    # if numpy.random.randint(2):
    #     return path
    MIN = numpy.min(path, axis=0)
    size = numpy.max(path, axis=0) - MIN
    M = size[0]; N = size[1]
    twoPi = 2 * numpy.pi
    alphaA = numpy.random.randint(low=69, high=89, size=2)
    alphaW = numpy.random.rand(2) * 0.5 + 0.5 #(0.5, 1)
    alphaP = numpy.random.rand(2) * twoPi #(0, twoPi)
    Ax = M * 1.0 / alphaA[0] 
    Ay = N * 1.0 / alphaA[1]
    Wx = twoPi / M * alphaW[0]
    Wy = twoPi / N * alphaW[1]
    path[:, 0] += Ax * numpy.sin((path[:,0] - MIN[0]) * Wx + alphaP[0])
    path[:, 1] += Ay * numpy.sin((path[:,1] - MIN[1]) * Wy + alphaP[1])
    return path

class butterLPFilter(object):
    """docstring for butterLPFilter"""
    def __init__(self, highcut=10.0, fs=200.0, order=3):
        super(butterLPFilter, self).__init__()
        nyq = 0.5 * fs
        highcut = highcut / nyq
        b, a = signal.butter(order, highcut, btype='low')
        self.b = b
        self.a = a
    def __call__(self, data):
        y = signal.filtfilt(self.b, self.a, data)
        return y

bf = butterLPFilter(15, 100)

def featExt(pathList, feats, gpnoise=None, dim=2, transform=False):
    for path in pathList:
        p = path[:,dim]
        path = path[:, 0:dim] #(x,y,p)
        path[:,0] = bf(path[:,0])
        path[:,1] = bf(path[:,1])
        # sinusoidalTransform(path)
        dx = diff(path[:, 0]); dy = diff(path[:, 1])
        v = numpy.sqrt(dx**2+dy**2)
        theta = numpy.arctan2(dy, dx)
        cos = numpy.cos(theta)
        sin = numpy.sin(theta)
        dv = diff(v)
        dtheta = numpy.abs(diffTheta(theta))
        logCurRadius = numpy.log((v+0.05) / (dtheta+0.05))
        dv2 = numpy.abs(v*dtheta)
        totalAccel = numpy.sqrt(dv**2 + dv2**2)
        # ddx = diff(dx); ddy = diff(dy)
        # dtotalAccel = diff(totalAccel)
        # dlogCurRadius = diff(logCurRadius)
        feat = numpy.concatenate((dx[:,None], dy[:,None], v[:,None], cos[:,None], sin[:,None], theta[:,None], 
                                  logCurRadius[:,None], totalAccel[:,None], dv[:,None], dv2[:,None], dtheta[:,None], p[:,None]), axis=1).astype(numpy.float32) 
        feat = (feat - numpy.mean(feat, axis=0)) / numpy.std(feat, axis=0)
        ### For finger scenario
        # feat[:,:-1] = (feat[:,:-1] - numpy.mean(feat[:,:-1], axis=0)) / numpy.std(feat[:,:-1], axis=0)
        feats.append(feat.astype(numpy.float32))
    return feats
