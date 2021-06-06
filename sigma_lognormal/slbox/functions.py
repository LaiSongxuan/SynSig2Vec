# -*- coding:utf-8 -*-
import numpy
import sympy
import heapq
from scipy.special import erf, erfinv
from scipy.stats import lognorm
from scipy import convolve, signal, ndimage
from scipy.interpolate import interp1d

def findTm(velocity):
    '''
    Find the maximum value of a single log-normal.
    '''
    return heapq.nlargest(1, enumerate(velocity), key=lambda x: x[1])[0][0]

def findTinf(velocity, time):
    '''
    Find the maximum and minimum values of the acceleration signal of a single log-normal.
    '''
    kernel = [1, 0, -1]
    dV = convolve(velocity, kernel, 'same'); dV[0] = dV[1]; dV[-1] = dV[-2]  
    inf1Idx = heapq.nlargest(1, enumerate(dV), key=lambda x: x[1])[0][0] # dV / dT
    inf2Idx = heapq.nsmallest(1, enumerate(dV), key=lambda x: x[1])[0][0] # dV / dT

    return [inf1Idx, inf2Idx]

def find3sigmaPoint(velocity, vm):
    '''
    Find the 3-sigma points of a single log-normal.
    '''
    threshold = 0.01 * vm
    inds = signal.medfilt(velocity >= threshold, 3)
    
    idx = numpy.where(inds)[0]
    tminIdx = idx[0] 
    tmaxIdx = idx[-1]

    return [tminIdx, tmaxIdx]

def findTinfDeltaLN(velocity, vm):
    dV = convolve(velocity, [1, 0, -1], 'same'); dV[0] = dV[-1] = 0 #dV[0] = dV[1]; dV[-1] = dV[-2] 
    
    peaks, valleys = _robustExtremums(dV)
    peaks_v = [velocity[v] for v in peaks]
    valleys_v = [velocity[v] for v in valleys]

    if peaks_v[0] < 0.0618 * vm: #Left
        return peaks[-1], valleys[-1]
    elif valleys_v[-1] < 0.0618 * vm: #Right
        return peaks[0], valleys[0]
    else:
        return peaks[0], valleys[-1]

def runDistPoint(D, sigma, point=1):
    a = 0 if point == 1 else \
        D if point == 5 else \
        D / 2. * (1 + erf(-(1.5*sigma+numpy.sqrt(0.25*sigma**2+1)) * 0.707107)) if point == 2 else \
        D / 2. * (1 + erf(- sigma  * 0.707107)) if point == 3 else \
        D / 2. * (1 + erf(-(1.5*sigma-numpy.sqrt(0.25*sigma**2+1)) * 0.707107)) if point == 4 else numpy.nan
    return a

def runDist(t, D, t0, mu, sigma):
    t = t - t0; t[t<=0] = 0 #Assume numpy array
    dist = D / 2. * (1 + erf((numpy.log(t) - mu) / sigma * 0.707107))
    return dist

def runDistInv(dist, D, t0, mu, sigma):
    ''' Inverse the runDist function.
    '''
    t = numpy.exp(1.414214 * sigma *  erfinv(2 * dist / D - 1) + mu)
    return t

def velocAtDist(dist, D, t0, mu, sigma):
    ''' Substitude the result of runDistInv into the lognormal, hence get the velocity at a specific running distance.
    '''
    t = numpy.exp(1.414214 * sigma *  erfinv(2 * dist / D - 1) + mu)
    v = D * numpy.exp(-1.414214 * sigma * erfinv(2 * dist - 1)) / (2.506628 * sigma * t) #2.506628: sqrt(2pi)
    return v

def cheby1_lowpass_filter(highcut, fs, order=3, rp=1):
    nyq = 0.5 * fs
    highcut = highcut / nyq
    b, a = signal.cheby1(order, rp, highcut, btype='low')
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass_filter(data, highcut=10.0, fs=100.0, order=3):
    nyq = 0.5 * fs
    highcut = highcut / nyq
    b, a = signal.butter(order, highcut, btype='low')
    # https://dsp.stackexchange.com/questions/19084/applying-filter-in-scipy-signal-use-lfilter-or-filtfilt
    y = signal.filtfilt(b, a, data)
    return y

def SavgolFiltering(velocity, win=7, order=1):
    return signal.savgol_filter(velocity, win, order)

def GaussianSmoothing(velocity, sigma=1.5, truncate=3.):
    return ndimage.filters.gaussian_filter1d(velocity, sigma, truncate)

def cubicSplineInterp(path, nfs=2, nPoints=None, interp="cubic"):
    assert isinstance(nfs, int)
    if nfs == 1 or path.shape[0] <= 1 or path.shape[0] == nPoints:
        return path
    '''Interpolate the path by to 1/(fs*nfs) Hz, fs is the base frequency.'''
    px = path[:, 0]
    py = path[:, 1]
    # times = numpy.arange(len(path))
    # times_interp = numpy.arange(len(path)-1, step=1/nfs) 
    times = numpy.linspace(0, len(path)-1, num=len(path), endpoint=True)
    if nPoints is None:
        times_interp = numpy.linspace(0, len(path)-1, num=1+(len(path)-1)*nfs, endpoint=True)
    else:
        times_interp = numpy.linspace(0, len(path)-1, num=nPoints, endpoint=True)
    if len(path) > 3:
        interp = "cubic"
    else:
        interp = "linear"
    import pdb
    fx = interp1d(times, px, kind=interp)
    fy = interp1d(times, py, kind=interp)
    px_interp = fx(times_interp)    
    py_interp = fy(times_interp)    
    path_interp = numpy.concatenate([px_interp[:,None], py_interp[:,None]], axis=1)
    return path_interp

def _robustExtremums(signal):
    grad = convolve(signal, [1, -1], 'valid')

    #Checking for sign-flipping
    S = numpy.sign(grad)
    dS = convolve(S, [1, -1], 'same'); dS[0] = 0
    # valleys = numpy.concatenate([numpy.where(dS==2)[0], numpy.where(dS==1)[0]]) 
    # valleys_v = [abs(velocity[v]) for v in valleys]
    # peaks = numpy.concatenate([numpy.where(dS==-2)[0], numpy.where(dS==-1)[0]])
    # peaks_v = [abs(velocity[v]) for v in peaks]

    ''' Detect two adjacent valleys and remove the first one:
    1) dS: 0 0 0 1 1 0 0 1 0 -> ddS: 0 0 0 1 0 0/-1 0 1 0/-1 -> 0 0 0 1 0 0 0 1 0
    2) dS: 0 1 1 -1 -1 1 1 0 -> ddS: 0 1 0 0/-2 0 2 0 0/-1 -> 0 1 0 0 0 1 0 0
    '''
    ddS = convolve(dS, [1, -1], 'same'); ddS[0] = 0; ddS[ddS<0] = 0; ddS[ddS>0] = 1 
    ddS = dS * ddS
    valleys = numpy.concatenate([numpy.where(dS==2)[0], numpy.where(ddS==1)[0]]) 
    '''Detect two adjacent peaks and remove the first one
    1) 0 0 0 1 1 0 -1 -1 0 -> 0 0 0 0/1 0 -1 -1 0 0/1 -> 0 0 0 0 0 (1) 1 0 0
    2) 0 1 1 -1 -1 1 1 0 -> 0 0/1 0 -2 0 0/2 0 -1 -> 0 0 0 1 0 0 0 0
    '''
    ddS = convolve(dS, [1, -1], 'same'); ddS[0] = 0; ddS[ddS>0] = 0; ddS[ddS<0] = 1
    ddS = dS * ddS
    peaks = numpy.concatenate([numpy.where(dS==-2)[0], numpy.where(ddS==-1)[0]])
    
    return peaks, valleys

def findExtremums(velocity):
    '''
    Find the maximum and minimum values of a signal.
    '''
    peaks, valleys = _robustExtremums(velocity)

    return peaks, valleys

def findTinfs(velocity):
    '''
    Find the inflexion points of a signal.
    '''
    dV = convolve(velocity, [1, 0, -1], 'same'); dV[0] = dV[-1] = 0 #dV[0] = dV[1]; dV[-1] = dV[-2] 
    peaks, valleys = _robustExtremums(dV)

    return peaks, valleys

def constantSpeed(path, density, multiplier=1):
    lengths = [0]
    for i in range(1,len(path)):
        lengths.append(lengths[i-1]+pow(numpy.sum(pow(path[i,0:2]-path[i-1,0:2],2)), 0.5))
    lTotal = lengths[-1]
    n = round(lTotal / density + 0.5)
    n *= multiplier
    r = []
    j = 0
    alpha = 0
    r.append(path[0])
    for i in range(1,int(n+1)):
        while (n * lengths[j+1] < i * lTotal): 
            j += 1
        alpha = (lengths[j+1] - i*lTotal/(float)(n)) / (float)(lengths[j+1] -lengths[j])
        r.append(path[j]*alpha + path[j+1]*(1-alpha))
    return numpy.array(r) #.astype("float32")

def ddlognormal(mu, sigma, loc=0):
    '''2-nd order derivative of the lognormal.'''
    x = sympy.Symbol('x')
    lognormal = sympy.exp(-(sympy.log(x - loc) - mu) ** 2 / (2 * sigma**2)) / ((x - loc)* sigma * numpy.sqrt(2 * numpy.pi))
    k = (sympy.log(x - loc) - mu) / sigma
    expr = lognormal * (k**2 + 3*k*sigma + 2*sigma**2 - 1) / sigma**2 / (x - loc)**2
    def eval(val):
        return [expr.subs(x, v).evalf() if v!=loc else sympy.limit(expr, x, v).evalf() for v in val]
    
    return eval

def dlognormal(mu, sigma, loc=0):
    '''1-st order derivative of the lognormal.'''
    x = sympy.Symbol('x')
    lognormal = sympy.exp(-(sympy.log(x - loc) - mu) ** 2 / (2 * sigma**2)) / ((x - loc) * sigma * numpy.sqrt(2 * numpy.pi))
    k = (sympy.log(x - loc) - mu) / sigma
    expr = -lognormal * (k + sigma) / sigma / (x - loc)
    def eval(val):
        return [expr.subs(x, v).evalf() if v!=loc else sympy.limit(expr, x, v).evalf() for v in val]
    
    return eval

# def lognormal(mu, sigma, loc=0):
#     scale = numpy.exp(mu)
#     def eval(val):
#         return lognorm.pdf(x, sigma, loc=loc, scale=scale)

#     return eval

class lognormal(object):
    """log(x) is normally distributed"""
    def __init__(self, mu, sigma, loc=0):
        super(lognormal, self).__init__()
        self.mu = mu
        self.scale = numpy.exp(mu)
        self.sigma = sigma
        self.loc = loc
        if self.scale <= 0:
            print (self.mu, self.scale, self.sigma, self.loc)

    def __call__(self, x):
        return lognorm.pdf(x, self.sigma, loc=self.loc, scale=self.scale)

    def eval(self, x):
        return lognorm.pdf(x, self.sigma, loc=self.loc, scale=self.scale)

    def sampling(self, size):
        return numpy.random.lognormal(self.mu, self.sigma, size) + self.loc


class deltaLognormal(object):
    """docstring for deltaLognormal"""
    def __init__(self, D1, D2, mu1, mu2, sigma1, sigma2, loc=0):
        super(deltaLognormal, self).__init__()
        self.agonist = lognormal(mu1, sigma1, loc)
        self.antagonist = lognormal(mu2, sigma2, loc)
        self.D1 = D1
        self.D2 = D2

    def __call__(self, x):
        return self.D1 * self.agonist(x) - self.D2 * self.antagonist(x)

    def eval(self, x):
        return self.D1 * self.agonist(x) - self.D2 * self.antagonist(x)
        
    def sampleAgonist(self, size):
        return self.D1 * self.agonist.sampling(size)

    def sampleAntagonist(self, size):
        return -self.D2 * self.antagonist.sampling(size)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    mu = -1; sigma = 0.55; x0 = 1
    x = numpy.linspace(0, 2*numpy.pi, 100)

    ln = lognormal(mu, sigma, loc=x0)
    # s = ln.sampling(1000)
    # count, bins, ignored = plt.hist(s, 100, normed=True, align='mid')
    # plt.show()
    s = ln.eval(x)
    plt.plot(x, s)
    plt.xlabel("time (s)")

    Tm = findTm(s)
    Tinf = findTinf(s, x)
    plt.scatter(x[Tm], s[Tm])
    plt.scatter(x[Tinf], s[Tinf])

    plt.show()

    dln = dlognormal(mu, sigma, loc=x0)
    diff = dln(x)
    plt.plot(x, diff)
    
    ddln = ddlognormal(mu, sigma, loc=x0)
    ddiff = ddln(x)
    plt.plot(x, ddiff)

    plt.show()

    # val = (numpy.exp(-(numpy.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * numpy.sqrt(2 * numpy.pi)))    
    # plt.plot(x, val)

    # val = - 0.2 * (numpy.exp(-(numpy.log(x) - mu - 1)**2 / (2 * sigma**2)) / (x * sigma * numpy.sqrt(2 * numpy.pi)))    
    # plt.plot(x, val)

    delta_ln = deltaLognormal(1, 0.2, mu, mu + 1, sigma, sigma, loc=x0)
    val = delta_ln.eval(x)
    plt.plot(x, val)

    plt.show()
