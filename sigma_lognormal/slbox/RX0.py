# -*- coding:utf-8 -*-
import numpy
import math

from . import functions

def estimateMu(t1, t2, a1, a2):
    mu = math.log((t1 - t2) / (math.exp(-a1) - math.exp(-a2)))
    return mu

def estimateT0(t1, a1, mu):
    t0 = t1 - math.exp(mu - a1)
    return t0

def estimateD(a1, mu, sigma, v1):
    D = math.sqrt(math.pi*2) * v1 * sigma * math.exp(mu-a1+a1**2/(2*sigma**2))
    return D

def _estimateSigmat23(tinf1, tm, tinf2, vinf1, vm, vinf2):
    beta23 = vinf1 / vm
    return math.sqrt(-2 - 2*math.log(beta23) - 0.5/math.log(beta23))

def _estimateSigmat34(tinf1, tm, tinf2, vinf1, vm, vinf2):
    beta43 = vinf2 / vm
    return math.sqrt(-2 - 2*math.log(beta43) - 0.5/math.log(beta43))

def _estimateSigmat24(tinf1, tm, tinf2, vinf1, vm, vinf2):
    beta42 = vinf2 / vinf1
    return math.sqrt(2 * math.sqrt(1 + math.log(beta42)**2) - 2)

# def _estimateSigmat234(tm, tinf1, tinf2, vm, vinf1, vinf2, sigma0=0.5):
#     def f(sigma, *args): #(f(x|arg), given arg, solve x)
#         a3 = sigma ** 2
#         a2 = 1.5 * a3 + sigma * math.sqrt(0.25 * a3 + 1)
#         a4 = 1.5 * a3 - sigma * math.sqrt(0.25 * a3 + 1)
#         return (tinf1 - tm) * (math.exp(-a4) - math.exp(-a2)) - (tinf2 - tinf1) * (math.exp(-a4) - math.exp(-a3))
#     sigma = optimize.fsolve(f, x0=sigma0) #args=(tm, tmin, tmax)
#     return sigma[0]

def _dt2(sigma, mu): #t2-t3
    return -math.exp(mu - sigma**2) * (1 - math.exp(-0.5*sigma*(sigma+math.sqrt(sigma**2+4)))) 
def _dt4(sigma, mu): #t4-t3
    return -math.exp(mu - sigma**2) * (1 - math.exp(-0.5*sigma*(sigma-math.sqrt(sigma**2+4)))) 
def _dv2(sigma): #v2/v3
    return math.exp(-0.25*(sigma**2 + sigma*math.sqrt(sigma**2+4) + 2))
def _dv4(sigma): #v4/v3
    return math.exp(-0.25*(sigma**2 - sigma*math.sqrt(sigma**2+4) + 2))
def _f(x, dp2_1, dp2_2):
    x1, y1 = dp2_1
    x2, y2 = dp2_2
    return (y2 - y1) / (x2 - x1) * (x - x1) + y1
def _applyConstrait(dt, dv, *args):
    [dt_bottomLeft, dt_bottomRight, dt_upperLeft, dt_upperRight, \
    dv_bottomLeft, dv_bottomRight, dv_upperLeft, dv_upperRight] = args
    if dt < dt_bottomLeft:
        dt = dt_bottomLeft
        dv = dv_bottomLeft
        return dt, dv
    if dt > dt_upperRight:
        dt = dt_upperRight
        dv = dv_upperRight
        return dt, dv
    temp = _f(dt, (dt_bottomLeft, dv_bottomLeft), (dt_upperLeft, dv_upperLeft))
    if not (dv <= temp):
        dv = temp
    temp = dv_upperRight #_f(dt, (dt_upperLeft, dv_upperLeft), (dt_upperRight, dv_upperRight))
    if not (dv <= temp):
        dv = temp
        return dt, dv
    temp = _f(dt, (dt_bottomRight, dv_bottomRight), (dt_upperRight, dv_upperRight))
    if not (dv >= temp):
        dv = temp
    temp = dv_bottomLeft #_f(dt, (dt_bottomLeft, dv_bottomLeft), (dt_bottomRight, dv_bottomRight))
    if not (dv >= temp):
        dv = temp
    return dt, dv

def paramConstraintP2(dt, dv, mu_l=-2.0, mu_u=-1.0, sigma_l=0.12, sigma_u=0.5):
    # Constrain dv within a proper range.
    dt_bottomLeft = _dt2(sigma_u, mu_u)
    dt_bottomRight = _dt2(sigma_u, mu_l)
    dt_upperLeft = _dt2(sigma_l, mu_u)
    dt_upperRight = _dt2(sigma_l, mu_l)
    dv_bottomLeft = dv_bottomRight = _dv2(sigma_u)
    dv_upperLeft = dv_upperRight = _dv2(sigma_l)
    
    return _applyConstrait(dt, dv, *[dt_bottomLeft, dt_bottomRight, dt_upperLeft, dt_upperRight,\
                                    dv_bottomLeft, dv_bottomRight, dv_upperLeft, dv_upperRight])

def paramConstraintP4(dt, dv, mu_l=-2.0, mu_u=-1.0, sigma_l=0.12, sigma_u=0.5):
    dt_bottomLeft = _dt4(sigma_l, mu_l)
    dt_bottomRight = _dt4(sigma_l, mu_u)
    dt_upperLeft = _dt4(sigma_u, mu_l)
    dt_upperRight = _dt4(sigma_u, mu_u)
    dv_bottomLeft= dv_bottomRight = _dv4(sigma_l)
    dv_upperLeft= dv_upperRight = _dv4(sigma_u)

    return _applyConstrait(dt, dv, *[dt_bottomLeft, dt_bottomRight, dt_upperLeft, dt_upperRight,\
                                    dv_bottomLeft, dv_bottomRight, dv_upperLeft, dv_upperRight])

def RX0(velocity, time, idxs, constrainParam=True):   
    tinf1Idx, tmIdx, tinf2Idx = idxs
    t = [time[tinf1Idx], time[tmIdx], time[tinf2Idx]]
    v = [velocity[tinf1Idx], velocity[tmIdx], velocity[tinf2Idx]]

    if constrainParam:
        dt2 = t[0] - t[1]; dt4 = t[2] - t[1]
        dv2 = v[0] / v[1]; dv4 = v[2] / v[1]   
        dt2, dv2 = paramConstraintP2(dt2, dv2)
        dt4, dv4 = paramConstraintP4(dt4, dv4)
        t[0] = t[1] + dt2; t[2] = t[1] + dt4
        v[0] = v[1] * dv2; v[2] = v[1] * dv4

    # plt.scatter(tinf1Idx, velocity[tinf1Idx])
    # plt.scatter(tinf2Idx, velocity[tinf2Idx])
    # plt.scatter(tmIdx, velocity[tmIdx])
    # plt.show(); quit()

    combs = [[1, 0], [1, 2], [0, 2]]
    params = []
    # velocity selections
    for c_v in combs:
        idx1_v, idx2_v = sorted(c_v)
        func = eval("_estimateSigmat%d%d"%(idx1_v+2, idx2_v+2))
        sigma = func(*(t + v))
        sigma2 = sigma**2
        a = [1.5*sigma2+sigma*math.sqrt(0.25*sigma2+1), sigma2, 1.5*sigma2-sigma*math.sqrt(0.25*sigma2+1)]
        # time selections
        for c_t in combs:
            idx1_t, idx2_t = sorted(c_t)
            mu = estimateMu(t[idx1_t], t[idx2_t], a[idx1_t], a[idx2_t])
            #The maximum point is more stable and prefered 
            D = estimateD(a[c_v[0]], mu, sigma, v[c_v[0]]) 
            t0 = estimateT0(t[c_t[0]], a[c_t[0]], mu)
            params.append([D, t0, mu, sigma])

    # plt.plot(velocity, c='r')
    # for param in params:
    #     D, t0, mu, sigma = param
    #     ln_agonist = functions.lognormal(mu, sigma, loc=t0)
    #     reconVeloc = ln_agonist.eval(time) * D 
    #     plt.plot(reconVeloc, c='r')
    # plt.show(); quit()

    bestParam = None
    minError = numpy.inf
    # Return the best parameter set that minimizes the reconstruction error.
    for param in params:
        D, t0, mu, sigma = param
        ln = functions.lognormal(mu, sigma, loc=t0)
        reconVeloc = ln.eval(time) * D 
        reconError = numpy.sum((reconVeloc - velocity)**2) #[tinf1Idx:tinf2Idx+1]
        if reconError < minError:
            minError = reconError
            bestParam = param

    return bestParam, t, v
