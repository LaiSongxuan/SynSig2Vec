# -*- coding:utf-8 -*-
import numpy
import math
from scipy.optimize import leastsq, least_squares

from . import functions
from . import angleEst

def localOptim(t, path, speed, tm, vm, idxs, param, dt=0.01, loop=1):
    ''' Param: The list of parameters for one signal.
    '''
    D, t0, mu, sigma, theta_s, theta_e = param

    def residual(vars, t, path, speed, D, t0):
        mu, sigma, theta_s, theta_e = vars
        ''' Eqs. (09) & (10)
        ''' 
        # ln = functions.lognormal(mu, sigma, loc=t0)
        # velocity = ln.eval(t) * D
        # vx, vy = angleEst.estimateVxy(velocity, t, D, t0, mu, sigma, theta_s, theta_e)
        # lx = numpy.cumsum(vx) + path[0,0]
        # ly = numpy.cumsum(vy) + path[0,1]
        ''' Eqs. (11) & (12)
        ''' 
        lx, ly = angleEst.estimateLxy(t, D, t0, mu, sigma, theta_s, theta_e, shift=path[0], dt=dt) #
        error = numpy.square(path[idxs[0]:idxs[2],0]-lx[idxs[0]:idxs[2]]) + \
                numpy.square(path[idxs[0]:idxs[2],1]-ly[idxs[0]:idxs[2]])
        return error 

    while (loop > 0):
        # [mu, sigma, theta_s, theta_e] = leastsq(residual, [mu, sigma, theta_s, theta_e], args=(t, path, speed, D, t0), maxfev=400)[0]
        [mu, sigma, _, _] = leastsq(residual, [mu, sigma, theta_s, theta_e], args=(t, path, speed, D, t0), maxfev=400)[0]
        t0 = tm - math.exp(mu - sigma**2)
        D = vm * sigma * 2.506628 * math.exp(mu - 0.5 * sigma**2) #2.506628: sqrt(2pi)
        # theta_s ,theta_e, _ = angleEst.estimateThetaSE2(speed, t, idxs, D, t0, mu, sigma)
        loop -= 1

    return [D, t0, mu, sigma, theta_s, theta_e]


def globalOptim(t, path, speed, params, dt=0.01):
    ''' Params: list or numpy array of parameters for the full signal.
    '''
    J = len(params)
    if 6 * J > path.shape[0]: #More parameters than observations
        print ("More parameters than observations! Skip global optimization!")
        return params

    def residualVeloc(vars, t, path, speed, theta_s_all, theta_e_all, decay=None):
        D_all = vars[0:J]; t0_all = vars[J:2*J]; mu_all = vars[2*J:3*J]; sigma_all = vars[3*J:4*J]
        reconVelocX = reconVelocY = 0
        for k, _ in enumerate(D_all):
            ln = functions.lognormal(mu_all[k], sigma_all[k], loc=t0_all[k])
            veloc = ln.eval(t) * D_all[k]
            vx, vy = angleEst.estimateVxy(veloc, t, D_all[k], t0_all[k], mu_all[k], sigma_all[k], theta_s_all[k], theta_e_all[k])
            reconVelocX += vx
            reconVelocY += vy
        error = numpy.square(speed[:,0] - reconVelocX) + numpy.square(speed[:,1] - reconVelocY)
        # if decay is not None:
        #     error += 0.1 * numpy.mean(numpy.square(vars - decay))
        return error

    def residualAngle(vars, t, path, speed, D_all, t0_all, mu_all, sigma_all, decay=None):
        theta_s_all = vars[0:J]; theta_e_all = vars[J:2*J]
        reconX = reconY = 0
        for k, _ in enumerate(D_all):
            ln = functions.lognormal(mu_all[k], sigma_all[k], loc=t0_all[k])
            lx, ly = angleEst.estimateLxy(t, D_all[k], t0_all[k], mu_all[k], sigma_all[k], theta_s_all[k], theta_e_all[k], dt=dt)
            reconX += lx
            reconY += ly
        shiftX = numpy.mean(path[:,0])-numpy.mean(reconX)
        shiftY = numpy.mean(path[:,1])-numpy.mean(reconY)
        error = numpy.square(path[:,0]-shiftX-reconX) + \
                numpy.square(path[:,1]-shiftY-reconY)
        # if decay is not None:
        #     error += 0.1 * numpy.mean(numpy.square(vars - decay))
        return error        

    def residualFull(vars, t, path, speed, decay=None):
        D_all = vars[0:J]; t0_all = vars[J:2*J]; mu_all = vars[2*J:3*J]
        sigma_all = vars[3*J:4*J]; theta_s_all = vars[4*J:5*J]; theta_e_all = vars[5*J:6*J]
        reconVelocX = reconVelocY = 0
        for k, _ in enumerate(D_all):
            ln = functions.lognormal(mu_all[k], sigma_all[k], loc=t0_all[k])
            veloc = ln.eval(t) * D_all[k]
            vx, vy = angleEst.estimateVxy(veloc, t, D_all[k], t0_all[k], mu_all[k], sigma_all[k], theta_s_all[k], theta_e_all[k])
            reconVelocX += vx
            reconVelocY += vy
        error = numpy.square(speed[:,0] - reconVelocX) + numpy.square(speed[:,1] - reconVelocY)
        # if decay is not None:
        #     error += 0.1 * numpy.mean(numpy.square(vars - decay))
        return error 

    D, t0, mu, sigma, theta_s, theta_e = map(numpy.array, zip(*params))
    
    vars1 = numpy.concatenate((D, t0, mu, sigma), axis=0)
    vars1 = leastsq(residualVeloc, vars1, args=(t, path, speed, theta_s, theta_e), maxfev=400)[0]
    # vars1 = least_squares(residualVeloc, vars1, args=(t, path, speed, theta_s, theta_e), method='lm', max_nfev=400)['x']
    D = vars1[0:J]; t0 = vars1[J:2*J]; mu = vars1[2*J:3*J]; sigma = vars1[3*J:4*J]
    
    vars2 = numpy.concatenate((theta_s, theta_e), axis=0)
    vars2 = leastsq(residualAngle, vars2, args=(t, path, speed, D, t0, mu, sigma), maxfev=400)[0]
    # vars2 = least_squares(residualAngle, vars2, args=(t, path, speed, D, t0, mu, sigma), method='lm', max_nfev=400)['x']
    theta_s = vars2[0:J]; theta_e = vars2[J:2*J]

    # vars3 = numpy.concatenate((vars1, vars2), axis=0)
    # vars3 = leastsq(residualFull, vars3, args=(t, path, speed), maxfev=1000)[0]
    # # vars3 = least_squares(residualFull, vars3, args=(t, path, speed), method='lm', max_nfev=1000)['x']
    # D = vars3[0:J]; t0 = vars3[J:2*J]; mu = vars3[2*J:3*J]; sigma = vars3[3*J:4*J]; theta_s = vars3[4*J:5*J]; theta_e = vars3[5*J:6*J]

    params = numpy.concatenate((D[:,None], t0[:,None], mu[:,None], sigma[:,None], theta_s[:,None], theta_e[:,None]), axis=1)
    # D_med = numpy.median(D)
    # Prevent some extreme values
    params = params[params[:,1]<t[-1]]
    params = params[params[:,2]<0]
    params = params[params[:,2]>-3]
    params = params[params[:,3]>0.035] 

    return params

if __name__ == '__main__':
    import RX0
    import matplotlib.pyplot as plt

    speed = numpy.load("./demo_data/speed.npy")
    velocity = numpy.load("./demo_data/velocity.npy")
    path = numpy.load("./demo_data/path.npy")[:, 0:2]

    peaks, valleys = functions.findExtremums(velocity)
    time = numpy.arange(len(velocity)) * 0.01

    p1 = valleys[0] 
    p2 = valleys[0+1]
    veloc = velocity[p1:p2+1]
    speed = speed[p1:p2+1]
    path = path[p1:p2+1]
    time = time[p1:p2+1] - time[p1]

    if 1:
        from extractionFirstMode import paramExtraction, paramReconstruction
        params, AUC, R, _, _, _, _ = paramExtraction(time, path, lAmbda=15., smoothing=True, suffix="")
        # params_res, AUC_res, _, _, _, _, _ = paramExtraction(time, R, lAmbda=5., smoothing=False, suffix="_res")
        # params = numpy.concatenate((params, params_res[0:2]), axis=0)
        px, py, vx, vy = paramReconstruction(params, time)
        plt.plot(time, veloc, c='r')
        plt.plot(time, numpy.sqrt(vx**2+vy**2), c='b')
        plt.show()
        plt.scatter(path[:,0], path[:,1])
        plt.scatter(px+path[0,0], py+path[0,1])
        plt.show()
        print (numpy.sum((px-path[:,0]+path[0,0])**2+(py-path[:,1]+path[0,1])**2))
        
        print (params)
        params = globalOptim(time, path, speed, params)
        px, py, vx, vy = paramReconstruction(params, time)
        plt.plot(time, veloc, c='r')
        plt.plot(time, numpy.sqrt(vx**2+vy**2), c='b')
        plt.show()
        plt.scatter(path[:,0], path[:,1])
        plt.scatter(px+path[0,0], py+path[0,1])
        plt.show()
        print (numpy.sum((px-path[:,0]+path[0,0])**2+(py-path[:,1]+path[0,1])**2))
    else:
        tmIdx = functions.findTm(veloc)
        tinf1Idx, tinf2Idx = functions.findTinf(veloc, time)
        idxs = [tinf1Idx, tmIdx, tinf2Idx]
        param, t, v = RX0.RX0(veloc, time, idxs, constrainParam=False)
        D, t0, mu, sigma = param

        print ("Key point index:", idxs)
        plt.scatter(path[:,0], path[:,1])
        plt.scatter(path[idxs[0],0], path[idxs[0],1], c='r', s=50)
        plt.scatter(path[idxs[1],0], path[idxs[1],1], c='g', s=50)
        plt.scatter(path[idxs[2],0], path[idxs[2],1], c='b', s=50)
        plt.show()

        plt.plot(time, veloc, c='r')
        ln = functions.lognormal(mu, sigma, loc=t0)
        _veloc = ln.eval(time) * D 
        plt.plot(time, _veloc, c='b')
        plt.scatter(time[idxs[0]], veloc[idxs[0]], c='r', s=20)
        plt.scatter(time[idxs[1]], veloc[idxs[1]], c='r', s=20)
        plt.scatter(time[idxs[2]], veloc[idxs[2]], c='r', s=20)
        plt.scatter(time[idxs[0]], _veloc[idxs[0]], c='r', s=20)
        plt.scatter(time[idxs[1]], _veloc[idxs[1]], c='r', s=20)
        plt.scatter(time[idxs[2]], _veloc[idxs[2]], c='r', s=20)
        plt.show()

        theta_s, theta_e, dtheta = angleEst.estimateThetaSE2(speed, time, idxs, D, t0, mu, sigma)
        # theta_s, theta_e, dtheta = angleEst.estimateThetaSE(speed, time, idxs, D, t0, mu, sigma)
        print ("D, t0, mu, sigma:", D, t0, mu, sigma)
        print ("theta_s, theta_e, dtheta:", theta_s, theta_e, dtheta)

        param = param + [theta_s, theta_e]
        param = localOptim(time, path, speed, t[1], v[1], idxs, param, loop=1)

        D, t0, mu, sigma, theta_s, theta_e = param
        print ("D, t0, mu, sigma:", D, t0, mu, sigma)
        print ("theta_s, theta_e, dtheta:", theta_s, theta_e, dtheta)
        
        plt.plot(time, veloc, c='r')    
        ln = functions.lognormal(mu, sigma, loc=t0)
        _veloc = ln.eval(time) * D 
        plt.plot(time, _veloc, c='g')
        plt.show()

        vx, vy = angleEst.estimateVxy(veloc, time, D, t0, mu, sigma, theta_s, theta_e)
        plt.plot(speed[:,0], c='r', marker='o')
        plt.plot(speed[:,1], c='r', marker='*')
        plt.plot(vx, c='b', marker='o')
        plt.plot(vy, c='b', marker='x')
        plt.show()

        lx, ly = angleEst.estimateLxy(time, D, t0, mu, sigma, theta_s, theta_e, shift=path[0], dt=0.01)
        lx = lx + path[idxs[1], 0] - lx[idxs[1]]
        ly = ly + path[idxs[1], 1] - ly[idxs[1]]
        plt.plot(lx, ly, c='b', marker='o')
        plt.plot(path[:,0], path[:,1], c='r', marker='o')
        plt.show()
        print (lx)
        print (path)
        




