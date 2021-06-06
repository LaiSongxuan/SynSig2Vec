# -*- coding:utf-8 -*-
import numpy
import math

from . import functions

def estimateTheta(t, D, t0, mu, sigma, theta_s, theta_e):
    theta = functions.runDist(t, D, t0, mu, sigma) / D * (theta_e - theta_s) + theta_s 
    return theta

def estimateVxy(velocity, t, D, t0, mu, sigma, theta_s, theta_e):
    dtheta = theta_e - theta_s
    if abs(dtheta) < 1e-3:
        vx = velocity * numpy.cos(theta_s)
        vy = velocity * numpy.sin(theta_s)
    else:
        theta = estimateTheta(t, D, t0, mu, sigma, theta_s, theta_e)
        vx = velocity * numpy.cos(theta)
        vy = velocity * numpy.sin(theta)
    return vx, vy

def estimateLxy(t, D, t0, mu, sigma, theta_s, theta_e, shift=[0, 0], dt=0.01):
    dtheta = theta_e - theta_s
    if abs(dtheta) < 1e-3:
        dist = functions.runDist(t, D, t0, mu, sigma) 
        lx = dist * numpy.cos(theta_s) / dt + shift[0]
        ly = dist * numpy.sin(theta_s) / dt + shift[1]
    else:
        theta = estimateTheta(t, D, t0, mu, sigma, theta_s, theta_e)
        dD = D / dtheta
        lx = dD * (numpy.sin(theta) - numpy.sin(theta_s)) / dt + shift[0]
        ly = dD * (numpy.cos(theta_s) - numpy.cos(theta)) / dt + shift[1]
    return lx, ly

def estimateThetaSE(speed, time, idxs, D, t0, mu, sigma):
    tinf1Idx, tmIdx, tinf2Idx = idxs

    l1 = 0
    l2 = functions.runDistPoint(D, sigma, point=2)
    l3 = functions.runDistPoint(D, sigma, point=3)
    l4 = functions.runDistPoint(D, sigma, point=4)
    l5 = D

    angleT2 = numpy.arctan2(speed[tinf1Idx, 1], speed[tinf1Idx, 0])
    angleT3 = numpy.arctan2(speed[tmIdx, 1], speed[tmIdx, 0])
    angleT4 = numpy.arctan2(speed[tinf2Idx, 1], speed[tinf2Idx, 0])
    # print ("Key point angles:", angleT2, angleT3, angleT4)

    dAngle = (angleT4 - angleT2) 
    ''' Note: It's important to ascertain the orientation of the stroke. A simple solution is to constrain dAngle within a certain range.
    '''
    dAngle = math.copysign(2 * math.pi - abs(dAngle), -dAngle) if abs(dAngle) > 3./2 * math.pi else dAngle 
    ''' An ideal but less robust method
    '''
    # if abs(dAngle) > math.pi:
    #     s1 = numpy.sign(angleT3 - angleT2)
    #     s2 = numpy.sign(angleT4 - angleT2)
    #     flip = s1 * s2 # flip = 0?   
    #     dAngle = dAngle if flip == 1 else math.copysign(2 * math.pi - abs(dAngle), -dAngle) 

    dAngle = dAngle / (l4 - l2)
    theta_s = angleT3 - dAngle * (l3 - l1)
    theta_e = angleT3 + dAngle * (l5 - l3)

    return theta_s, theta_e, dAngle

def estimateThetaSE2(speed, time, idxs, D, t0, mu, sigma):
    tinf1Idx, tmIdx, tinf2Idx = idxs

    l1 = 0
    l2 = functions.runDistPoint(D, sigma, point=2)
    l3 = functions.runDistPoint(D, sigma, point=3)
    l4 = functions.runDistPoint(D, sigma, point=4)
    l5 = D

    angleT2 = numpy.arctan2(speed[tinf1Idx, 1], speed[tinf1Idx, 0])
    angleT3 = numpy.arctan2(speed[tmIdx, 1], speed[tmIdx, 0])
    angleT4 = numpy.arctan2(speed[tinf2Idx, 1], speed[tinf2Idx, 0])
    # print ("Key point angles:", angleT2, angleT3, angleT4)

    dAngle1 = (angleT4 - angleT2) 
    r1 = (l3 - l1) / (l4 - l2)
    r5 = (l5 - l3) / (l4 - l2)
    theta_s1 = angleT3 - dAngle1 * r1  
    theta_e1 = angleT3 + dAngle1 * r5
    if abs(dAngle1) < math.pi / 2:
        return theta_s1, theta_e1, dAngle1

    dAngle2 = math.copysign(2 * math.pi - abs(dAngle1), -dAngle1)
    theta_s2 = angleT3 - dAngle2 * r1
    theta_e2 = angleT3 + dAngle2 * r5

    ln = functions.lognormal(mu, sigma, loc=t0)
    velocity_a = ln.eval(time) * D
    vxa, vya = estimateVxy(velocity_a, time, D, t0, mu, sigma, theta_s1, theta_e1)
    N1 = numpy.sum((numpy.square(speed[:,0] - vxa) + numpy.square(speed[:,1] - vya)))
    vxa, vya = estimateVxy(velocity_a, time, D, t0, mu, sigma, theta_s2, theta_e2)
    N2 = numpy.sum((numpy.square(speed[:,0] - vxa) + numpy.square(speed[:,1] - vya)))
    
    if N1 < N2:
        return theta_s1, theta_e1, dAngle1
    else:
        return theta_s2, theta_e2, dAngle2

if __name__ == '__main__':
    import RX0
    import matplotlib.pyplot as plt

    speed = numpy.load("./demo_data/speed.npy")
    velocity = numpy.load("./demo_data/velocity.npy")
    path = numpy.load("./demo_data/path.npy")[:, 0:2]

    velocity[-4] = velocity[-5]
    peaks, valleys = functions.findExtremums(velocity)
    time = numpy.arange(len(velocity)) * 0.01
    
    plt.plot(velocity)
    plt.scatter(peaks, velocity[peaks])
    plt.scatter(valleys, velocity[valleys])
    plt.show()

    p1 = valleys[0]
    p2 = valleys[1]
    veloc = velocity[p1:p2+1]
    speed = speed[p1:p2+1]
    path = path[p1:p2+1]
    time = time[p1:p2+1] 
    time = time - time[0] 

    tmIdx = functions.findTm(veloc)
    tinf1Idx, tinf2Idx = functions.findTinf(veloc, time)
    idxs = [tinf1Idx, tmIdx, tinf2Idx]

    param, t, v = RX0.RX0(veloc, time, idxs, constrainParam=True)
    D, t0, mu, sigma = param

    print ("Key point index:", idxs)
    plt.scatter(path[:,0], path[:,1])
    plt.scatter(path[idxs[0],0], path[idxs[0],1], c='r', s=50)
    plt.scatter(path[idxs[1],0], path[idxs[1],1], c='g', s=50)
    plt.scatter(path[idxs[2],0], path[idxs[2],1], c='b', s=50)
    plt.show()
    
    theta_s, theta_e, dtheta = estimateThetaSE(speed, time, idxs, D, t0, mu, sigma)
    # theta_s, theta_e, dtheta = estimateThetaSE2(speed, time, idxs, D, t0, mu, sigma)
    print ("D, t0, mu, sigma:", D, t0, mu, sigma)
    print ("theta_s, theta_e, dtheta:", theta_s, theta_e, dtheta)

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

    theta = estimateTheta(time, D, t0, mu, sigma, theta_s, theta_e)
    print ("Estimated key point angles:", theta[idxs[0]], theta[idxs[1]], theta[idxs[2]])

    vx, vy = estimateVxy(_veloc, time, D, t0, mu, sigma, theta_s, theta_e)
    plt.plot(speed[:,0], c='r', marker='o')
    plt.plot(speed[:,1], c='r', marker='*')
    plt.plot(vx, c='b', marker='o')
    plt.plot(vy, c='b', marker='x')
    plt.show()

    lx, ly = estimateLxy(time, D, t0, mu, sigma, theta_s, theta_e, shift=path[0], dt=0.01)
    lx = lx + path[idxs[1], 0] - lx[idxs[1]]
    ly = ly + path[idxs[1], 1] - ly[idxs[1]]
    plt.plot(lx, ly, c='b', marker='o')
    llx = numpy.cumsum(vx) + path[0, 0]
    lly = numpy.cumsum(vy) + path[0, 1]
    llx = llx + path[idxs[1], 0] - llx[idxs[1]]
    lly = lly + path[idxs[1], 1] - lly[idxs[1]]
    plt.plot(llx, lly, c='g', marker='o')
    
    plt.plot(path[:,0], path[:,1], c='r', marker='o')
    plt.show()
