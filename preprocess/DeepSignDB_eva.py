#!/usr/bin/env python
# -*- coding:utf-8 -*-	
import os, re
import numpy 
import pickle

import matplotlib.pyplot as plt  
from scipy import interpolate, signal

numpy.set_printoptions(threshold=1e10)  

def centerNormSize(crt, coords=[0, 1]):
	assert len(coords)==2
	pps = crt[:, coords] # x & y coordinates
	
	m = numpy.min(pps, axis=0)
	M = numpy.max(pps, axis=0)
	pps = (pps - (M + m) / 2.0) / numpy.max(M - m) 
	# pps = (pps - (M + m) / 2.0) / (M - m) 
	crt[:, coords] = pps
	return crt

def centroidNormSize(crt, coords=[0, 1]):
	assert len(coords)==2
	pps = crt[:, coords] # x & y coordinates

	pps = (pps - numpy.mean(pps, axis=0)) / numpy.std(pps, axis=0) 
	crt[:, coords] = pps
	return crt

def normPressure(crt, pres=2):
	prs = crt[:, pres] # pressure
	M = float(numpy.max(prs))
	prs = prs / M
	crt[:, pres] = prs
	return crt

def padAvergeSpeed(crt, indicator, coords=[0, 1]):
	assert coords==[0, 1]
	# Valid points (non-zero pressure)
	inds = numpy.where(crt[:, indicator])[0]
	# Start point of each stroke
	rowIdx = numpy.where((inds[1:] - inds[0:-1]) != 1)[0] + 1
	rowIdx = numpy.pad(rowIdx, (1, 1), mode="constant")
	# plt.plot(crt[:, indicator], marker="*")
	# plt.scatter(inds[rowIdx], crt[inds[rowIdx], indicator], marker="o", color="r")
	# plt.show()
	rowIdx[-1] = inds.shape[0] 
	DIST = []; count = []
	crt_new = []
	press = []
	for i in range(0,len(rowIdx)-1):
		segLen = rowIdx[i+1]-rowIdx[i]
		segment = crt[inds[rowIdx[i]]:inds[rowIdx[i]]+segLen, coords]
		# print (numpy.where(numpy.sum(segment[1:] == segment[0:-1], axis=1)==2))
		if segLen == 1: #Ignore it?
			crt_new.append(crt[inds[rowIdx[i]]:inds[rowIdx[i]]+segLen, :])
			press.append(crt_new[-1][0, indicator])
			crt_new[-1][0, indicator] = 0 #Mark the start point of each stroke with 0
			continue
		inc = segment[1:] - segment[0:-1]
		dist = numpy.sqrt(numpy.sum(inc**2, axis=1))
		DIST.append(numpy.sum(dist))
		count.append(numpy.count_nonzero(dist))
		crt_new.append(crt[inds[rowIdx[i]]:inds[rowIdx[i]]+segLen, :])
		press.append(crt_new[-1][0, indicator])
		crt_new[-1][0, indicator] = 0 #Mark the start point of each stroke with 0
	if len(count) > 0:
		aveInc = sum(DIST) / sum(count) #Global average velocity
	crt = crt_new = numpy.concatenate(crt_new, axis=0)
	count = 0
	count_press = 0
	crt_new[0, indicator] = press[0] #Restore the indicator value
	# Skip the first point.
	for i in range(1, crt.shape[0]):
		if crt[i, indicator] == 0: #button_status
			# Last point of the last line and first point of current line
			data = crt[i-1:i+1, coords]
			dist = (sum((data[1] - data[0])**2))**0.5
			l = int(dist / aveInc) + 1
			if l <= 2:
				# penUp = numpy.zeros((1, crt.shape[1])).astype(crt.dtype); penUp[0, 0:2] = numpy.mean(data, axis=0)
				# crt_new = numpy.concatenate((crt_new[0:i+count], penUp, crt_new[i+count:]))
				# count = count + 1
				## If the above three lines are commented, this stroke is then regarded connected to the previous stroke, 
				## because there is no zero point in between.
				count_press += 1
				crt_new[i+count, indicator] = press[count_press]
				continue
			interv = numpy.array([0 + t * 10 for t in range(l)])
			intervT = interv - interv[0]
			fX = interpolate.interp1d(intervT[[0,-1]], data[:, 0], kind='slinear')
			fY = interpolate.interp1d(intervT[[0,-1]], data[:, 1], kind='slinear')
			intervX = fX(intervT)[1:-1]
			intervY = fY(intervT)[1:-1]
			intervT = interv[1:-1]
			penUp = numpy.zeros((l-2, crt.shape[1])).astype(crt.dtype)
			penUp[:, 0] = intervX
			penUp[:, 1] = intervY
			crt_new = numpy.concatenate((crt_new[0:i+count], penUp, crt_new[i+count:]))
			count = count + l - 2
			count_press += 1
			crt_new[i+count, indicator] = press[count_press]
	# plt.plot(crt_new[:, indicator], marker="*")
	# plt.show()
	return crt_new.astype("float32")

def interp(crt, ratio):
    x = crt[:, 0]
    y = crt[:, 1]
    p = crt[:, 2]
    t = numpy.linspace(0, len(crt)-1, num=len(crt), endpoint=True)
    t_new = numpy.linspace(0, len(crt)-1, num=1+int((len(crt)-1)*ratio), endpoint=True)
    f = interpolate.interp1d(t, x, 'cubic')
    x_new = f(t_new)
    f = interpolate.interp1d(t, y, 'cubic')
    y_new = f(t_new)
    f = interpolate.interp1d(t, p, 'linear')
    p_new = f(t_new)
    crt = numpy.concatenate([x_new[:, None], y_new[:, None], p_new[:, None]], axis=1).astype(crt.dtype)
    return crt

# Development set of DeepSignDB:
# MCYT: 1-230; BioSecurID: 231-498; BioSecure DB2: 499-1008; e-BioSign1: 1009-1038; e-BioSign2: 1039-1084
# Evaluation set of DeepSignDB:
# MCYT: 1-100; BioSecurID: 101-232; BioSecure DB2: 233-372; e-BioSign1: 373-407; e-BioSign2: 408-442

# Sampling rates and devices:
# MCYT: 100 Hz (Wacom Intuos3 A6); BioSecurID: 100 Hz (Wacom Intuos3 A4); BioSecure DB2 100 Hz (Wacom Intuos3 A6); 
# e-BioSign1: 200Hz (W1: Wacom STU-500, W2: Wacom STU-530, W3: Wacom DTU-1031); e-BioSign2: 200 Hz (W1-3: Wacom STU-530)

# The finger scenario:
# (e-BioSign1, W4) records the "End" status (i.e. last point of the stroke). The others not.

# We assume uniform sampling, i.e., intevals between two timestamps are equal. Although this assumption is not true, 
# especially for finger-written signatures on mobile-devices, it works well and simplifies the preprocessing.

virtualPenup = False # It is recommended to set as "False"
finger = False
if finger:
	path = "Path_To_DeepSignDB/DeepSignDB/DeepSignDB/Evaluation/finger"
else:
	path = "Path_To_DeepSignDB/DeepSignDB/DeepSignDB/Evaluation/stylus"
usersMCYT = {}
usersBSID = {}
usersBSDB2 = {}
usersEBio1 = {}
usersEBio2 = {}
lens = []
lens2 = []
for p, dirs, files in os.walk(path):
	if len(dirs) != 0:
		continue
	fs = []
	for fn in files:
		temp = fn.split("_")
		label = int(temp[0][1:]) 
		if 1 <= label <= 100:
			# Move this line for a step-by-step preprocessing of different subsets.
			fs.append(fn)
			continue
		elif 101 <= label <= 232:
			continue
		elif 233 <= label <= 372:
			continue
		elif 374 <= label <= 407:
			continue
		elif 408 <= label <= 442:
			continue			
	
	### Important: Rename BSDB2 for a correct sorting.
	# for f in fs:
	# 	temp = f.split("_")
	# 	if len(temp[-1]) == 7:
	# 		digit = int(re.findall(r'\d', temp[-1])[0])
	# 		fnew = f[:-7]+'sg%.2d.txt'%digit
	# 		fnew = os.path.join(path, fnew)
	# 		f = os.path.join(path, f)
	# 		print(f, fnew)
	# 		#### os.rename(f, fnew)
	
	fs = sorted(fs)

	for fidx, fn in enumerate(fs):
		temp = fn.split("_")
		label = int(temp[0][1:]) 
		veri = temp[1]
		veri = True if veri == 'g' else False 
		
		print (fn, label, veri)

		fhd = open(os.path.join(p, fn), "r")
		lineNum = int(fhd.readline())
		lines = fhd.readlines()
		fhd.close()
		assert len(lines) == lineNum
		crt = []
		for l in lines:
			ll = [float(ll) for ll in l.split()]
			crt.append(ll) 

		# MCYT: x, y, timestamp, azimuth angle, altitude angle, p
		# BioSecurID & BioSecure DB2: x, y, timestamp, button status, azimuth angle, altitude angle, p
		# e-BioSign1 & e-BioSign2: x, y, timestamp, p
		crt = numpy.array(crt, "float32")[:,[0,1,-1]] #x, y, p
		crt = centerNormSize(crt)
		crt = normPressure(crt, 2)

		### padAvergeSpeed is used to replace real pen-ups with virtual pen-ups. 
		if not finger and virtualPenup:
			# plt.plot(crt[:,0], crt[:,1], marker="*")
			crt = padAvergeSpeed(crt, 2)
			# plt.plot(crt[:,0], crt[:,1], marker="o")
			# crt = crt[numpy.where(crt[:,2]!=0)[0]]
			# plt.plot(crt[:,0], crt[:,1], marker=".")
			# plt.show()

		# plt.plot(crt[:,0], crt[:,1], c="r", marker="*") 
		# idx = numpy.where(numpy.sum(crt[0:-1, 0:2] == crt[1:, 0:2], axis=1)==2)[0]
		# plt.scatter(crt[idx, 0], crt[idx, 1], marker="o")
		# plt.show()

		if 1 <= label <= 100:
			usersTemp = usersMCYT
		elif 101 <= label <= 232:
			usersTemp = usersBSID
		elif 233 <= label <= 372:
			usersTemp = usersBSDB2
		elif 373 <= label <= 407:
			#Average length: 628, 615, 641, 451, 213 for w1 - w5
			crt[:,1] = -crt[:,1]
			w = int(temp[-3][1])
			if w == 1 or w == 2 or w == 3:
				crt = interp(crt, 0.5)
			if w == 4:
				# if veri:
				# 	lens.append(crt.shape[0])
				# else:
				# 	lens2.append(crt.shape[0])
				crt = interp(crt, 1/1.5)
			if w == 5:
				crt = interp(crt, 1.5)
			usersTemp = usersEBio1
		elif 408 <= label <= 442:
			crt[:,1] = -crt[:,1]
			w = int(temp[-4][1])
			if w == 5:
				crt = interp(crt, 1.5)
			if w == 6:
				crt = interp(crt, 1.5)
			if w == 2:
				crt = interp(crt, 0.5)
			usersTemp = usersEBio2
		else:
			raise ValueError("")

		if label not in usersTemp.keys():
			usersTemp[label] = {}
		user = usersTemp[label]
		if veri not in user.keys():
			user[veri] = []	
		user[veri].append(crt)

# Pickle dictionary using protocol 0.
if finger:
	if len(usersEBio1) > 0:			
		output = open('../data/EBio1_eva_finger.pkl', 'wb')
		pickle.dump(usersEBio1, output)
		output.close()

	if len(usersEBio2) > 0:			
		output = open('../data/EBio2_eva_finger.pkl', 'wb')
		pickle.dump(usersEBio2, output)
		output.close()

elif virtualPenup:
	if len(usersMCYT) > 0:
		output = open('../data/MCYT_eva_pad.pkl', 'wb')
		pickle.dump(usersMCYT, output)
		output.close()

	if len(usersBSID) > 0:
		output = open('../data/BSID_eva_pad.pkl', 'wb')
		pickle.dump(usersBSID, output)
		output.close()

	if len(usersBSDB2) > 0:
		output = open('../data/BSDS2_eva_pad.pkl', 'wb')
		pickle.dump(usersBSDB2, output)
		output.close()
				
	if len(usersEBio1) > 0:			
		output = open('../data/EBio1_eva_pad.pkl', 'wb')
		pickle.dump(usersEBio1, output)
		output.close()

	if len(usersEBio2) > 0:
		output = open('../data/EBio2_eva_pad.pkl', 'wb')
		pickle.dump(usersEBio2, output)
		output.close()
else:
	if len(usersMCYT) > 0:
		output = open('../data/MCYT_eva.pkl', 'wb')
		pickle.dump(usersMCYT, output)
		output.close()

	if len(usersBSID) > 0:
		output = open('../data/BSID_eva.pkl', 'wb')
		pickle.dump(usersBSID, output)
		output.close()

	if len(usersBSDB2) > 0:
		output = open('../data/BSDS2_eva.pkl', 'wb')
		pickle.dump(usersBSDB2, output)
		output.close()
				
	if len(usersEBio1) > 0:			
		output = open('../data/EBio1_eva.pkl', 'wb')
		pickle.dump(usersEBio1, output)
		output.close()

	if len(usersEBio2) > 0:
		output = open('../data/EBio2_eva.pkl', 'wb')
		pickle.dump(usersEBio2, output)
		output.close()