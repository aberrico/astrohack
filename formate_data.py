import csv
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib.pyplot as plt
import dace
from dace.spectroscopy import Spectroscopy
from astropy.timeseries import LombScargle


data = {}
number_of_planets = []
names = []
with open('new_data\\hardstars.csv', newline='', errors='ignore') as csvfile:
	planets = csv.reader(csvfile, delimiter=',', quotechar='#')
	for row in planets:
		names.append(row[0])
names = set(names)
parse = False
dates = []
rvs = []
stars = []
here = []
with open('aat_vels_.txt', newline='') as csvfile:
	for line in csvfile:
		x = line.split(" ")
		try:
			float(x[0])
			#data
			date = x[0]
			rv = x[0]
			if parse:
				dates[-1].append(float(date))
				rvs[-1].append(float(rv))

		except:
			star = x[0].split('.')
			star = star[0].split('_')
			star = star[0]
			here.append(star)
			if star in names:
				parse = True
				dates.append([])
				rvs.append([])
				stars.append(star)
				
			else:
				parse = False
print(len(stars))
print("p")
# for i in range(len(stars)):
# 	date_data = np.array(dates[i])
# 	rv_data = np.array(rvs[i])
# 	frequency, power = LombScargle(date_data, rv_data).autopower()
# 	np.savetxt("new_data\\{}_freq.csv".format(stars[i]), frequency, delimiter=",")
# 	np.savetxt("new_data\\{}_power.csv".format(stars[i]), power, delimiter=",")