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
with open('PS_2021.01.11_18.33.36.csv', newline='') as csvfile:
	planets = csv.reader(csvfile, delimiter=',', quotechar='#')
	for row in planets:
		if row[1] in data:
			data[row[1]].append(row[11])
		else:
			data[row[1]] = [row[11]]

for key in data.keys():
	data[key] = sorted(data[key])[0]
# 	number_of_planets.append(len(data[key]))

# print(number_of_planets.count(1))
# print(number_of_planets.count(2))
# print(number_of_planets.count(3))
# print(number_of_planets.count(4))
# print(number_of_planets.count(5))
# print(number_of_planets.count(6))
# n, bins, patches = plt.hist(number_of_planets)
# plt.xlabel('Planets')
# plt.ylabel('Count')
# plt.title('Counts of planets')
# plt.grid(True)
# plt.show()


# 		formatted_planets.writerow([key]  + data[key])

# for key in data.keys():
# Data settings


# Download data
lenghts = []
with open('data\\stars.csv', 'w', newline='') as csvfile:
	formatted_planets = csv.writer(csvfile, delimiter=',', quotechar='#', quoting=csv.QUOTE_MINIMAL)
	for target in data.keys():
		print(target)
		# target = 'HD 51608'
		data_from_Dace = Spectroscopy.get_timeseries(target, sorted_by_instrument=False)
		if data_from_Dace['rv']:
			try:
				frequency, power = LombScargle(data_from_Dace['rjd'], data_from_Dace['rv']).autopower()
				np.savetxt("data\\{}_freq.csv".format(target), frequency, delimiter=",")
				np.savetxt("data\\{}_power.csv".format(target), power, delimiter=",")
				formatted_planets.writerow([target]  + [data[target]])
				lenghts.append(len(frequency))
			except:
				pass
print(max(lenghts))
lenghts = [max(lenghts)] + lenghts
lenghts = np.array(lenghts)
np.savetxt("data\\lenghts.csv", lenghts, delimiter=",")