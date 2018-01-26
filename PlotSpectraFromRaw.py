import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
from display import APDDisplay

def MakePlot(filename):
	data = np.genfromtxt(file(filename), delimiter=',')

	dsp = APDDisplay()

	if data.shape[1] == 2:
		dsp.PlotHistos(data[:,0], data[:,1], '', True, '')
	else:
		dsp.PlotHistos(data[:,0:3], data[:,3:6])
		dsp.DisplayPosition(data[:,0:3], data[:,3:6], False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputFile', type=str, default='')

	args = parser.parse_args()

	MakePlot(args.inputFile)

