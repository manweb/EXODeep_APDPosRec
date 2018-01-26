import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import os
import pickle as pk
from utilities import DataSet
from display import APDDisplay
from tensorflow.python import pywrap_tensorflow

def FindBadEvents(filename):
	outFileGood = filename.replace('.csv','_good.csv')
	outFileBad = filename.replace('.csv','_bad.csv')

	count = 0
	nGood = 0
	nBad = 0
	with open(filename) as infile:
		for line in infile:
			im = np.fromstring(line, dtype=np.float32, sep=',')

			img = np.reshape(im[0:74*350], (74,350))
			img_sum = np.sum(img, axis=0)
			img_std = np.std(img_sum[0:100])
			img_max = np.max(img_sum)

			if im[-1] > 1500.0 and img_max < 4*img_std:
				option = 'a'
				if nBad == 0:
					option = 'w'

				f = open(outFileBad, option)
				f.write(",".join(np.char.mod('%f', im))+'\n')
				f.close()

				nBad += 1
			else:
				option = 'a'
                                if nGood == 0:
                                        option = 'w'

                                f = open(outFileGood, option)
                                f.write(",".join(np.char.mod('%f', im))+'\n')
                                f.close()

                                nGood += 1

			count += 1

	print('%i events processed'%count)
	print('%i bad events found (%.2f %%)'%(nBad,nBad/count*100))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputFile', type=str)

	args = parser.parse_args()

	FindBadEvents(args.inputFile)

