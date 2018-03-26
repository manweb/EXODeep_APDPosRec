import numpy as np
import csv
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import argparse

def PlotLabels(filenames):
	reader = csv.reader(open(filenames[0]), delimiter=',')
	#nFeatures = len(next(reader)) - 4
	nFeatures = 74*350
	print("Number of features: %i"%nFeatures)

        dataX = []
	dataY = []
	dataZ = []
	dataE = []
	dataES = []
	dataESSum = []
        for inputFile in filenames:
                print("Processing file %s"%inputFile)
		nEvents = sum(1 for row in open(inputFile))
		count = 0
                with open(inputFile) as infile:
                        for line in infile:
                                x = np.fromstring(line, dtype=float, sep=',')
				cols = np.reshape(x[0:nFeatures],(74,350))
                                dataX.append(x[nFeatures])
				dataY.append(x[nFeatures+1])
				dataZ.append(x[nFeatures+2])
				dataE.append(x[nFeatures+3])
				dataES.append(x[nFeatures+4])
				dataESSum.append(np.max(np.sum(cols,axis=0)))

				count += 1

				if count%1000 == 0:
					print("%i events processed (%.2f%%)"%(count, float(count)/float(nEvents)*100))

	fig = plt.figure(figsize=(12,8))
	ax1 = fig.add_subplot(2,3,1)
	ax2 = fig.add_subplot(2,3,2)
	ax3 = fig.add_subplot(2,3,3)
	ax4 = fig.add_subplot(2,3,4)
	ax5 = fig.add_subplot(2,3,5)
	ax6 = fig.add_subplot(2,3,6)

	ax1.hist(dataX, np.linspace(-200,200,80), histtype='step', color='b')
	ax2.hist(dataY, np.linspace(-200,200,80), histtype='step', color='b')
	ax3.hist(dataZ, np.linspace(-200,200,80), histtype='step', color='b')
	ax4.hist(dataE, np.linspace(300,3500,80), histtype='step', color='b')
	ax5.hist(dataES, np.linspace(300,3500,80), histtype='step', color='b')

	H, xedges, yedges = np.histogram2d(dataE, dataES, bins=(np.linspace(0,3500,80), np.linspace(0,3500,80)))
	ax6.imshow(np.flipud(H.T), interpolation='nearest', extent=[0, 3500, 0, 3500], aspect='auto', cmap='summer')

	ax1.set_xlim([-200, 200])
	ax1.set_ylim(ymin=0)
	ax2.set_xlim([-200, 200])
	ax2.set_ylim(ymin=0)
	ax3.set_xlim([-200, 200])
	ax3.set_ylim(ymin=0)
	ax4.set_xlim([300, 3500])
	ax4.set_ylim(ymin=0)
	ax5.set_xlim([300, 3500])
	ax5.set_ylim(ymin=0)
	#ax6.set_xlim([300, 3500])
	#ax6.set_ylim(ymin=0)

	ax1.set_xlabel('x (mm)')
	ax2.set_xlabel('y (mm)')
	ax3.set_xlabel('z (mm)')
	ax4.set_xlabel('energy (keV)')
	ax5.set_xlabel('scintillation energy (keV)')

	ax2.set_title('X-Y distribution')
	ax3.set_title('E-Z distribution')

	plt.show()

def PlotLablesPolished(filenames):
	reader = csv.reader(open(filenames[0]), delimiter=',')
        #nFeatures = len(next(reader)) - 4
        nFeatures = 74*350
        print("Number of features: %i"%nFeatures)

        dataX = []
        dataY = []
        dataZ = []
        dataE = []
        for inputFile in filenames:
                print("Processing file %s"%inputFile)
                nEvents = sum(1 for row in open(inputFile))
                count = 0
                with open(inputFile) as infile:
                        for line in infile:
                                x = np.fromstring(line, dtype=float, sep=',')
                                cols = np.reshape(x[0:nFeatures],(74,350))
                                dataX.append(x[nFeatures])
                                dataY.append(x[nFeatures+1])
                                dataZ.append(x[nFeatures+2])
                                dataE.append(x[nFeatures+3])

                                count += 1

                                if count%1000 == 0:
                                        print("%i events processed (%.2f%%)"%(count, float(count)/float(nEvents)*100))

	fig = plt.figure(figsize=(12,6))

	gs = gridspec.GridSpec(2,4, width_ratios=[1,4,4,1], height_ratios=[4,1])
	ax1 = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1])
	ax3 = plt.subplot(gs[2])
	ax4 = plt.subplot(gs[3])
	ax5 = plt.subplot(gs[5])
	ax6 = plt.subplot(gs[6])

	fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.1, wspace=0.1)

	ax1.hist(dataY, np.linspace(-200,200,40), facecolor='green', edgecolor='black', linewidth=1, orientation='horizontal')
	ax4.hist(dataZ, np.linspace(-200,200,40), facecolor='green', edgecolor='black', linewidth=1, orientation='horizontal')
	ax5.hist(dataX, np.linspace(-200,200,40), facecolor='green', edgecolor='black', linewidth=1)
	ax6.hist(dataE, np.linspace(500,3500,40), facecolor='green', edgecolor='black', linewidth=1)

	H1, xedges1, yedges1 = np.histogram2d(dataX, dataY, bins=(np.linspace(-200,200,40), np.linspace(-200,200,40)))
	#H1[H1 == 0.0] = np.nan
	ax2.imshow(np.flipud(H1.T), interpolation='gaussian', extent=[-200, 200, -200, 200], aspect='auto', cmap='viridis')

	H2, xedges2, yedges2 = np.histogram2d(dataE, dataZ, bins=(np.linspace(500,3500,40), np.linspace(-200,200,40)))
	#H2[H2 == 0.0] = np.nan
	im = ax3.imshow(np.flipud(H2.T), interpolation='gaussian', extent=[500, 3500, -200, 200], aspect='auto', cmap='viridis')

	cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.15])

	fig.colorbar(im, cax=cbar_ax, ticks=[0, 60, 120, 180])

	ax2.set_xlim(-200,200)
	ax3.set_xlim(500,3500)
	ax5.set_xlim(-200,200)
	ax6.set_xlim(500,3500)

	ax1.set_ylim(-200,200)
	ax2.set_ylim(-200,200)
	ax3.set_ylim(-200,200)
	ax4.set_ylim(-200,200)

	ax1.xaxis.set_ticklabels([])
	ax2.xaxis.set_ticklabels([])
	ax3.xaxis.set_ticklabels([])
	ax4.xaxis.set_ticklabels([])

	ax2.yaxis.set_ticklabels([])
	ax3.yaxis.set_ticklabels([])
	ax5.yaxis.set_ticklabels([])
	ax6.yaxis.set_ticklabels([])

	ax4.invert_xaxis()
	ax4.yaxis.tick_right()
	ax4.yaxis.set_label_position('right')

	ax1.grid(linestyle=':')
	ax2.grid(linestyle=':')
	ax3.grid(linestyle=':')
	ax4.grid(linestyle=':')
	ax5.grid(linestyle=':')
	ax6.grid(linestyle=':')

	ax1.set_ylabel('y [mm]', fontsize=14)
	ax4.set_ylabel('z [mm]', fontsize=14)
	ax5.set_xlabel('x [mm]', fontsize=14)
	ax6.set_xlabel('energy [keV]', fontsize=14)

	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputFile', type=str)

	args = parser.parse_args()

	filenames = glob.glob(args.inputFile)

	#PlotLabels(filenames)

	PlotLablesPolished(filenames)

