import numpy as np
import csv
import matplotlib.pyplot as plt
import argparse

def PlotWaveform(inputFile):
	histos = []
	
	nsamples = 350
	offset = 50
	
	with open(inputFile) as infile:
		for line in infile:
			x = np.fromstring(line, dtype=float, sep=',')

			plt.close('all')
		
			fig = plt.figure(figsize=(12,4))
			ax1 = fig.add_subplot(1,3,1)
			ax2 = fig.add_subplot(1,3,2)
			ax3 = fig.add_subplot(1,3,3)
		
			wf = x[0:74*350]
			labels = x[74*350:74*350+5]
			wfR = np.reshape(wf,(74,nsamples))
			wfRoffsetY = wfR-np.mean(wfR[:,:100],axis=1).reshape(74,1)+np.arange(74*offset,0,-offset).reshape(74,1)
			wfRoffsetX = np.array([np.arange(nsamples),]*74)

			img_sum = np.sum(wfR, axis=0)
                        img_std = np.std(img_sum[0:100])
                        img_max = np.max(img_sum)

			print('std: %.2f, max = %.2f'%(img_std,img_max))
		
			ax1.imshow(np.flipud(wfR), interpolation='nearest', aspect='auto', cmap='summer')
			ax2.plot(wfRoffsetX.T,wfRoffsetY.T,c='black')
			ax3.plot(np.arange(nsamples),np.sum(wfR-np.mean(wfR[:,:100],axis=1).reshape(74,1),axis=0),c='black')

			ax1.set_ylim(0,73)

			ax1.set_xlabel('Time [$\mu$s]', fontsize=14)
			ax1.set_ylabel('Channel', fontsize=14)

			ax2.set_xlabel('Time [$\mu$s]', fontsize=14)
			ax2.set_ylabel('Amplitude + offset [a.u.]', labelpad=10, fontsize=14)

			ax3.set_xlabel('Time [$\mu$s]', fontsize=14)
			ax3.set_ylabel('Amplitude [a.u.]', labelpad=10, fontsize=14)

			plt.subplots_adjust(left=0.05, right=0.95, bottom=0.13, top=0.95, wspace=0.3)

			#plt.title('E_c = %.2f, E_s = %.2f, x = (%.2f,%.2f,%.2f)'%(labels[3], labels[4], labels[0], labels[1], labels[2]))

			print('E_c = %.2f, E_s = %.2f, x = (%.2f,%.2f,%.2f)'%(labels[3], labels[4], labels[0], labels[1], labels[2]))

			plt.show(block=False)
		
			raw_input("Enter...")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputFile', type=str)

	args = parser.parse_args()

	PlotWaveform(args.inputFile)

