import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from math import sqrt, pow
import csv
import pickle as pk

class APDDisplay(object):
	def __init__(self):
		self.xPos = np.array([-3.327, 2.218, 7.764, -2.218, 3.327, -1.109, 13.309, 14.418, 15.527, 8.873, 9.982, 4.436, 16.636, 12.2, 7.764, 11.091, 6.655, 5.545, 3.327, -2.218, -7.764, 2.218, -3.327, 1.109, -13.309, -14.418, -15.527, -8.873, -9.982, -4.436, 0, -16.636, -12.2, -7.764, -11.091, -6.655, -5.545, 3.327, -2.218, -7.764, 2.218, -3.327, 1.109, -13.309, -14.418, -15.527, -8.873, -9.982, -4.436, 0, -16.636, -12.2, -7.764, -11.091, -6.655, -5.545, -3.327, 2.218, 7.764, -2.218, 3.327, -1.109, 13.309, 14.418, 15.527, 8.873, 9.982, 4.436, 16.636, 12.2, 7.764, 11.091, 6.655, 5.545])
		self.yPos = np.array([17.289, 15.368, 13.447, 11.526, 9.605, 5.763, 11.526, 5.763, 0, 7.684, 1.921, 3.842, -5.763, -9.605, -13.447, -3.842, -7.684, -1.921, -17.289, -15.368, -13.447, -11.526, -9.605, -5.763, -11.526, -5.763, 0, -7.684, -1.921, -3.842, 0, 5.763, 9.605, 13.447, 3.842, 7.684, 1.921, -17.289, -15.368, -13.447, -11.526, -9.605, -5.763, -11.526, -5.763, 0, -7.684, -1.921, -3.842, 0, 5.763, 9.605, 13.447, 3.842, 7.684, 1.921, 17.289, 15.368, 13.447, 11.526, 9.605, 5.763, 11.526, 5.763, 0, 7.684, 1.921, 3.842, -5.763, -9.605, -13.447, -3.842, -7.684,
 -1.921])

	def DisplayAPDSignal(self,data):
		plt.close('all')

		fig = plt.figure(figsize=(10,5))
		ax1 = fig.add_subplot(1,2,1)
		ax2 = fig.add_subplot(1,2,2)

		ax1.set_ylim(-20,20)
		ax2.set_ylim(-20,20)
		ax1.set_xlim(-20,20)
		ax2.set_xlim(-20,20)

		cmap = plt.cm.get_cmap('hot', 100)

		dataMin = data.min()
		dataMax = data.max()

		offsets = np.array([[0,2.218,-2.218,1.109,1.109,-1.109,-1.109],[0,0,0,-1.921,1.921,-1.921,1.921]])

		circles = []
		for i in range(74):
			colorID = int(round(100/(dataMax-dataMin)*data[i]*(1-dataMin)))
			col = cl.rgb2hex(cmap(colorID)[:3])
			for k in range(7):
				circle = plt.Circle((self.xPos[i]+offsets[0][k],self.yPos[i]+offsets[1][k]), radius=0.5, color=col)
				circles.append(circle)

		for i in range(74*7):
			if i < 37*7:
				ax1.add_patch(circles[i])
			else:
				ax2.add_patch(circles[i])

		plt.show(block=False)

	def DisplayPosition(self,data):
		fig = plt.figure(figsize=(10,5))
		ax1 = fig.add_subplot(1,2,1)
		ax2 = fig.add_subplot(1,2,2)

		ax1.set_ylim(-200,200)
		ax2.set_ylim(-200,200)
		ax1.set_xlim(-200,200)
		ax2.set_xlim(-200,200)

		circle1 = plt.Circle((data[0],data[1]), radius=2, color='b', fill=False)
		circle2 = plt.Circle((data[2],data[1]), radius=2, color='b', fill=False)

		ax1.add_patch(circle1)
		ax2.add_patch(circle2)

		plt.show(block=False)

	def DisplayPosition(self,data,trueData,showDist=True,filename='',title='',scaleLabels=False):
		fig = plt.figure(figsize=(10,5))
		ax1 = fig.add_subplot(1,2,1)
		ax2 = fig.add_subplot(1,2,2)

		ax1.set_ylim(-250,250)
		ax2.set_ylim(-250,250)
		ax1.set_xlim(-250,250)
		ax2.set_xlim(-250,250)

		plt.subplots_adjust(left=0.085, right=0.915)

		fig.suptitle(title)

		ax1.set_xlabel('x (mm)')
		ax1.set_ylabel('y (mm)')

		ax2.set_xlabel('z (mm)')
		ax2.set_ylabel('y (mm)')

		ax1.grid(linestyle=':')
		ax2.grid(linestyle=':')

		fv_circle = plt.Circle((0,0), radius=183.2, color=(0,0,0), fill=False)
		fv_rectangle = plt.Rectangle((-198.4,-183.2), 2*198.4, 2*183.2, color=(0,0,0), fill=False)
		fv_cathode = mlines.Line2D((0,0), (-183.2,183.2), ls='--', color=(0,0,0))

		ax1.add_patch(fv_circle)
		ax2.add_patch(fv_rectangle)
		ax2.add_line(fv_cathode)

		if scaleLabels:
			data *= 400
			data -= 200
			trueData *= 400
			trueData -= 200

		circles1 = []
		circles2 = []
		circles3 = []
		circles4 = []

		numSamples = trueData.shape[0]
		for i in range(numSamples):
			circles1.append(plt.Circle((data[i,0],data[i,1]), radius=3, color='r', fill=False))
			circles2.append(plt.Circle((data[i,2],data[i,1]), radius=3, color='r', fill=False))
			circles3.append(plt.Circle((trueData[i,0],trueData[i,1]), radius=2, color='g'))
			circles4.append(plt.Circle((trueData[i,2],trueData[i,1]), radius=2, color='g'))
	
			l1 = mlines.Line2D([data[i,0],trueData[i,0]],[data[i,1],trueData[i,1]], linewidth=1, color=(0.7,0.7,0.7))
			l2 = mlines.Line2D([data[i,2],trueData[i,2]],[data[i,1],trueData[i,1]], linewidth=1, color=(0.7,0.7,0.7))
	
			#ax1.add_patch(circle1)
			#ax2.add_patch(circle2)
			ax1.add_patch(circles3[i])
			ax2.add_patch(circles4[i])

			if showDist:	
				ax1.add_line(l1)
				ax2.add_line(l2)
		
				#ax1.text(data[0]+5, data[1]+5, r'%.1f'%(sqrt(pow(data[i,0]-trueData[i,0],2)+pow(data[i,1]-trueData[i,1],2))), fontsize=10)
				#ax2.text(data[2]+5, data[1]+5, r'%.1f'%(sqrt(pow(data[i,2]-trueData[i,2],2)+pow(data[i,1]-trueData[i,1],2))), fontsize=10)

		for i in range(numSamples):
			ax1.add_patch(circles1[i])
			ax2.add_patch(circles2[i])

		circle1 = plt.Circle((-999, -999), radius=3, color='r', fill=False, label='predicted')
		circle2 = plt.Circle((-999, -999), radius=2, color='g', label='true')
		circle3 = plt.Circle((-999, -999), radius=2, color=(0,0,0), fill=False, label='fid. volume')

		ax1.add_patch(circle1)
		ax1.add_patch(circle2)
		ax1.add_patch(circle3)

		ax1.legend(bbox_to_anchor=(0.75, 0.75), loc='lower left')

		if filename:
			if filename.endswith('.pickle'):
				with open(filename,'w') as pickleFile:
					pk.dump(fig, pickleFile)
			else:
				plt.savefig(filename)
			print('Plot saved in %s'%filename)

			plt.close()
		else:
			plt.show()

	def PlotEnergyHisto(self,data,trueData,filename=''):
		fig = plt.figure(figsize=(10,6))

		plt.hist(data[:,0], np.linspace(500,3500,100), histtype='step', color='b', label='predicted')
		plt.hist(trueData[:,0], np.linspace(500,3500,100), histtype='step', color='g', label='true')

		plt.legend(bbox_to_anchor=(0.75, 0.75), loc='lower left')

		plt.xlabel('energy (keV)')
		plt.ylabel('entries / 30 keV')

		if filename:
			if filename.endswith('.pickle'):
				with open(filename,'w') as pickleFile:
					pk.dump(fig, pickleFile)
			else:
				plt.savefig(filename)
			print("Plot saved in %s"%filename)
		else:
			plt.show()

	def Plot2DEnergyHisto(self,data,trueData):
		fig = plt.figure(figsize=(8,8))

		plt.hist2d(trueData[:,0], data[:,0], bins=50, cmap='viridis')

		plt.xlabel('charge (keV)')
		plt.ylabel('predicted scintillation (keV)')

		plt.show()

	def GetScintSpectrum(self,inputFile):
		reader = csv.reader(open(inputFile), delimiter=',')
		n = len(next(reader))
		nFeatures = 74
		if n > 100:
			nFeatures = 74*350

		if not nFeatures + 5 == n:
			print("This file doesn't contain the scintillation label")

			return

		scint = []
		with open(inputFile) as infile:
			for line in infile:
				x = np.fromstring(line, dtype=float, sep=',')
				scint.append(x[n-1])

		return scint

	def PlotEnergyHistos(self,data,trueData,title='',filename='',inputFile=''):
		fig = plt.figure(figsize=(15,6))
		ax1 = fig.add_subplot(1,2,1)
		ax2 = fig.add_subplot(1,2,2)

		ax1.hist(data[:,0], np.linspace(500,3500,100), histtype='step', color='b', label='predicted')
		ax1.hist(trueData[:,0], np.linspace(500,3500,100), histtype='step', color='g', label='true')

		if inputFile:
			scint = self.GetScintSpectrum(inputFile)
			ax1.hist(scint, np.linspace(500,3500,100), histtype='step', color='r', label='scint')

		ax1.legend(bbox_to_anchor=(0.6, 0.75), loc='lower left')

		ax1.set_xlabel('energy (keV)')
		ax1.set_ylabel('entries / 30 keV')

		ax2.hist2d(trueData[:,0], data[:,0], bins=[np.linspace(900,3500,65), np.linspace(900,3500,65)], cmap='viridis')

		ax2.set_xlabel('charge (keV)')
		ax2.set_ylabel('predicted scintillation (keV)')

		if title:
			plt.suptitle(title)

		if filename:
			if filename.endswith('.pickle'):
				with open(filename,'w') as pickleFile:
					pk.dump(fig, pickleFile)
			else:
				plt.savefig(filename)
			print("Plot saved in %s"%filename)
		else:
			plt.show()

	def PlotHistos(self,data,trueData,title='',filename='',trainEnergy=False,inputFile=''):
		if trainEnergy:
			self.PlotEnergyHistos(data,trueData,title,filename,inputFile)

			return

		fig = plt.figure(figsize=(12,4))

		gs = gridspec.GridSpec(2,3, width_ratios=[1,1,1], height_ratios=[4,1])
		ax1 = plt.subplot(gs[0])
		ax2 = plt.subplot(gs[1])
		ax3 = plt.subplot(gs[2])
		ax4 = plt.subplot(gs[3])
		ax5 = plt.subplot(gs[4])
		ax6 = plt.subplot(gs[5])

		#data[data == 0] = 1e-9
		#trueData[trueData == 0] = 1e-9

		h1_true, bins,_ = ax1.hist(trueData[:,0], np.linspace(-200,200,40), facecolor='green', edgecolor='black', label='true')
		h1_pred,_ ,_ = ax1.hist(data[:,0], np.linspace(-200,200,40), histtype='step', color='red', linewidth=1.5, label='predicted')

		h2_true,_ ,_ = ax2.hist(trueData[:,1], np.linspace(-200,200,40), facecolor='green', edgecolor='black', label='true')
		h2_pred,_ ,_ = ax2.hist(data[:,1], np.linspace(-200,200,40), histtype='step', color='red', linewidth=1.5, label='predicted')

		h3_true,_ ,_ = ax3.hist(trueData[:,2], np.linspace(-200,200,40), facecolor='green', edgecolor='black', label='true')
		h3_pred,_ ,_ = ax3.hist(data[:,2], np.linspace(-200,200,40), histtype='step', color='red', linewidth=1.5, label='predicted')

		dataRes1 = np.subtract(h1_pred, h1_true)
		dataRes1 = np.divide(dataRes1, h1_true, out=np.zeros_like(dataRes1), where=h1_true!=0)

		dataRes2 = np.subtract(h2_pred, h2_true)
		dataRes2 = np.divide(dataRes2, h2_true, out=np.zeros_like(dataRes2), where=h2_true!=0)

		dataRes3 = np.subtract(h3_pred, h3_true)
		dataRes3 = np.divide(dataRes3, h3_true, out=np.zeros_like(dataRes3), where=h3_true!=0)

		#dataRes1 = data[:,0]-trueData[:,0]
		#dataRes2 = data[:,1]-trueData[:,1]
		#dataRes3 = data[:,2]-trueData[:,2]

		#ax4.hist(dataRes1, np.linspace(-200,200,80), histtype='step', color='red')
		#ax5.hist(dataRes2, np.linspace(-200,200,80), histtype='step', color='red')
		#ax6.hist(dataRes3, np.linspace(-200,200,80), histtype='step', color='red')

		binWidth = np.absolute(bins[1]-bins[0])
		nBins = dataRes1.shape[0]

		ax4.bar(bins[:-1], dataRes1*100.0, np.ones(nBins)*binWidth, align='edge', facecolor='blue', edgecolor='black', label='true')

		ax5.bar(bins[:-1], dataRes2*100.0, np.ones(nBins)*binWidth, align='edge', facecolor='blue', edgecolor='black', label='true')

		ax6.bar(bins[:-1], dataRes3*100.0, np.ones(nBins)*binWidth, align='edge', facecolor='blue', edgecolor='black', label='true')

		ax1.legend(bbox_to_anchor=(0.05, 0.75), loc='lower left')

		ax1.set_xlim(-200,200)
		ax2.set_xlim(-200,200)
		ax3.set_xlim(-200,200)
		ax4.set_xlim(-200,200)
		ax5.set_xlim(-200,200)
		ax6.set_xlim(-200,200)

		ax1.grid(linestyle=':')
		ax2.grid(linestyle=':')
		ax3.grid(linestyle=':')
		ax4.grid(linestyle=':')
		ax5.grid(linestyle=':')
		ax6.grid(linestyle=':')

		ax4.set_xlabel('x (mm)')
		ax1.set_ylabel('entries / 10mm')
		ax4.set_ylabel('residual (%)')

		ax5.set_xlabel('y (mm)')
		#ax2.set_ylabel('entries / 10mm')
		#ax5.set_ylabel('(pred-true)/true (%%)')

		ax6.set_xlabel('z (mm)')
		#ax3.set_ylabel('entries / 10mm')
		#ax6.set_ylabel('(pred-true)/true (%%)')

		ax1.xaxis.set_ticklabels([])
		ax2.xaxis.set_ticklabels([])
		ax3.xaxis.set_ticklabels([])

		fig.subplots_adjust(left=0.06, right=0.95, bottom=0.11, top=0.95, wspace=0.2, hspace=0.1)

		if title:
			plt.suptitle(title)

		if filename:
			if filename.endswith('.pickle'):
				with open(filename,'w') as pickleFile:
					pk.dump(fig, pickleFile)
			else:
				plt.savefig(filename)
			print("Plot saved in %s"%filename)
		else:
			plt.show()

