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

# Definition of the network architecture

# Name of the weight variables for each layer
weightName = {  0: 'WConv0',
                1: 'WConv1',
                2: 'WConv2',
                3: 'W0',
                4: 'W1',
                5: 'W2',
                6: 'W3',
                7: 'W4'}

# Name of the bias variable for each layer
biasName = {	0: 'Variable',
		1: 'Variable_1',
		2: 'Variable_2',
		3: 'Variable_3',
		4: 'Variable_4',
		5: 'Variable_5',
		6: 'Variable_6',
		7: 'Variable_7'}

# type of the layer
layerType = {	0: 'conv',
		1: 'conv',
		2: 'conv',
		3: 'fc',
		4: 'fc',
		5: 'fc',
		6: 'fc',
		7: 'fc'}

def PlotWeights(weights):
	nFiltersIn = weights.shape[2]
	nFiltersOut = weights.shape[3]
	nColsIn = int(np.ceil(np.sqrt(nFiltersIn)))
	nColsOut = int(np.ceil(np.sqrt(nFiltersOut)))
	nCols = nColsIn*nColsOut
	
	print(nColsIn,nColsOut)
	
	wmin = np.min(weights)
	wmax = np.max(weights)
	
	fig, axarr = plt.subplots(nCols,nCols, figsize=(3.0/74.0*weights.shape[0]*nCols,3.0/74.0*weights.shape[0]*nCols))
	fig.subplots_adjust(wspace=0,hspace=0,left=0.1,right=0.9,top=0.9,bottom=0.1)
	
	axarr = axarr.ravel()
	
	for i, ax in enumerate(axarr):
		n = np.unravel_index(i, (nCols,nCols))
		outLayerCoords_x = int(float(n[1])/float(nColsIn))
		outLayerCoords_y = int(float(n[0])/float(nColsIn))
		outLayer = outLayerCoords_y*nColsOut+outLayerCoords_x
	
		inLayerCoords_x = n[1] - outLayerCoords_x*nColsIn
		inLayerCoords_y = n[0] - outLayerCoords_y*nColsIn
		inLayer = inLayerCoords_y*nColsIn+inLayerCoords_x
	
		print(inLayer,outLayer)

		if inLayer > weights.shape[2]-1 or outLayer > weights.shape[3]-1:
			ax.xaxis.set_visible(False)
                	ax.yaxis.set_visible(False)

			continue
	
		w = np.ones((weights.shape[0],weights.shape[1]))
		if inLayer < weights.shape[2] and outLayer < weights.shape[3]:
			w = weights[:,:,inLayer,outLayer]
	
		cax = ax.imshow(w, interpolation='nearest', cmap='Greys', vmin=wmin, vmax=wmax)
	
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
	
	#cbar = fig.colorbar(cax, ax=axarr.ravel().tolist(), orientation='vertical')
	
	plt.show(block=False)

def PlotActivationsConv(layer, img, model, plot3D=False):
	reader = pywrap_tensorflow.NewCheckpointReader(model)

	x = tf.placeholder(tf.float32, shape=[None, 74*350])
        x_image = tf.reshape(x, [-1, 74, 350, 1])

	w = np.empty((1,layer), dtype=object)
	b = np.empty((1,layer), dtype=object)
	#H = np.empty((1,layer), dtype=object)
	#H_pool = np.empty((1,layer), dtype=object)

	H = 0
	H_pool = 0

	for l in range(layer):
		weights = reader.get_tensor(weightName[l])
        	bias = reader.get_tensor(biasName[l])

		#w[l] = weights
		#b[l] = bias

		if l == 0:
			H = tf.nn.relu(tf.nn.conv2d(x_image, weights, strides=[1,1,1,1], padding='SAME') + bias)
		else:
			H = tf.nn.relu(tf.nn.conv2d(H_pool, weights, strides=[1,1,1,1], padding='SAME') + bias)

		H_pool = tf.nn.max_pool(H, ksize=[1,2,3,1], strides=[1,2,3,1], padding='SAME')

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		in_image, out_image = sess.run([x_image, H], feed_dict={x: img})

		sess.close()

	print out_image.shape

	nKernels = weights.shape[3]
	#nKernels = 2

	if plot3D:
		X, Y = np.meshgrid(np.linspace(0,350,350), np.linspace(0,74,74))
		Z = np.ones(X.shape)

		fig = plt.figure(figsize=(10,4))

		ax2 = plt.subplot(projection='3d')

		for i in range(nKernels):
			ax2.contourf(X, np.flipud(out_image[0,:,:,i]), Y, alpha=1, zdir='y', offset=20*i, interpolation='none', aspect='auto', cmap='summer')

		ax2.set_xlim(0,350)
		ax2.set_ylim(0,i*20)
		ax2.set_zlim(0,74)

		ax2.can_zoom()

		ax2._axis3don = False

		ax2.view_init(elev=25, azim=-35)

		fig.subplots_adjust(left=0,right=1.0,top=1.0,bottom=0)

		plt.show()
	else:
		nCol = int(np.ceil(np.sqrt(weights.shape[3])))

		fig, axarr = plt.subplots(nCol, nCol, figsize=(6,6))

		axarr = axarr.ravel()

		for i, ax in enumerate(axarr):
			if i > out_image.shape[3]-1:
				ax.xaxis.set_visible(False)
                        	ax.yaxis.set_visible(False)

				continue

			ax.imshow(out_image[0,:,:,i], interpolation='nearest', aspect='auto', cmap='summer')

			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)

		fig.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.0,hspace=0.0)

		plt.show(block=False)

def PlotImage(img):
	img = np.reshape(img, (74, 350))

	fig = plt.figure(figsize=(3,3))

	ax = plt.subplot(111)

	ax.imshow(img, interpolation='nearest', aspect='auto', cmap='summer')

	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)

	fig.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9)

	plt.show(block=False)

def PrintVariables(reader):
	var_to_shape_map = reader.get_variable_to_shape_map()

	for key in var_to_shape_map:
    		print("tensor_name: ", key)
    		#print(reader.get_tensor(key))

def GetImage(filename):
	filenames = []
	filenames.append(filename)

	data_set = DataSet(filenames, None, True, False)
        batch = data_set.GetBatch(1)

	x = tf.placeholder(tf.float32, shape=[None, 74*350])
	x_image = tf.reshape(x, [-1, 74, 350, 1])

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		try:
			batch_x, batch_y = sess.run([batch[0], batch[1]])
			img = sess.run(x_image, feed_dict={x: batch_x})
		except tf.errors.OutOfRangeError:
                        print('Done training -- epoch limit reached')
                finally:
                        coord.request_stop()

		coord.join(threads)
                sess.close()

	return img

def GetImageCSV(filename):
	with open(filename) as infile:
		count = 0
		for line in infile:
			x = np.fromstring(line, dtype=float, sep=',')

			img = x[0:74*350]

			count += 1

			if count == 2:
				break

	return np.reshape(img, (1,74*350))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputFile', type=str)
	parser.add_argument('--model', type=str)
	parser.add_argument('--layer', type=str)

	args = parser.parse_args()
	layer = int(args.layer)

	reader = pywrap_tensorflow.NewCheckpointReader(args.model)

	weights = reader.get_tensor(weightName[layer])
	bias = reader.get_tensor(biasName[layer])

	x_image = GetImageCSV(args.inputFile)

	PlotImage(x_image)
	#PlotWeights(weights)

	for i in range(layer+1):
		if layerType[i] == 'conv':
			PlotActivationsConv(i+1, x_image, args.model)
		elif layerType[i] == 'fc':
			PlotActivationsFC(weights,bias,x_image)

	raw_input("Enter...")


