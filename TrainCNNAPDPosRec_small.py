import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle as pk
from utilities import DataSet
from display import APDDisplay

# Get weight variable
def weight_variable(name, shape):
	var = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
	return var

# Get bias variable
def bias_variable(shape):
	const = tf.constant(0.1, shape=shape)
	return tf.Variable(const)

# 2D convolution of input x with filter W
def conv2d(x, W, strides):
	return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

# Max pooling of size s
def max_pool(x, s):
	return tf.nn.max_pool(x, ksize=[1,s[0],s[1],1], strides=[1,s[0],s[1],1], padding='SAME')

# Build the deep neural network
#
# The input is a 74x350 image with the 74 APD channels arranged in rows
# of 350 samples. The network consists of 3 convolutional layers followed
# by 4 fully connected layers. Between the convolutional layers max pooling
# of size 2x2 is applied
#
# Sizes:    Conv1:  74x350x16
#           Conv2:  37x117x32
#           Conv3:  19x39x64
#           Conv4:  10x20x128
#           DNN1:   1024x512
#           DNN2:   512x128

def deepcnn(x, trainEnergy = False):
	# First convolutional layer
	WConv0 = weight_variable("WConv0", [5,5,1,16])
	bConv0 = bias_variable([16])
	HConv0 = tf.nn.relu(conv2d(x, WConv0, [1,1,1,1]) + bConv0)
	HPool0 = max_pool(HConv0, [2,3])

	# Second convolutional layer
	WConv1 = weight_variable("WConv1", [5,5,16,32])
	bConv1 = bias_variable([32])
	HConv1 = tf.nn.relu(conv2d(HPool0, WConv1, [1,1,1,1]) + bConv1)
	HPool1 = max_pool(HConv1, [2,3])

	# Third convolutional layer
	WConv2 = weight_variable("WConv2", [5,5,32,64])
	bConv2 = bias_variable([64])
	HConv2 = tf.nn.relu(conv2d(HPool1, WConv2, [1,1,1,1]) + bConv2)
	HPool2 = max_pool(HConv2, [2,2])

	# Fourth convolutional layer
	WConv3 = weight_variable("WConv3", [5,5,64,128])
	bConv3 = bias_variable([128])
	HConv3 = tf.nn.relu(conv2d(HPool2, WConv3, [1,1,1,1]) + bConv3)
	HPool3 = max_pool(HConv3, [5,5])

	# Flatten structure
	HPool3_flat = tf.reshape(HPool3, [-1, 1024])

	# First hidden layer
	W0 = weight_variable("W0", [1024,512])
	b0 = bias_variable([512])
	H0 = tf.nn.relu(tf.matmul(HPool3_flat,W0)+b0)

	# Add dropout
	keep_prob = tf.placeholder(tf.float32)
	H0_drop = tf.nn.dropout(H0,keep_prob)
	tf.add_to_collection('keep_prob', keep_prob)

	# Second hidden layer
	W1 = weight_variable("W1", [512,128])
	b1 = bias_variable([128])
	H1 = tf.nn.relu(tf.matmul(H0_drop,W1)+b1)

	# Third hidden layer
	#W2 = weight_variable("W2", [256,128])
	#b2 = bias_variable([128])
	#H2 = tf.nn.relu(tf.matmul(H1,W2)+b2)

	# Fourth hidden layer
	#W3 = weight_variable("W3", [200,100])
	#b3 = bias_variable([100])
	#H3 = tf.nn.relu(tf.matmul(H2,W3)+b3)

	# Output layer
	nOutput = 3
	if trainEnergy:
		nOutput = 1
	W3 = weight_variable("W3", [128,nOutput])
	b3 = bias_variable([nOutput])
	y = tf.matmul(H1,W3)+b3
	tf.add_to_collection('y', y)

	# add regularization terms
	regularization = tf.placeholder(tf.float32)
	tf.add_to_collection('regularization', regularization)
	reg_term = regularization*(tf.nn.l2_loss(WConv0)+tf.nn.l2_loss(bConv0)+tf.nn.l2_loss(WConv1)+tf.nn.l2_loss(bConv1)+tf.nn.l2_loss(WConv2)+tf.nn.l2_loss(bConv2)+tf.nn.l2_loss(WConv3)+tf.nn.l2_loss(bConv3)+tf.nn.l2_loss(W0)+tf.nn.l2_loss(b0)+tf.nn.l2_loss(W1)+tf.nn.l2_loss(b1)+tf.nn.l2_loss(W3)+tf.nn.l2_loss(b3))

	return y, keep_prob, regularization, reg_term

# Get filenames for training
def get_files(location = '.'):
	filenames = []
	#filenames = tf.train.match_filenames_once("./data/train/APDmaxSignalsNormalized_*.csv")

	filenames.append(location+'APDWFSignalsEnergyThreshold_Flattened_Merged_Train_112017.csv')

	return filenames

def scale_batch(data):
	data += 200
	data /= 400

	return data

def unscale_batch(data):
	data *= 400
	data -= 200

	return data

# Function which runs some checks to make sure the network is built properly
def VerifyNetwork(sess, x, y_, loss, keep_prob, regularization, batch_x, batch_y):
	# Let's make sure the data is centered
	print("The average mean over the features is: %f")%(np.mean(batch_x))

	# Plot two randonly selected features to make sure it is zero centered
	idx = np.random.randint(25900, size=2)
	x_plot = batch_x[:,idx[0]]
	y_plot = batch_x[:,idx[1]]

	fig = plt.figure(figsize=(5,5))
	plt.scatter(x_plot, y_plot)
	plt.show()

	# Evaluate the loss function at different regularizations to make sure it increases with increasing regularization
	eval_loss1 = sess.run(loss, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, regularization: 1e-5})
	eval_loss2 = sess.run(loss, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, regularization: 1e-2})

	print("The loss function with reg = 1e-5 is: %f and with reg = 1e-2: %f"%(eval_loss1,eval_loss2))
# Main function

def main(optimizer = 'Adam', numEpochs = None, batchSize = 500, maxTrainSteps = 10000, learningRate = 0.001, regTerm = 1e-5, dropout = 1.0, outDir = './output/', trainingSet='', testSet='', model='', trainEnergy = False, prefix='', savePredPlots = False, scaleLabels = False):
	print("#################################################")
	print("  Training neural network with")
	print("    optimizer: 		%s"%optimizer)
	if numEpochs:
		print("    numEpochs: 		%i"%numEpochs)
	else:
		print("    numEpochs:		None")
	print("    batchSize: 		%i"%batchSize)
	print("    maxTrainSteps: 	%i"%maxTrainSteps)
	print("    learningRate:	%.7f"%learningRate)
	print("    regTerm:		%.7f"%regTerm)
	print("    dropout:		%.2f"%dropout)
	if trainingSet:
		print("    trainingSet:	%s"%trainingSet)
	else:
		print("    trainingSet:	default")
	if testSet:
		print("    testSet:		%s"%testSet)
	else:
		print("    testSet:		default")
	if trainEnergy:
		print("    trainEnergy:	True")
	else:
		print("    trainEnergy:	False")
	if model:
		print("    Restoring model from: %s"%model)
	print("#################################################")

	if not model:
		# input
		x = tf.placeholder(tf.float32, shape=[None, 74*350])
		tf.add_to_collection('x', x)
		
		# output
		nOutput = 3
		if trainEnergy:
			nOutput = 1
		y_ = tf.placeholder(tf.float32, shape=[None, nOutput])
		tf.add_to_collection('y_', y_)

		# Build the graph for the neural network
		x_image = tf.reshape(x, [-1, 74, 350, 1])
		y, keep_prob, regularization, reg_term = deepcnn(x_image, trainEnergy)

		# the loss function is the reduced mean square
		mse = tf.reduce_mean(tf.square(y-y_))
		tf.add_to_collection('mse', mse)

		# add regulrization terms to loss function
		loss = mse + reg_term
		tf.add_to_collection('reg_term', reg_term)
		tf.add_to_collection('loss', loss)

		# Build optimizer
		if optimizer == 'Gradient':	
			opt = tf.train.GradientDescentOptimizer(learningRate)
		elif optimizer == 'Adam':
			opt = tf.train.AdamOptimizer(learningRate)

		train_step = opt.minimize(loss)
		tf.add_to_collection('train_step', train_step)

		# Create saver to save checkpoints
		saver = tf.train.Saver()

	# Prepare the graph for plotting the loss function	
	#fig = plt.figure(figsize=(10,5))
	
	plot_x = []
	plot_y = []
	plot_y_eval = []

	# Get a list of filenames with input data
	#filenames = get_files('/global/u2/m/maweber/EXODeep/projects/PosFromAPD/data/train/')
	filenames = []
	if not trainingSet:
		trainingSet = '/global/cscratch1/sd/maweber/APDWFSignalsEnergyThreshold_Flattened_Merged_Train_112017.csv'

	filenames.append(trainingSet)
	
	data_set = DataSet(filenames, numEpochs, True, trainEnergy)
	batch = data_set.GetBatch(batchSize)

	if not testSet:
		testSet = '/global/cscratch1/sd/maweber/APDWFSignalsEnergyThreshold_Flattened_Merged_Test_112017.csv'
	cv_set_x, cv_set_y = data_set.GetCVSet(testSet,2000)
	if scaleLabels:
		cv_set_y = scale_batch(cv_set_y)

	dsp = APDDisplay()
	
	with tf.Session() as sess:
		if model:
			path, filename = os.path.split(model)

			# Restore the model
			saver = tf.train.import_meta_graph(model)

			# Load the trained parameters
			saver.restore(sess, tf.train.latest_checkpoint(path))

			x = tf.get_collection('x')[0]
			y_ = tf.get_collection('y_')[0]
			keep_prob = tf.get_collection('keep_prob')[0]
			regularization = tf.get_collection('regularization')[0]
			y = tf.get_collection('y')[0]
			mse = tf.get_collection('mse')[0]
			loss = tf.get_collection('loss')[0]
			train_step = tf.get_collection('train_step')[0]

			print("Restored model")

		# Initialize variables if not restored from model
		if not model:
			sess.run(tf.global_variables_initializer())
	
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	
		batch_x, batch_y = sess.run([batch[0], batch[1]])
		if scaleLabels:
			batch_y = scale_batch(batch_y)
		VerifyNetwork(sess, x, y_, loss, keep_prob, regularization, batch_x, batch_y)
	
		try:
			print 'Learning...'

			modelName = '%s/model%s_%.7f_%.7f'%(outDir,prefix,regTerm,learningRate)
			lossName = modelName.replace('model','loss')+'.csv'
			if model:
				modelName = model.replace('.meta','')
				modelName = modelName.replace('model','model_resumed')

			for i in range(maxTrainSteps):
				batch_x, batch_y = sess.run([batch[0], batch[1]])
				if scaleLabels:
					batch_y = scale_batch(batch_y)
				sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout, regularization: regTerm})
				current_loss = sess.run(loss, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, regularization: 0})
				if i%10==0:
					cv_loss, predicted = sess.run([loss, y], feed_dict={x: cv_set_x, y_: cv_set_y, keep_prob: 1.0, regularization: 0})
					print('step %i: train loss = %f, CV loss = %f'%(i,current_loss,cv_loss))
					plot_x.append(i)
					plot_y.append(current_loss)
					plot_y_eval.append(cv_loss)

					option = 'w'
					if i > 0 or i == 0 and model:
						option = 'a'

					f_out_loss = open(lossName, option)
					f_out_loss.write(','.join(np.char.mod('%f', np.array([i, current_loss, cv_loss])))+'\n')
					f_out_loss.close()

					if savePredPlots:
						dsp.DisplayPosition(np.array(predicted), np.array(cv_set_y), False, outDir+'/plots/plot_%i.png'%i, 'loss=%.1f'%cv_loss, scaleLabels)

				# Save every 1000th step
				if i%1000==0:
					saver.save(sess, modelName, global_step=i)
					#plt.plot(plot_x,plot_y,color='b')
					#plt.savefig(outDir+'loss_function_%i.png'%i)

				if coord.should_stop():
					break

			# Save the final model
			saver.save(sess, modelName, global_step=i)
	
			print 'Done training -- max train steps reached'
	
		except tf.errors.OutOfRangeError:
			print('Done training -- epoch limit reached')
		finally:
			coord.request_stop()
	
		coord.join(threads)
		sess.close()

	fig = plt.figure(figsize=(10,5))	
	plt.plot(plot_x,plot_y,color='b')
	plt.plot(plot_x,plot_y_eval,color='tab:orange')

	#plt.savefig(modelName.replace('model','loss_function')+'.png')
	with open(modelName.replace('model','loss_function')+'.pickle','w') as pickleFile:
		pk.dump(fig, pickleFile)

# Function that loads the trained parameters and evaluates the model on test samples
def eval(model, batchSize = 500, trainEnergy = False, evalPlotName = '', evalFilename = '', saveRawFile = False, scaleLabels = False):
	# Get the test samples
	filenames = []
	if evalFilename:
		filenames.append(evalFilename)
	else:
		filenames.append('/global/cscratch1/sd/maweber/APDWFSignalsEnergyThreshold_5058_FeatureNormalized.csv') # Th-228 S5
		#filenames.append('/global/cscratch1/sd/maweber/APDWFSignalsEnergyThreshold_5051_FeatureNormalized.csv') # Th-228 S11
		#filenames.append('/global/cscratch1/sd/maweber/APDWFSignalsEnergyThreshold_5056_FeatureNormalized.csv') # Th-228 S2
		#filenames.append('/global/cscratch1/sd/maweber/APDWFSignalsEnergyThreshold_5327_FeatureNormalized.csv') # Ra-226 S5
	data_set = DataSet(filenames, 1, True, trainEnergy)
	sample = data_set.GetBatch(1)
	batch = data_set.GetBatch(batchSize)

	path, filename = os.path.split(model)

	print("#################################################")
	print("  Evaluating model")
	print("    model:	%s"%filename)
	print("    path:	%s"%path)
	if evalFilename:
		print("    evaluating:	%s"%evalFilename)
	if evalPlotName:
		print("    saving plot:	%s"%evalPlotName)
	if trainEnergy:
		print("    trainEnergy:	True")
	else:
		print("    trainEnergy:	False")
	print("#################################################")

	if trainEnergy:
		trueData = np.empty((0,1), float)
		predictedData = np.empty((0,1), float)
	else:
		trueData = np.empty((0,3), float)
		predictedData = np.empty((0,3), float)

	with tf.Session() as sess:
		# Restore the model
		saver = tf.train.import_meta_graph(model)

		# Load the trained parameters
		saver.restore(sess, tf.train.latest_checkpoint(path))

		x = tf.get_collection('x')[0]
		y_ = tf.get_collection('y_')[0]
		keep_prob = tf.get_collection('keep_prob')[0]
		regularization = tf.get_collection('regularization')[0]
		y = tf.get_collection('y')[0]
		mse = tf.get_collection('mse')[0]
		loss = tf.get_collection('loss')[0]
		if trainEnergy:
			mean_error = tf.reduce_mean(tf.abs(y-y_)/y_)
		else:
			mean_error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y-y_),1)))

		sess.run(tf.local_variables_initializer())

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		count = 0
		nBatches = 0
		sum_mean_error = 0
		nOutliers = 0
		try:
			while 1:
				batch_x, batch_y = sess.run([batch[0], batch[1]])
				if scaleLabels:
					batch_y = scale_batch(batch_y)
				current_loss, predicted, error, mean_dist = sess.run([loss, y, mse, mean_error], feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, regularization: 0})

				trueData = np.append(trueData, np.array(batch_y), axis=0)
				predictedData = np.append(predictedData, np.array(predicted), axis=0)

				count += batchSize
				nBatches += 1
				sum_mean_error += mean_dist

				if batchSize == 1:
					center = [100,0,100]
					box_size = 50

					if predicted[0,0] < center[0]-box_size or predicted[0,0] > center[0]+box_size or predicted[0,1] < center[1]-box_size or predicted[0,1] > center[1]+box_size or predicted[0,2] < center[2]-box_size or predicted[0,2] > center[2]+box_size:
						option = 'a'
						if nOutliers == 0:
							option = 'w'

						x = np.array(batch_x)
						x = np.append(x,np.array(batch_y))

						f = open('outliers.csv', option)
						f.write(",".join(np.char.mod('%f', x))+'\n')
						f.close()

						nOutliers += 1

						print('Number of outliers: %i'%nOutliers)

				print('Current loss: %f'%current_loss)

				print("%i events processed"%count)
		except tf.errors.OutOfRangeError:
			print("All data used")
		finally:
			coord.request_stop()

		sum_mean_error /= nBatches

		unit = "mm"
		if trainEnergy:
			unit = "%"
		print("mean distance = %.2f %s"%(sum_mean_error,unit))

		title = "mean error = %.2f %s"%(sum_mean_error*100.0,unit)

		dsp = APDDisplay()
		dsp.PlotHistos(predictedData, trueData, title, evalPlotName, trainEnergy, evalFilename)
		dsp.DisplayPosition(predictedData, trueData, False, evalPlotName.replace('.','_2D.'), scaleLabels)
		#dsp.PlotEnergyHistos(predictedData, trueData)

		if saveRawFile:
			rawOutFile = evalPlotName+'.csv'

			f = open(rawOutFile, 'w')
			for i in range(trueData.shape[0]):
				line = np.append(trueData[i,:], predictedData[i,:])

				f.write(",".join(np.char.mod('%f', line))+'\n')

			f.close()

		if not trainEnergy:
			for i in range(10):
				sample_x, sample_y = sess.run([sample[0], sample[1]])
				predicted, error = sess.run([y, mean_error], feed_dict={x: sample_x, y_: sample_y, keep_prob: 1.0, regularization: 0})

				print("Sample		Predicted		True")
				print("%i		x = %f		x = %f"%(i,predicted[0][0],sample_y[0][0]))
				print("		y = %f		y = %f"%(predicted[0][1],sample_y[0][1]))
				print("		z = %f		z = %f"%(predicted[0][2],sample_y[0][2]))
				print("		error = %f"%error)
				print("-------------------------------------------------------------\n")

				dsp.DisplayAPDSignal(np.array(sample_x[0]))
				dsp.DisplayPosition(np.array(predicted[0]), np.array(sample_y[0]))

				raw_input("Press enter to continue...")

		coord.join(threads)
		sess.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--optimizer', type=str, default='Adam')
	parser.add_argument('--numEpochs', type=int, default=None)
	parser.add_argument('--batchSize', type=int, default=500)
	parser.add_argument('--maxTrainSteps', type=int, default=10000)
	parser.add_argument('--learningRate', type=float, default=0.001)
	parser.add_argument('--regTerm', type=float, default=1e-7)
	parser.add_argument('--dropout', type=float, default=1.0)
	parser.add_argument('--eval', action='store_true')
	parser.add_argument('--outDir', type=str, default='./output/')
	parser.add_argument('--model', type=str, default='')
	parser.add_argument('--trainEnergy', action='store_true')
	parser.add_argument('--evalPlotName', type=str, default='')
	parser.add_argument('--evalFilename', type=str, default='')
	parser.add_argument('--prefix', type=str, default='')
	parser.add_argument('--saveRawFile', action='store_true')
	parser.add_argument('--trainingSet', type=str, default='')
	parser.add_argument('--testSet', type=str, default='')
	parser.add_argument('--savePredPlots', action='store_true')
	parser.add_argument('--scaleLabels', action='store_true')

	args = parser.parse_args()

	if not args.eval:
		main(args.optimizer, args.numEpochs, args.batchSize, args.maxTrainSteps, args.learningRate, args.regTerm, args.dropout, args.outDir, args.trainingSet, args.testSet, args.model, args.trainEnergy, args.prefix, args.savePredPlots, args.scaleLabels)

	else:
		eval(args.model, args.batchSize, args.trainEnergy, args.evalPlotName, args.evalFilename, args.saveRawFile, args.scaleLabels)

