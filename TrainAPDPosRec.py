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

# Build the deep neural network

# the NN has 5 layers, 1 input, 3 hidden and 1 output layer
# layer sizes: 74 x 37 x 37 x 37 x 3
def deepnn(x, trainEnergy = False):
	# First hidden layer
	W0 = weight_variable("W0", [74,74])
	b0 = bias_variable([74])
	H0 = tf.nn.relu(tf.matmul(x,W0)+b0)

	# Add dropout
	keep_prob = tf.placeholder(tf.float32)
	H0_drop = tf.nn.dropout(H0,keep_prob)
	tf.add_to_collection('keep_prob', keep_prob)

	# Second hidden layer
	W1 = weight_variable("W1", [74,74])
	b1 = bias_variable([74])
	H1 = tf.nn.relu(tf.matmul(H0_drop,W1)+b1)

	# Third hidden layer
	W2 = weight_variable("W2", [74,37])
	b2 = bias_variable([37])
	H2 = tf.nn.relu(tf.matmul(H1,W2)+b2)

	# Fourth hidden layer
	W3 = weight_variable("W3", [37,37])
	b3 = bias_variable([37])
	H3 = tf.nn.relu(tf.matmul(H2,W3)+b3)

	# Output layer
	nOutput = 3
	if trainEnergy:
		nOutput = 1
	W4 = weight_variable("W4", [37,nOutput])
	b4 = bias_variable([nOutput])
	y = tf.matmul(H3,W4)+b4
	tf.add_to_collection('y', y)

	# add regularization terms
	regularization = tf.placeholder(tf.float32)
	tf.add_to_collection('regularization', regularization)
	reg_term = regularization*(tf.nn.l2_loss(W0)+tf.nn.l2_loss(b0)+tf.nn.l2_loss(W1)+tf.nn.l2_loss(b1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(b2)+tf.nn.l2_loss(W3)+tf.nn.l2_loss(b3)+tf.nn.l2_loss(W4)+tf.nn.l2_loss(b4))

	return y, keep_prob, regularization, reg_term

# Get filenames for training
def get_files(location = '.'):
	filenames = []
	#filenames = tf.train.match_filenames_once("./data/train/APDmaxSignalsNormalized_*.csv")
        filenames.append(location+'APDmaxSignalsEnergyThreshold_5003_Flattened_FeatureNormalized.csv')
	filenames.append(location+'APDmaxSignalsEnergyThreshold_5008_Flattened_FeatureNormalized.csv')
	filenames.append(location+'APDmaxSignalsEnergyThreshold_5017_Flattened_FeatureNormalized.csv')
	filenames.append(location+'APDmaxSignalsEnergyThreshold_5018_Flattened_FeatureNormalized.csv')
	filenames.append(location+'APDmaxSignalsEnergyThreshold_5019_Flattened_FeatureNormalized.csv')
        filenames.append(location+'APDmaxSignalsEnergyThreshold_5020_Flattened_FeatureNormalized.csv')

	return filenames

# Function which runs some checks to make sure the network is built properly
def VerifyNetwork(sess, x, y_, loss, keep_prob, regularization, batch_x, batch_y):
	# Let's print the first row of the batch
	print("The first row of this batch is:")
	for i in batch_x[0]:
		print i

	print("The corresponding label is:")
	for i in batch_y[0]:
		print i

	# Let's make sure the data is centered
	print("The average mean over the features is: %f")%(np.mean(batch_x))

	# Plot two randonly selected features to make sure it is zero centered
	idx = np.random.randint(74, size=2)
	x_plot = batch_x[:,idx[0]]
	y_plot = batch_x[:,idx[1]]

	fig = plt.figure(figsize=(5,5))
	plt.plot(x_plot, y_plot, 'o', color='b')
	plt.show()

	# Evaluate the loss function at different regularizations to make sure it increases with increasing regularization
	eval_loss1 = sess.run(loss, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, regularization: 1e-5})
	eval_loss2 = sess.run(loss, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, regularization: 1e-2})

	print("The loss function with reg = 1e-5 is: %f and with reg = 1e-2: %f"%(eval_loss1,eval_loss2))

# Main function
def main(optimizer = 'Adam', numEpochs = None, batchSize = 500, maxTrainSteps = 10000, learningRate = 0.001, regTerm = 1e-7, dropout = 1.0, outDir = './output/', model = '', trainEnergy = False):
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
	if trainEnergy:
		print("    trainEnergy:	True")
	else:
		print("    trainEnergy:	False")
	if model:
		print("    Restoring model from: %s"%model)
	print("#################################################")

	if not model:
		# input
		x = tf.placeholder(tf.float32, shape=[None, 74])
		tf.add_to_collection('x', x)
		
		# output
		nOutput = 3
		if trainEnergy:
			nOutput = 1
		y_ = tf.placeholder(tf.float32, shape=[None, nOutput])
		tf.add_to_collection('y_', y_)

		# Build the graph for the neural network
		y, keep_prob, regularization, reg_term = deepnn(x, trainEnergy)
		
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
	filenames = get_files('/global/u2/m/maweber/EXODeep/projects/PosFromAPD/data/train/')
	
	data_set = DataSet(filenames, numEpochs, False, trainEnergy, True)
	batch = data_set.GetBatch(batchSize)
	cv_set = data_set.GetCVSet('data/test/APDmaxSignalsEnergyThreshold_5051_FeatureNormalized.csv',5000)
	
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
		VerifyNetwork(sess, x, y_, loss, keep_prob, regularization, batch_x, batch_y)

		try:
			print 'Learning...'

			modelName = '%s/model_%.7f_%.7f'%(outDir,regTerm,learningRate)
			lossName = modelName.replace('model','loss')+'.csv'
			if model:
				modelName = model.replace('.meta','')
				modelName = modelName.replace('model','model_resumed')

			for i in range(maxTrainSteps):
				batch_x, batch_y = sess.run([batch[0], batch[1]])
				sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout, regularization: regTerm})
				current_loss = sess.run(loss, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, regularization: regTerm})
				if i%10==0:
					cv_loss = sess.run(loss, feed_dict={x: cv_set[0], y_: cv_set[1], keep_prob: 1.0, regularization: regTerm})
					print('step %i: train loss = %f, CV loss = %f'%(i,current_loss,cv_loss))
					plot_x.append(i)
					plot_y.append(current_loss)
					plot_y_eval.append(cv_loss)

					option = 'w'
                                        if i > 0:
                                                option = 'a'

                                        f_out_loss = open(lossName, option)
                                        f_out_loss.write(','.join(np.char.mod('%f', np.array([i, current_loss, cv_loss])))+'\n')
                                        f_out_loss.close()

				# Save every 1000th step
				if i%1000==0:
					saver.save(sess, modelName, global_step=i)
					#plt.plot(plot_x,plot_y,color='b')
					#plt.savefig('%s/loss_function_%.7f_%.2f_%i.png'%(outDir,regTerm,dropout,i))

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
def eval(model, batchSize = 500, trainEnergy = False):
	# Get the test samples
	filenames = []
	filenames.append('./data/test/APDmaxSignalsEnergyThreshold_5058_FeatureNormalized.csv')
	#filenames.append('./data/test/APDmaxSignalsEnergyThreshold_5051_FeatureNormalized.csv')
	#filenames.append('./data/test/APDmaxSignalsEnergyThreshold_5056_FeatureNormalized.csv')
	#filenames.append('./data/test/APDmaxSignalsEnergyThreshold_5327_FeatureNormalized.csv')
	#filenames.append('./data/test/APDmaxSignalsEnergyThreshold_5371_FeatureNormalized.csv')
	#data_set = DataSet(['./data/test/APDmaxSignalsEnergyThreshold_5058_FeatureNormalized.csv'], None) # S5
	#data_set = DataSet(['./data/test/APDmaxSignalsEnergyThreshold_5051_FeatureNormalized.csv'], None) # S11
	#data_set = DataSet(['./data/test/APDmaxSignalsEnergyThreshold_5056_FeatureNormalized.csv'], None) # S2
	#data_set = DataSet(['./data/test/APDmaxSignalsEnergyThreshold_5327_FeatureNormalized.csv'], None) # S5 Ra-226
	#data_set = DataSet(['./data/test/APDmaxSignalsEnergyThreshold_5371_FeatureNormalized.csv'], None) # S5 Co-60
	data_set = DataSet(filenames, 1, False, trainEnergy, True)
	sample = data_set.GetBatch(1)
	batch = data_set.GetBatch(batchSize)

	path, filename = os.path.split(model)

	print("#################################################")
	print("  Evaluating model")
	print("    model:	%s"%filename)
	print("    path:	%s"%path)
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
		mean_error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y-y_),1)))

		sess.run(tf.local_variables_initializer())

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		count = 0
		nBatches = 0
		sum_mean_error = 0
		try:
			while 1:
				batch_x, batch_y = sess.run([batch[0], batch[1]])
				predicted, error, mean_dist = sess.run([y, mse, mean_error], feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, regularization: 0})

				trueData = np.append(trueData, np.array(batch_y), axis=0)
				predictedData = np.append(predictedData, np.array(predicted), axis=0)

				count += batchSize
				nBatches += 1
				sum_mean_error += mean_dist

				print("%i events processed"%count)
		except tf.errors.OutOfRangeError:
			print("All data used")
		finally:
			coord.request_stop()

		sum_mean_error /= nBatches

		unit = "mm"
		if trainEnergy:
			unit = "keV"
		print("mean distance = %.2f %s"%(sum_mean_error,unit))

		dsp = APDDisplay()
		#dsp.PlotHistos(predictedData, trueData, trainEnergy)
		dsp.PlotEnergyHistos(predictedData, trueData)

		if not trainEnergy:
			for i in range(10):
				sample_x, sample_y = sess.run([sample[0], sample[1]])
				predicted, error = sess.run([y, mse], feed_dict={x: sample_x, y_: sample_y, keep_prob: 1.0, regularization: 0})

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

	args = parser.parse_args()

	if not args.eval:
		main(args.optimizer, args.numEpochs, args.batchSize, args.maxTrainSteps, args.learningRate, args.regTerm, args.dropout, args.outDir, args.model, args.trainEnergy)

	else:
		eval(args.model, args.batchSize, args.trainEnergy)

