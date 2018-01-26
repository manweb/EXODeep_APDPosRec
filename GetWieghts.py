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

with tf.Session() as sess:
	model = '/global/cscratch1/sd/maweber/output/CNN/position/112017/training2/model_0.0100000_0.0000010-15000.meta'
	path = '/global/cscratch1/sd/maweber/output/CNN/position/112017/training2/'
	saver = tf.train.import_meta_graph(model)
	saver.restore(sess, tf.train.latest_checkpoint(path))

	sess.run(tf.local_variables_initializer())

	for v in tf.global_variables():
		print v.name

	sess.close()

