import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from architecture_medium import *

def tf_model_medium(model, input_img):
	reader = pywrap_tensorflow.NewCheckpointReader(model)

	nLayers = len(layerType)
        oldLayer = layerType[0]
	current_layer = input_img
	output = {}
        for layer in range(nLayers):
                weights = reader.get_tensor(weightName[layer])
                bias = reader.get_tensor(biasName[layer])

                print('Layer %i, type %s'%(layer,layerType[layer]))
                print('    w: %s'%str(weights.shape))
                print('    b: %s'%str(bias.shape))
                print('    max_pool: %s'%str(maxPoolingSize[layer]))

                if not layerType[layer] == oldLayer:
                        current_layer = tf.reshape(current_layer, [-1, weights.shape[0]])

                if layerType[layer] == 'conv':
                        H = tf.nn.relu(tf.nn.conv2d(current_layer, weights, strides=[1,1,1,1], padding='SAME') + bias)
                elif layer < nLayers-1:
                        H = tf.nn.relu(tf.matmul(current_layer,weights)+bias)
                else:
                        H = tf.matmul(current_layer,weights)+bias

                if layerType[layer] == 'conv':
                        current_layer = tf.nn.max_pool(H, ksize=maxPoolingSize[layer], strides=maxPoolingSize[layer], padding='SAME')
                else:
                        current_layer = H

                oldLayer = layerType[layer]

		layer_name = '%s%i'%(layerType[layer], layer)
		output[layer_name] = current_layer

	return output

