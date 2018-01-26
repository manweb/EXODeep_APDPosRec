# Definition of the network architecture

num_layers = 8

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
biasName = {    0: 'Variable',
                1: 'Variable_1',
                2: 'Variable_2',
                3: 'Variable_3',
                4: 'Variable_4',
                5: 'Variable_5',
                6: 'Variable_6',
                7: 'Variable_7'}

# type of the layer
layerType = {   0: 'conv',
                1: 'conv',
                2: 'conv',
                3: 'fc',
                4: 'fc',
                5: 'fc',
                6: 'fc',
                7: 'fc'}

# size of max pooling
maxPoolingSize = {	0: [1,2,3,1],
			1: [1,2,3,1],
			2: [1,2,3,1],
			3: [],
			4: [],
			5: [],
			6: [],
			7: []}

