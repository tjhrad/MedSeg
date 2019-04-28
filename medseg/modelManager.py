#!/bin/python

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Deconvolution2D
from keras.optimizers import Adam, Nadam
from keras.models import load_model

from medseg.cost_functions import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient, weighted_dice_coefficient_loss,weighted_dice_coefficient

K.set_image_data_format("channels_first")
try:
	from keras.engine import merge
except ImportError:
	from keras.layers.merge import concatenate

def get_new_model(input_shape, pool_size=(2, 2, 2), kernel_size=(3,3,3), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
				  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
				  batch_normalization=False, activation_last="sigmoid",activation_conv='relu',
				  pooling_strides=(2,2,2),convolution_strides=(1,1,1),model_2dimensional=False,metrics=dice_coefficient):#metrics=dice_coefficient,
	print("\n\n***********************")
	print("Generating new model...")
	print("***********************\n\n")
	inputs = Input(input_shape)
	current_layer = inputs
	levels = list()

	for layer_depth in range(depth):
		
		layer1 = create_convolution_block(input_layer=current_layer,kernel=kernel_size, n_filters=n_base_filters*(2**layer_depth),
										  batch_normalization=batch_normalization,activation=activation_conv,strides=convolution_strides,model_2dimensional=model_2dimensional)
		
		layer2 = create_convolution_block(input_layer=layer1,kernel=kernel_size, n_filters=n_base_filters*(2**layer_depth)*2,
										  batch_normalization=batch_normalization,activation=activation_conv,strides=convolution_strides,model_2dimensional=model_2dimensional)

		if layer_depth < depth - 1:
			if model_2dimensional:
				current_layer = MaxPooling2D(pool_size=pool_size)(layer2)
			else:
				current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
			levels.append([layer1,layer2, current_layer])
		else:
			current_layer = layer2
			levels.append([layer1,layer2])

	for layer_depth in range(depth-2, -1, -1):
		up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
											n_filters=current_layer._keras_shape[1],model_2dimensional=model_2dimensional)(current_layer)

		concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)

		current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],kernel=kernel_size,
												 input_layer=concat, batch_normalization=batch_normalization,
												 activation=activation_conv,strides=convolution_strides,model_2dimensional=model_2dimensional)
		current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],kernel=kernel_size,
												 input_layer=current_layer,batch_normalization=batch_normalization,
												 activation=activation_conv,strides=convolution_strides,model_2dimensional=model_2dimensional)
	if model_2dimensional:
		final_convolution = Conv2D(n_labels, (1, 1))(current_layer)
	else:
		final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
	act = Activation(activation_last)(final_convolution)

	model = Model(inputs=inputs, outputs=act)

	if not isinstance(metrics, list):
		metrics = [metrics]

	if include_label_wise_dice_coefficients and n_labels > 1: # TODO if include_label_wise_dice_coefficients is True, the model will report the dice for each label individually
		label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
		if metrics:
			metrics = metrics + label_wise_dice_metrics
		else:
			metrics = label_wise_dice_metrics
	

	model.compile(optimizer=Nadam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)

	return model


#TODO: Add ability to change kernel size
def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation='relu',
							 padding='same', strides=(1, 1, 1), instance_normalization=False, model_2dimensional=False):
	if model_2dimensional:
		layer = Conv2D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
	else:
		layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
	if batch_normalization:
		layer = BatchNormalization(axis=1)(layer)
	elif instance_normalization:
		try:
			from keras_contrib.layers import InstanceNormalization
		except ImportError:
			raise ImportError("Install keras_contrib in order to use instance normalization."
							  "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
		layer = InstanceNormalization(axis=1)(layer)

	return Activation(activation)(layer)


def create_conv_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):

    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)

def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
	output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
	return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),deconvolution=False,model_2dimensional=False):
	if deconvolution:
		if model_2dimensional:
			return Deconvolution2D(filters=n_filters, kernel_size=kernel_size,strides=strides)
		else:
			return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,strides=strides)
	else:
		if model_2dimensional:
			return UpSampling2D(size=pool_size)
		else:
			return UpSampling3D(size=pool_size)


def load_old_model(model_file):
	print("Loading pre-trained model")
	custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
					  'dice_coef': dice_coefficient, 'dice_coef_loss': dice_coefficient_loss,
					  'weighted_dice_coefficient': weighted_dice_coefficient,
					  'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss}
	try:
		from keras_contrib.layers import InstanceNormalization
		custom_objects["InstanceNormalization"] = InstanceNormalization
	except ImportError:
		pass
	try:
		return load_model(model_file, custom_objects=custom_objects)
	except ValueError as error:
		if 'InstanceNormalization' in str(error):
			raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
										  "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
		else:
			raise error