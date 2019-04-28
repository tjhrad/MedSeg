#!/bin/python

import os
import copy
from random import shuffle
import itertools
import random

import numpy as np

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from nilearn.image import new_img_like, resample_to_img
import nibabel as nib

def get_generator_and_steps(data_file, batch_size, n_labels, data_split=0.8, labels=None, augment=False,
										   augment_flip=True, augment_distortion_factor=0.25, patch_shape=None,
										   validation_patch_overlap=0, training_patch_start_offset=None,
										   validation_batch_size=None, skip_blank=True, permute=False,model_2dimensional=False):#training_keys_file, validation_keys_file,
	if not validation_batch_size:
		validation_batch_size = batch_size

	training_list, validation_list = get_validation_split(data_file, data_split=data_split)

	training_generator = data_generator(data_file, training_list,
										batch_size=batch_size,
										n_labels=n_labels,
										labels=labels,
										augment=augment,
										augment_flip=augment_flip,
										augment_distortion_factor=augment_distortion_factor,
										patch_shape=patch_shape,
										patch_overlap=validation_patch_overlap,
										patch_start_offset=training_patch_start_offset,
										skip_blank=skip_blank,
										permute=permute)

	validation_generator = data_generator(data_file, validation_list,
										batch_size=validation_batch_size,
										n_labels=n_labels,
										labels=labels,
										patch_shape=patch_shape,
										patch_overlap=validation_patch_overlap,
										skip_blank=skip_blank)

	# Set the number of training and testing samples per epoch correctly TODO: THIS NEEDS TO BE FIXED>>
	if model_2dimensional:
		num_training_steps =  batch_size
		num_validation_steps =  validation_batch_size
	else:
		num_training_steps = get_number_of_steps(get_number_of_patches(data_file, training_list, patch_shape,skip_blank=skip_blank,
																   patch_start_offset=training_patch_start_offset,
																   patch_overlap=0), batch_size)

		num_validation_steps = get_number_of_steps(get_number_of_patches(data_file, validation_list, patch_shape,skip_blank=skip_blank,
																	 patch_overlap=validation_patch_overlap),validation_batch_size)

	print("Number of training steps: " + str(num_training_steps))
	print("Number of validation steps: " + str(num_validation_steps))

	return training_generator, validation_generator, num_training_steps, num_validation_steps



def get_number_of_steps(n_samples, batch_size):
	print("n samps " + str(n_samples))
	if n_samples <= batch_size:
		return n_samples
	elif np.remainder(n_samples, batch_size) == 0:
		return n_samples//batch_size
	else:
		return n_samples//batch_size + 1


def get_number_of_patches(data_file, index_list, patch_shape=None, patch_overlap=0, patch_start_offset=None,
						  skip_blank=True):
	if patch_shape:
		index_list = create_patch_index_list(index_list, data_file.root.data.shape[-3:], patch_shape, patch_overlap,
											 patch_start_offset)
		count = 0
		for index in index_list:
			x_list = list()
			y_list = list()
			add_data(x_list, y_list, data_file, index, skip_blank=skip_blank, patch_shape=patch_shape)
			if len(x_list) > 0:
				count += 1
		return count
	else:
		return len(index_list)


def get_validation_split(data_file, data_split=0.8):
	print("Creating validation split...")
	nb_samples = data_file.root.data.shape[0]
	sample_list = list(range(nb_samples))
	training_list, validation_list = split_list(sample_list, split=data_split)
	return training_list, validation_list


def split_list(input_list, split=0.8, shuffle_list=True):
	if shuffle_list:
		shuffle(input_list)
	n_training = int(len(input_list) * split)
	training = input_list[:n_training]
	testing = input_list[n_training:]
	return training, testing


def data_generator(data_file, index_list, batch_size=1, n_labels=1, labels=None, augment=False, augment_flip=True,
				   augment_distortion_factor=0.25, patch_shape=None, patch_overlap=0, patch_start_offset=None,
				   shuffle_index_list=True, skip_blank=True, permute=False):
	orig_index_list = index_list
	while True:
		x_list = list()
		y_list = list()
		if patch_shape:
			index_list = create_patch_index_list(orig_index_list, data_file.root.data.shape[-3:], patch_shape,
												 patch_overlap, patch_start_offset)
		else:
			index_list = copy.copy(orig_index_list)

		if shuffle_index_list:
			shuffle(index_list)

		while len(index_list) > 0:
			index = index_list.pop()	
			add_data(x_list, y_list, data_file, index, augment=augment, augment_flip=augment_flip,
				 augment_distortion_factor=augment_distortion_factor, patch_shape=patch_shape,
				 skip_blank=skip_blank, permute=permute)
			if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
				yield convert_data_to_nparray(x_list, y_list, n_labels=n_labels, labels=labels)
				x_list = list()
				y_list = list()


def create_patch_index_list(index_list, image_shape, patch_shape, patch_overlap, patch_start_offset=None):
	patch_index = list()
	for index in index_list:
		if patch_start_offset is not None:
			random_start_offset = np.negative(get_random_nd_index(patch_start_offset))
			patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap, start=random_start_offset)
		else:
			patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap)
		patch_index.extend(itertools.product([index], patches))
	return patch_index


def get_random_nd_index(index_max):
	return tuple([np.random.choice(index_max[index] + 1) for index in range(len(index_max))])


def compute_patch_indices(image_shape, patch_size, overlap, start=None):
	if isinstance(overlap, int):
		overlap = np.asarray([overlap] * len(image_shape))
	if start is None:
		n_patches = np.ceil(image_shape / (patch_size - overlap))
		overflow = (patch_size - overlap) * n_patches - image_shape + overlap
		start = -np.ceil(overflow/2)
	elif isinstance(start, int):
		start = np.asarray([start] * len(image_shape))
	stop = image_shape + start
	step = patch_size - overlap
	return get_set_of_patch_indices(start, stop, step)


def get_set_of_patch_indices(start, stop, step):
	return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
							   start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)


def convert_data_to_nparray(x_list, y_list, n_labels=1, labels=None):
	x = np.asarray(x_list)
	y = np.asarray(y_list)
	if n_labels == 1:
		y[y > 0] = 1
	elif n_labels > 1:
		y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)
	return x, y

def get_multi_class_labels(data, n_labels, labels=None):
	new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
	y = np.zeros(new_shape, np.int8)
	for label_index in range(n_labels):
		if labels is not None:
			y[:, label_index][data[:, 0] == labels[label_index]] = 1
		else:
			y[:, label_index][data[:, 0] == (label_index + 1)] = 1
	return y

def add_data(x_list, y_list, data_file, index, augment=False, augment_flip=False, augment_distortion_factor=0.25,
			 patch_shape=False, skip_blank=True, permute=False):
	data, mask = get_data_from_file(data_file, index, patch_shape=patch_shape)
	if augment:
		if patch_shape is not None:
			affine = data_file.root.affine[index[0]]
		else:
			affine = data_file.root.affine[index]
		data, mask = augment_data(data, mask, affine, flip=augment_flip, scale_deviation=augment_distortion_factor)

	if permute:
		if data.shape[-3] != data.shape[-2] or data.shape[-2] != data.shape[-1]:
			raise ValueError("To utilize permutations, data array must be in 3D cube shape with all dimensions having "
							 "the same length.")
		data, mask = random_permutation_x_y(data, mask[np.newaxis])
	else:
		mask = mask[np.newaxis]

	if not skip_blank or np.any(mask != 0):
		x_list.append(data)
		y_list.append(mask)

def get_data_from_file(data_file, index, patch_shape=None):
	if patch_shape:
		index, patch_index = index
		data, mask = get_data_from_file(data_file, index, patch_shape=None)
		x = get_patch_from_3d_data(data, patch_shape, patch_index)
		y = get_patch_from_3d_data(mask, patch_shape, patch_index)
	else:
		x, y = data_file.root.data[index], data_file.root.mask[index, 0]
	return x, y


def get_patch_from_3d_data(data, patch_shape, patch_index):
	patch_index = np.asarray(patch_index, dtype=np.int16)
	patch_shape = np.asarray(patch_shape)
	image_shape = data.shape[-3:]
	if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
		data, patch_index = fix_out_of_bound_patch_attempt(data, patch_shape, patch_index)
	return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
				patch_index[2]:patch_index[2]+patch_shape[2]]


def fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim=3):
	image_shape = data.shape[-ndim:]
	pad_before = np.abs((patch_index < 0) * patch_index)
	pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))
	pad_args = np.stack([pad_before, pad_after], axis=1)
	if pad_args.shape[0] < len(data.shape):
		pad_args = [[0, 0]] * (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
	data = np.pad(data, pad_args, mode="edge")
	patch_index += pad_before
	return data, patch_index


def augment_data(data, mask, affine, scale_deviation=None, flip=True):
	n_dim = len(mask.shape)
	if scale_deviation:
		scale_factor = random_scale_factor(n_dim, std=scale_deviation)
	else:
		scale_factor = None
	if flip:
		flip_axis = random_flip_dimensions(n_dim)
	else:
		flip_axis = None
	data_list = list()
	for data_index in range(data.shape[0]):
		image = get_image(data[data_index], affine)
		data_list.append(resample_to_img(distort_image(image, flip_axis=flip_axis,
													   scale_factor=scale_factor), image,
										 interpolation="continuous").get_data())
	data = np.asarray(data_list)
	mask_image = get_image(mask, affine)
	mask_data = resample_to_img(distort_image(mask_image, flip_axis=flip_axis, 
		scale_factor=scale_factor),mask_image, interpolation="nearest").get_data()

	return data, mask_data


def random_scale_factor(n_dim=3, mean=1, std=0.25):
	return np.random.normal(mean, std, n_dim)


def random_flip_dimensions(n_dimensions):
	axis = list()
	for dim in range(n_dimensions):
		if random_boolean():
			axis.append(dim)
	return axis


def random_boolean():
	return np.random.choice([True, False])


def distort_image(image, flip_axis=None, scale_factor=None):
	if flip_axis:
		image = flip_image(image, flip_axis)
	if scale_factor is not None:
		image = scale_image(image, scale_factor)
	return image


def scale_image(image, scale_factor):
	scale_factor = np.asarray(scale_factor)
	new_affine = np.copy(image.affine)
	new_affine[:3, :3] = image.affine[:3, :3] * scale_factor
	new_affine[:, 3][:3] = image.affine[:, 3][:3] + (image.shape * np.diag(image.affine)[:3] * (1 - scale_factor)) / 2
	return new_img_like(image, data=image.get_data(), affine=new_affine)


def flip_image(image, axis):
	try:
		new_data = np.copy(image.get_data())
		for axis_index in axis:
			new_data = np.flip(new_data, axis=axis_index)
	except TypeError:
		new_data = np.flip(image.get_data(), axis=axis)
	return new_img_like(image, data=new_data)


def get_image(data, affine, nib_class=nib.Nifti1Image):
	return nib_class(dataobj=data, affine=affine)


def random_permutation_x_y(x_data, y_data):
	key = random_permutation_key()
	return permute_data(x_data, key), permute_data(y_data, key)


def random_permutation_key():
	return random.choice(list(generate_permutation_keys()))


def permute_data(data, key):
	data = np.copy(data)
	(rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = key
	if rotate_y:
		data = np.rot90(data, axes=(1, 3))
	if rotate_z:
		data = np.rot90(data, axes=(2, 3))
	if flip_x:
		data = data[:, ::-1]
	if flip_y:
		data = data[:, :, ::-1]
	if flip_z:
		data = data[:, :, :, ::-1]
	if transpose:
		for i in range(data.shape[0]):
			data[i] = data[i].T
	return data

def generate_permutation_keys():
	return set(itertools.product(itertools.combinations_with_replacement(range(2), 2), range(2), range(2), range(2), range(2)))


def get_multi_class_labels(data, n_labels, labels=None):
	new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
	y = np.zeros(new_shape, np.int8)
	for label_index in range(n_labels):
		if labels is not None:
			y[:, label_index][data[:, 0] == labels[label_index]] = 1
		else:
			y[:, label_index][data[:, 0] == (label_index + 1)] = 1
	return y



def get_callbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
				  learning_rate_patience=50, verbosity=1,
				  early_stopping_patience=None):
	callbacks = list()
	if learning_rate_epochs:
		callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
													   drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
	else:
		callbacks.append(ReduceLROnPlateau(monitor='val_loss',factor=learning_rate_drop, patience=learning_rate_patience,
										   verbose=verbosity))
	if early_stopping_patience:
		callbacks.append(EarlyStopping(monitor='val_loss',verbose=verbosity, patience=early_stopping_patience))
	return callbacks
