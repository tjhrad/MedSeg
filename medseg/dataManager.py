#!/bin/python

import os
import glob
import tables
import collections
import numpy as np
import nibabel as nib
import medseg.nilearn_customUtils
from skimage.transform import resize

from nilearn.image import reorder_img, resample_img, new_img_like
from nilearn.image.image import _crop_img_to as crop_img_to
from nilearn.image.image import check_niimg

def get_data_paths(sub_dir, file_names):
	training_file_paths = list()
	for path_to_subject in glob.glob(os.path.join(os.path.dirname(sub_dir),"*")):
		sub_file_paths = list()
		for file in file_names:
			sub_file_paths.append(os.path.join(path_to_subject,file + ".nii.gz"))
		training_file_paths.append(tuple(sub_file_paths))
	return training_file_paths

def write_images_to_file(file_paths,output_name,resize_shape,initial_cropping_shape=None,is_training=True,keep_dimensions=True,model_2dimensional=False, truth_dtype=np.uint8,crop=False,labels=None):
	'''
	Organizes and combines all of the training data into a single .h5 file
	'''
	if model_2dimensional:
		num_samples = len(file_paths) * resize_shape[0]
	else:
		num_samples = len(file_paths)
	print("Number of training examples: " + str(num_samples))
	num_channels = len(file_paths[0])-1 
	try:
		hdf5_file = tables.open_file(output_name, mode='w')
		filters = tables.Filters(complevel=5, complib='blosc')
		if model_2dimensional:
			data_shape = tuple([0, num_channels] + list((resize_shape[0],resize_shape[1])))
			if is_training:
				mask_shape = tuple([0,1] + list((resize_shape[0],resize_shape[1])))
		else:
			data_shape = tuple([0, num_channels] + list(resize_shape))
			if is_training:
				mask_shape = tuple([0, 1] + list(resize_shape))

		data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,filters=filters, expectedrows=num_samples)
		if is_training:
			mask_storage = hdf5_file.create_earray(hdf5_file.root, 'mask', tables.UInt8Atom(), shape=mask_shape,filters=filters, expectedrows=num_samples)
		affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4),filters=filters, expectedrows=num_samples)
		
		for subject_paths in file_paths:
			images = preprocess_images(subject_paths,resize_shape,initial_cropping_shape=initial_cropping_shape,keep_dimensions=keep_dimensions, label_index=num_channels,labels=labels)
			
			if model_2dimensional: 
				subject_data = list()
				subject_data = [image.get_fdata() for image in images]
				
				image_data = subject_data[0]
				mask_data = subject_data[1]

				subject_data = list()
				#n_slices = len(image_data[0])
				n_slices = image_data.shape[2]#len(image_data[2])
				i = 0
				while i < n_slices:
					subject_data.append(image_data[:,:,i]) # Order is sagital, coronal, then horizontal
					#subject_data.append(image_data[i]) # Order is sagital, coronal, then horizontal
					i = i + 1
				print(data_shape)
				
				for data in subject_data:
					#print(data.shape)
					data_storage.append(np.asarray(data)[np.newaxis][np.newaxis])

				subject_data = list()
				i = 0
				while i < n_slices:
					if(len(mask_data.shape)>3):
						temp_arr = np.squeeze(mask_data)
						subject_data.append(temp_arr[:,:,i])
					else:
						subject_data.append(mask_data[:,:,i])
						i = i + 1

				for data in subject_data:
					mask_storage.append(np.asarray(data, dtype=np.uint8)[np.newaxis][np.newaxis])

				affine_storage.append(np.asarray(images[0].affine)[np.newaxis])
			else:
				subject_data = [image.get_fdata() for image in images]
				print(np.amax(subject_data[0]))
				print(np.amax(subject_data[num_channels]))
				data_storage.append(np.asarray(subject_data[:num_channels])[np.newaxis])
				if is_training:
					mask_storage.append(np.asarray(subject_data[num_channels])[np.newaxis][np.newaxis])
				affine_storage.append(np.asarray(images[0].affine)[np.newaxis])

	except Exception as e:
		os.remove(output_name)
		raise e
	finally:
		hdf5_file.close()
	return output_name

def preprocess_images(in_files, resize_shape,initial_cropping_shape=None, keep_dimensions=True, label_index=None,crop=False,labels=None):
	if crop:
		print("Getting crop_parameters")
		crop_slices = get_cropping_parameters([in_files])
	else:
		crop_slices = None
	images = list()
	for index, in_file in enumerate(in_files):
		if index == label_index:
			print("label image")
			images.append(read_image(in_file,initial_cropping_shape=initial_cropping_shape, resize_shape=resize_shape,
				crop_parameters=crop_slices, keep_dimensions=keep_dimensions, is_label=True,labels=labels))
		else:
			images.append(read_image(in_file,initial_cropping_shape=initial_cropping_shape, resize_shape=resize_shape,
				crop_parameters=crop_slices, keep_dimensions=keep_dimensions))
	return images

# Returns a binary mask as an image that represents all of the foreground data the training dataset.
def get_combined_foreground_as_binary(paths):
	for i, subject_path in enumerate(paths):
		subject_foreground = get_foreground_from_set_of_files(subject_path)
		if i == 0:
			foreground = subject_foreground
		else:
			foreground[subject_foreground > 0] = 1

	return new_img_like(read_image(paths[0]), foreground)

def get_foreground_from_set_of_files(set_of_files, background_value=0, tolerance=0.00001, return_image=False):
	for i, image_file in enumerate(set_of_files):
		image = read_image(image_file)
		is_foreground = np.logical_or(image.get_fdata() < (background_value - tolerance),
									  image.get_fdata() > (background_value + tolerance))
		if i == 0:
			foreground = np.zeros(is_foreground.shape, dtype=np.uint8)

		foreground[is_foreground] = 1
	if return_image:
		return new_img_like(image, foreground)
	else:
		return foreground


def read_image(img_path,initial_cropping_shape=None, resize_shape=None,keep_dimensions=True ,interpolation='linear', crop_parameters=None,is_label=False,labels=None):
	print("Loading File: " + img_path)
	image = nib.load(os.path.abspath(img_path))
	#Fix image shape
	if int(image.shape[-1]) == int(1):
		temp_arr = np.squeeze(image.get_fdata())
		input_shape = np.asarray(temp_arr.shape,dtype=np.float16)
		image = new_img_like(image,temp_arr)
	if resize_shape and not keep_dimensions:
		if initial_cropping_shape:
			image = adjust_img_dimensions_to(image,initial_cropping_shape)
		#if crop_parameters:
		#	image = crop_img_to(image, crop_parameters, copy=True)
		return resize_image(image, new_shape=resize_shape,is_label=is_label,labels=labels)
	elif resize_shape and keep_dimensions:
		return adjust_img_dimensions_to(image,resize_shape)
	return image

def get_cropping_parameters(in_files):
	if len(in_files) > 1:
		foreground = get_combined_foreground_as_binary(in_files)
	else:
		foreground = get_foreground_from_set_of_files(in_files[0], return_image=True)
	return crop_img(foreground, return_slices=True, copy=True)

def resize_image(image, new_shape,is_label=False,labels=None):
	input_shape = np.asarray(image.shape)
	if(len(input_shape)>3):
		temp_arr = np.squeeze(image.get_fdata())
		input_shape = np.asarray(temp_arr.shape,dtype=np.float16)
		image = new_img_like(image,temp_arr)
	data = image.get_fdata()
	if is_label:
		label_data = np.zeros(input_shape)
		for label in labels:
			arr = np.copy(data)
			arr[arr != label] = 0
			label_data = np.asarray(label_data + arr)
		new_data = resize(label_data,new_shape,order=0, mode='constant',anti_aliasing=False)
	else:
		new_data = resize(data,new_shape, mode='constant',anti_aliasing=True)
	new_img = nib.Nifti1Image(new_data, np.eye(4))
	return new_img

# Uncrop data to the size of the target image. Needs slices as an input (which was used to crop the original image)
def uncrop_img(image,target_image, slices):
	target_shape = target_image.shape
	data = np.copy(image.get_fdata())
	data_target = target_image.get_fdata()
	data_target = np.squeeze(data_target)
	data_target[data_target!=0] = 0
	data_target[tuple(slices)] = data
	outputimg = new_img_like(target_image, data_target,affine=target_image.affine,copy_header=True)
	return outputimg

# Crops the shape of the image to resize_shape.
def adjust_img_dimensions_to(image,resize_shape,return_slices=False): # New dimensions MUST be smaller than the original image!
	target_shape = resize_shape
	data = np.copy(image.get_fdata())
	affine = np.copy(image.affine)
	input_shape = data.shape
	adjustments = list()
	x_adjust = input_shape[0] - target_shape[0]
	y_adjust = input_shape[1] - target_shape[1]
	z_adjust = input_shape[2] - target_shape[2]
	slices = tuple()
	slice1 = slice(0 ,input_shape[0],None)
	slice2 = slice(0 ,input_shape[1],None)
	slice3 = slice(0 ,input_shape[2],None)
	if x_adjust > 0:
		if (x_adjust % 2) == 0:
			#Even
			div_adjust = int(x_adjust/2)
			adjustments.append(div_adjust)
			slice1 = (slice(0 + div_adjust, input_shape[0] - div_adjust, None))
		else:
			#Odd
			div_adjust = int(x_adjust/2)
			adjustments.append(div_adjust)
			slice1 = (slice(0 + div_adjust, input_shape[0] - div_adjust - 1, None))
	if y_adjust > 0:
		if (y_adjust % 2) == 0:
			#Even
			div_adjust = int(y_adjust/2)
			adjustments.append(div_adjust)
			slice2 = (slice(0 + div_adjust, input_shape[1] - div_adjust, None))
		else:
			#Odd
			div_adjust = int(y_adjust/2)
			adjustments.append(div_adjust)
			slice2 = (slice(0 + div_adjust, input_shape[1] - div_adjust - 1, None))
	if z_adjust > 0:
		if (z_adjust % 2) == 0:
			#Even
			div_adjust = int(z_adjust/2)
			adjustments.append(div_adjust)
			slice3 = (slice(0 + div_adjust, input_shape[2] - div_adjust, None))
		else:
			#Odd
			div_adjust = int(z_adjust/2)
			adjustments.append(div_adjust)
			slice3 = (slice(0 + div_adjust, input_shape[2] - div_adjust - 1, None))
	slices = [slice1,slice2,slice3]
	cropped_data = data[tuple(slices)]
	cropped_data.copy()
	linear_part = affine[:3, :3]
	old_origin = affine[:3, 3]
	new_origin_voxel = np.array([s.start for s in slices])
	new_origin = old_origin + linear_part.dot(new_origin_voxel)
	new_affine = np.eye(4)
	new_affine[:3, :3] = linear_part
	new_affine[:3, 3] = new_origin
	outputimg = new_img_like(image, cropped_data, new_affine)

	if return_slices:
		return outputimg, slices
	else:
		return outputimg

def open_data_file(filename, readwrite="r"):
	#return h5py.File(filename, readwrite)
	return tables.open_file(filename, readwrite)

def crop_img(img, rtol=1e-8, copy=True, return_slices=False):
	img = check_niimg(img)
	data = img.get_fdata()
	infinity_norm = max(-data.min(), data.max())
	passes_threshold = np.logical_or(data < -rtol * infinity_norm,
									 data > rtol * infinity_norm)

	if data.ndim == 4:
		passes_threshold = np.any(passes_threshold, axis=-1)
	coords = np.array(np.where(passes_threshold))
	start = coords.min(axis=1)
	end = coords.max(axis=1) + 1

	# pad with one voxel to avoid resampling problems
	start = np.maximum(start - 1, 0)
	end = np.minimum(end + 1, data.shape[:3])

	slices = [slice(s, e) for s, e in zip(start, end)]

	if return_slices:
		return slices

	return crop_img_to(img, slices, copy=copy)