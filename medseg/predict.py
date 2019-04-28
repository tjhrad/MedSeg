#!/bin/python

import os
import nibabel as nib
import numpy as np
import tables
import medseg.dataManager as dm
from nilearn.image import resample_to_img, reorder_img
from skimage.transform import resize

from keras.models import load_model
from nilearn.image import new_img_like


def predict3d(image_path, model, out_name, crop_shape=None,model_dimensions=(128,128,128),labels=None): 
	print("Making prediction for: " + image_path)
	label_map = False
	if len(labels) > 0:
		label_map = True

	data_shape = model_dimensions
	input_image = nib.load(os.path.abspath(image_path))

	if len(crop_shape) == 3:
		input_image_cropped, slices = dm.adjust_img_dimensions_to(input_image,crop_shape,return_slices=True)
		resized_input_data = resize(input_image_cropped.get_fdata(), model_dimensions, mode='constant', anti_aliasing=False)
	else:
		resized_input_data = resize(input_image.get_fdata(), model_dimensions, mode='constant', anti_aliasing=True)

	img_data = resized_input_data 
	img_data = np.squeeze(img_data)
	prediction_data = model.predict(np.asarray(img_data)[np.newaxis][np.newaxis])
	prediction = np.squeeze(prediction_data)

	if len(prediction.shape) > 3:
		if label_map:
			if labels:
				index = 0
				label_map_data = np.zeros(np.squeeze(input_image).shape, np.int8)
				for label in labels:
					temporary_data = np.asarray(prediction[index])
					if len(crop_shape) == 3:
						temporary_prediction_data = resize(temporary_data, np.squeeze(input_image_cropped.get_fdata()).shape, mode='constant', anti_aliasing=True)
					else:
						temporary_prediction_data = resize(temporary_data, np.squeeze(input_image.get_fdata()).shape, mode='constant', anti_aliasing=True)
					d = temporary_prediction_data
					d[d>0.5] = int(label)
					d[d<=0.5] = 0
					label_map_data = np.asarray(label_map_data + d)
					label_map_data[label_map_data > int(label)] = int(label)
					index = index + 1
		else:
			label_map_data = np.zeros(np.squeeze(input_image).shape, np.int8)
			for index in range(prediction.shape[0]):
				temporary_data = np.asarray(prediction[index])
				if len(crop_shape) == 3:
					temporary_prediction_data = resize(temporary_data, np.squeeze(input_image_cropped.get_fdata()).shape, mode='constant', anti_aliasing=True)
				else:
					temporary_prediction_data = resize(temporary_data, np.squeeze(input_image.get_fdata()).shape, mode='constant', anti_aliasing=True)
				d = temporary_prediction_data
				d[d>0.5] = index + 1
				d[d<=0.5] = 0
				label_map_data = np.asarray(label_map_data + d)
				label_map_data[label_map_data > (index + 1)] = index + 1
				index = index + 1
	elif len(prediction.shape) == 3:
		label_map_data = np.zeros(np.squeeze(input_image).shape, np.int8)
		temporary_data = np.asarray(prediction)
		if len(crop_shape) == 3:
			temporary_prediction_data = resize(temporary_data, np.squeeze(input_image_cropped.get_fdata()).shape, mode='constant', anti_aliasing=True)
		else:
			temporary_prediction_data = resize(temporary_data, np.squeeze(input_image.get_fdata()).shape, mode='constant', anti_aliasing=True)
		d = temporary_prediction_data
		d[d>0.5] = 1
		d[d<=0.5] = 0
		label_map_data = np.asarray(d)
	else:
		raise RuntimeError("Invalid prediction shape: {0}".format(prediction.shape))

	if len(crop_shape) == 3:
		prediction_img_cropped = nib.Nifti1Image(label_map_data,input_image_cropped.affine, input_image_cropped.header)
		prediction_img = dm.uncrop_img(prediction_img_cropped,input_image, slices)
	else:
		prediction_img = nib.Nifti1Image(label_map_data,input_image.affine, input_image.header)

	nib.save(prediction_img, out_name)
	print("Prediction saved as: " + out_name)