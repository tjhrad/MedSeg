#!/bin/python
import os
import time
import numpy as np
import nibabel as nib
import dataManager as dm
import modelManager as mm

from nilearn.image import new_img_like

def predict(image, model,out_name, model_dimensions): 
	print("Making prediction for: " + image)
	input_image_uncropped = nib.load(os.path.abspath(image))
	if len(model_dimensions) == 3:
		data_shape = model_dimensions
		slices_per_image = data_shape[0]
		input_image_cropped, slices = dm.adjust_img_dimensions_to(input_image_uncropped,data_shape,return_slices=True)
		image_data = np.asarray(input_image_cropped.get_data()) 
		prediction_data = np.empty_like(image_data)
		for x in range(0,slices_per_image):
			slice_data = np.asarray(image_data[x])
			prediction = model.predict(np.asarray(slice_data[np.newaxis][np.newaxis]))
			prediction_data[x] = prediction[0][0] + prediction[0][1] + prediction[0][2]#prediction[0] + prediction[1] + prediction[2]
		prediction_image_cropped = new_img_like(input_image_cropped, prediction_data)
		prediction_image_uncropped = dm.uncrop_img(prediction_image_cropped,input_image_uncropped, slices)
		nib.save(prediction_image_uncropped, os.path.join(os.path.dirname(image),out_name))
		print("Prediction saved as:" + os.path.join(os.path.dirname(image),out_name))
	else: 
		image_data = np.asarray(input_image_uncropped.get_data())
		data_shape = image_data.shape
		slices_per_image = data_shape[0]
		prediction_data = np.empty_like(image_data)
		for x in range(0,slices_per_image):
			slice_data = np.asarray(image_data[x])
			prediction = model.predict(np.asarray(slice_data[np.newaxis][np.newaxis]))
			prediction_data[x] = prediction[0:-1]
		prediction_image = new_img_like(input_image_cropped, prediction_data)
		nib.save(prediction_image, os.path.join(os.path.dirname(image),out_name))
		print("Prediction saved as:" + os.path.join(os.path.dirname(image),out_name))

def main():
	model_dir="/work/unmcrads/trevorhuff/data_ventricleseg/"
	model_path="144_vents_2d_modelfinished_1.h5"
	test_dir="/work/unmcrads/trevorhuff/normal/test/"
	program_file="/home/unmcrads/trevorhuff/python_code/MedSeg3D/MedSeg/predict_2d.py"
	dimensions = (144,144,144)
	
	i = 0
	times = []
	print(model_path)
	# Loading the model before running all the predictions. 
	# If you are only making one prediction, you can comment this out and load the model in the function.
	model = mm.load_old_model(model_dir+model_path) 
	for subject in os.listdir(test_dir):
		start_time = time.time()
		predict(test_dir + subject + "/CT.nii.gz",model,"prediction_sag_144"+str(i)+".nii.gz",dimensions)
		total_time = (time.time() - start_time)
		print("--- %s seconds ---" % total_time)
		times.append(total_time)
	print(sum(times) / len(times))
	i = i + 1


if __name__ == "__main__":
	main()
