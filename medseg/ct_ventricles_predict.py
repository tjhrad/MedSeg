#!/bin/python

import os
import time
import numpy as np
import nibabel as nib
import dataManager as dm
import modelManager as mm

from nilearn.image import new_img_like



def predict(image, model, out_name, model_dimensions): 
	print("Making prediction for: " + image)

	data_shape = model_dimensions

	input_image_uncropped = nib.load(os.path.abspath(image))
	input_image_cropped, slices = dm.adjust_img_dimensions_to(input_image_uncropped,data_shape,return_slices=True)
	image_data_cropped = np.asarray(input_image_cropped.get_data()) 
	slices_per_image = image_data_cropped.shape[2]
	prediction_data = np.empty_like(image_data_cropped)

	for x in range(0,slices_per_image):
		slice_data = np.asarray(image_data_cropped[:,:,x])
		prediction = model.predict(np.asarray(slice_data[np.newaxis][np.newaxis]))
		prediction_data[:,:,x] = prediction[0][0]
	prediction_data[prediction_data>0.5] = 1
	prediction_data[prediction_data<=0.5] = 0
	prediction_image_cropped = new_img_like(input_image_cropped, prediction_data.astype(np.uint8))
	prediction_image_uncropped = dm.uncrop_img(prediction_image_cropped,input_image_uncropped, slices)
	nib.save(prediction_image_uncropped, os.path.join(os.path.dirname(image),out_name))
	print("Prediction saved as:" + os.path.join(os.path.dirname(image),out_name))
	return prediction_image_uncropped


def main():
	#model_dir="/work/unmcrads/trevorhuff/data_ventricleseg/models/"
	#test_dir="/work/unmcrads/trevorhuff/normal_split/test/"
	model_dir="C:\\Users\\Trevor\\Desktop\\My Python Code\\MedSeg3D\\models\\"
	test_dir="C:\\Users\\Trevor\\Desktop\\ventricle_segmentations\\0000312-063\\"
	dimensions = (256,256,200)
	
	label_n = 0
	for model_path in sorted(os.listdir(model_dir)):
		if "label3" in model_path:
			label_n = 3
			dimensions = (144,144,144)
		if "label2" in model_path:
			label_n = 2
			dimensions = (256,304,200)
		if "label1" in model_path:
			label_n = 1
			dimensions = (256,304,200)

		times = []
		print("Current model: " + model_path)
		model = mm.load_old_model(model_dir+model_path)
		for subject in os.listdir(test_dir):
			start_time = time.time()
			predict(test_dir + subject + "/CT.nii.gz",model,"unet_axial_prediction"+str(label_n)+".nii.gz",dimensions)
			total_time = (time.time() - start_time)
			print("--- %s seconds ---" % total_time)
			times.append(total_time)
		print(sum(times) / len(times))
		
	for subject in os.listdir(test_dir):
		label_data_list = []
		label_imgs = []
		for file in os.listdir(os.path.join(test_dir,subject)):
			if "prediction" in file:
				value = 1
				if "prediction1" in file:
					value = 1
				elif "prediction2" in file:
					value = 2
				elif  "prediction3" in file:
					value = 3
				img_temp = nib.load(os.path.join(test_dir,subject,file))
				img_data_temp = img_temp.get_data()
				img_data_temp[img_data_temp>0] = value
				label_data_list.append(img_data_temp)
		label_data_final = np.array(label_data_list[0])
		label_data_final = 0
		previous_label_value = 0
		for label in label_data_list:
			label_value = np.max(label)
			label_data_final = label_data_final + label
			label_data_final[label_data_final>label_value] = label_value
		subject_img = nib.load(os.path.join(test_dir,subject + "/CT.nii.gz"))
		final_image = new_img_like(subject_img,label_data_final.astype(np.uint8))
		nib.save(final_image,os.path.join(test_dir,subject + "/unet_axial_prediction_final.nii.gz"))

if __name__ == "__main__":
	main()
