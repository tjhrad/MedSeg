#!/bin/python

import sys
sys.path.insert(0,"..")
import os
import time

from medseg.predict import predict3d
from medseg.modelManager import load_old_model


def main():
	test_dir = "D:\\Research\\normal_split\\test\\"
	model_path = "D:\\Research\\ML_models\\ct\\ventricles_3d_ct_model.h5"
	crop_dimensions = (256,384,256)
	times = []

	print("Current model: " + model_path)
	model = load_old_model(model_path)

	for subject in sorted(os.listdir(test_dir)):
		image_path = os.path.join(test_dir,subject + '\\CT.nii.gz')
		out_path = os.path.join(test_dir,subject + '\\ventricles_prediction_v2.nii.gz')
		start_time = time.time()
		predict3d(image_path,model,out_path,crop_shape=crop_dimensions,labels=(1,2,3))
		total_time = (time.time() - start_time)
		print("--- Segmentation time: %s seconds ---" % total_time)
		times.append(total_time)
	print("")
	print("~~ Average time ~~")
	print(sum(times) / len(times))


if __name__ == "__main__":
	main()