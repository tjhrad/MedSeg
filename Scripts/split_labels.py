#!/bin/python

import sys
sys.path.insert(0,"..")
import os
import time
import nibabel as nib
import numpy as np


def main():
	test_dir = "D:\\Research\\normal_split\\test\\"
	labels=(1,2,3)

	for subject in sorted(os.listdir(test_dir)):
		print(subject)
		image_path = os.path.join(test_dir,subject + '\\ventricles_prediction_v2.nii.gz')
		image = nib.load(image_path)
		input_shape = image.shape
		data = image.get_fdata()
		for label in labels:
			out_path = os.path.join(test_dir,subject + '\\3d_unet_nearest_prediction_' + str(label) +'.nii.gz')
			arr = np.copy(np.rint(data))
			arr[arr > label] = 0
			arr[arr < label] = 0
			arr[arr == label] = 1
			img = nib.Nifti1Image(arr,image.affine)
			nib.save(img,out_path)


if __name__ == "__main__":
	main()