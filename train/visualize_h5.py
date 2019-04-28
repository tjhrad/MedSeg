#!/bin/python

import sys
sys.path.insert(0,"..\\")

import os
import numpy as np
import medseg.generators as gs
import medseg.dataManager as dm
from constants.const import CONST_DICT

import nibabel as nib

from keras import backend as K

def main():

	if os.path.isfile(CONST_DICT["training_data_path"]):
		print("")
		print("")

		print("Opening: " + CONST_DICT["training_data_path"])
		training_data_file = dm.open_data_file(CONST_DICT["training_data_path"])

		print("")
		print("")

		
		train_generator, validation_generator, n_train_steps, n_validation_steps = gs.get_generator_and_steps(training_data_file,
								batch_size=CONST_DICT["batch_size"],data_split=CONST_DICT["validation_split"],
								n_labels=CONST_DICT["num_labels"],labels=CONST_DICT["labels"],patch_shape=CONST_DICT["patch_shape"],
								validation_batch_size=CONST_DICT["validation_batch_size"],validation_patch_overlap=CONST_DICT["validation_patch_overlap"],
								training_patch_start_offset=CONST_DICT["training_patch_start_offset"],permute=CONST_DICT["permute"],
								augment=CONST_DICT["augment"],skip_blank=CONST_DICT["skip_blank"],augment_flip=CONST_DICT["flip"],
								augment_distortion_factor=CONST_DICT["distort"],model_2dimensional=CONST_DICT["model_2dimensional"])

		t = next(train_generator)
		print(len(t))
		print(np.ndim(t[0]))
		print("")
		print(t[0].shape)
		print(t[1].shape)

		img = nib.Nifti1Image(np.squeeze(t[0]),np.eye(4))
		mask = nib.Nifti1Image(np.squeeze(t[1]),np.eye(4))


		nib.save(img,CONST_DICT["out_dir"] + "image.nii.gz")
		nib.save(mask,CONST_DICT["out_dir"] + "mask.nii.gz")
		training_data_file.close()

if __name__ == "__main__":
	main()