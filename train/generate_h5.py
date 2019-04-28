#!/bin/python

import sys
sys.path.insert(0,"..\\")

import os
import medseg.dataManager as dm
from constants.const import CONST_DICT

def main():
	if CONST_DICT["overwrite_training_data_file"] or not os.path.exists(CONST_DICT["training_data_path"]):

		data_paths = dm.get_data_paths(CONST_DICT["train_dir"],CONST_DICT["training_names"])
		save_location = dm.write_images_to_file(data_paths,CONST_DICT["training_data_path"],
											CONST_DICT["data_shape"],initial_cropping_shape=CONST_DICT["initial_cropping_shape"],keep_dimensions=CONST_DICT["keep_dimensions"], 
											model_2dimensional=CONST_DICT["model_2dimensional"],labels=CONST_DICT["labels"]) 

		print("Data saved to: " + save_location)

if __name__ == "__main__":
	main()