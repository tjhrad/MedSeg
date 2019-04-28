#!/bin/python

import os

CONST_DICT = dict()


##########################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##########################################################

"""
CONST_DICT["mode"] = "3d_t1"
CONST_DICT["training_names"] = ["T1w_acpc_dc","aparc+aseg"]
"""
CONST_DICT["mode"] = "3d_ct"
CONST_DICT["training_names"] = ["CT","label_1"]
CONST_DICT["labels"] = (1)#(1,2,3) #
#CONST_DICT["labels"] = (17,53)#(1,2,3)#(1)

if isinstance(CONST_DICT["labels"],int):
	CONST_DICT["num_labels"] = 1
else:
	CONST_DICT["num_labels"] = len(CONST_DICT["labels"])

CONST_DICT["predict_file_names"] = ["CT"]
CONST_DICT["num_images"] = 250 # 1000 TODO - think about getting rid of this
CONST_DICT["data_shape"] = (128, 128, 128)
CONST_DICT["initial_cropping_shape"] = (256, 384, 256) #None

#CONST_DICT["train_dir"] = "/work/unmcrads/trevorhuff/normal_split/train/"
#CONST_DICT["out_dir"] = "/work/unmcrads/trevorhuff/data_ventricleseg/"
CONST_DICT["train_dir"] = "D:\\Research\\normal_split\\train\\"
CONST_DICT["out_dir"] = "D:\\Research\\normal_split\\out\\"
#CONST_DICT["train_dir"] = "C:\\Users\\neurorad\\Box\\ML\\imaging_data\\connectome_scans\\train\\"
#CONST_DICT["out_dir"] = "F:\\ML_data\\Out\\"

##########################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##########################################################

CONST_DICT["training_data_path"] = os.path.join(CONST_DICT["out_dir"],
	CONST_DICT["training_names"][1]+"_"+CONST_DICT["mode"]+"_master_data.h5")

CONST_DICT["model_path"] = os.path.join(CONST_DICT["out_dir"],
	CONST_DICT["training_names"][1]+"_"+CONST_DICT["mode"]+"_model.h5")

CONST_DICT["model_checkpoint_path"] = os.path.join(CONST_DICT["out_dir"],
	CONST_DICT["training_names"][1]+"_"+CONST_DICT["mode"]+"_model_checkpoint.h5")

CONST_DICT["csv_log_path"] = os.path.join(CONST_DICT["out_dir"],
	CONST_DICT["training_names"][1]+"_"+CONST_DICT["mode"]+"_log.csv")

CONST_DICT["input_image_modalities"] = ["CT"]
CONST_DICT["num_channels"] = len(CONST_DICT["input_image_modalities"])

CONST_DICT["patch_shape"] = None
if CONST_DICT["patch_shape"] is not None:
	CONST_DICT["input_shape"] = tuple([CONST_DICT["num_channels"]] + list(CONST_DICT["patch_shape"]))
else:
	CONST_DICT["input_shape"] = tuple([CONST_DICT["num_channels"]] + list(CONST_DICT["data_shape"]))
	
CONST_DICT["kernel_size"] = (3, 3, 3)
CONST_DICT["convolution_strides"] = (1, 1, 1)
CONST_DICT["pool_size"] = (2, 2, 2)
CONST_DICT["pooling_strides"] = (2, 2, 2)

CONST_DICT["depth_of_model"] = 4 
CONST_DICT["validation_split"] = 0.8 # If == 1 no validation used
CONST_DICT["validation_patch_overlap"] = 0
CONST_DICT["training_patch_start_offset"] = None #(16, 16, 16)

CONST_DICT["base_filters"] = 16
CONST_DICT["batch_size"] = 1
CONST_DICT["validation_batch_size"] = 1
CONST_DICT["activation_conv"] = 'relu'

CONST_DICT["n_epochs"] = 500
CONST_DICT["initial_learning_rate"] = 0.0001
CONST_DICT["early_stopping_patience"] = 20
CONST_DICT["learning_rate_patience"] = 10
CONST_DICT["learning_rate_epochs"] = None
CONST_DICT["learning_rate_drop"] = 0.5

CONST_DICT["batch_normalization"] = False
CONST_DICT["deconvolution"] = True
CONST_DICT["permute"] = False
CONST_DICT["flip"] = False
CONST_DICT["distort"] = None
CONST_DICT["augment"] = CONST_DICT["flip"] or CONST_DICT["distort"]
CONST_DICT["skip_blank"] = True

CONST_DICT["crop_to_foreground"] = False
CONST_DICT["keep_dimensions"] = False
CONST_DICT["include_label_coeffs"] = False
CONST_DICT["overwrite_training_data_file"] = False
CONST_DICT["is_training_new_model"] = True
CONST_DICT["model_2dimensional"] = False 

if CONST_DICT["model_2dimensional"]:
	CONST_DICT["d_shape"] = (CONST_DICT["data_shape[0]"], CONST_DICT["data_shape[1]"])
	CONST_DICT["input_shape"] = tuple([CONST_DICT["num_channels"]] + list(CONST_DICT["d_shape"]))
	CONST_DICT["patch_shape"] = None
	CONST_DICT["kernel_size"] = (3, 3)
	CONST_DICT["pool_size"] = (2, 2)
	CONST_DICT["pooling_strides"] = (2, 2)
	CONST_DICT["convolution_strides"] = (1, 1)
	CONST_DICT["batch_size"] = 18 #1 * data_shape[0] # If you are using 2D CNN, batch size must be a multiple of your data shape for now
	CONST_DICT["validation_batch_size"] = 9
	CONST_DICT["keep_dimensions"] = True
	CONST_DICT["batch_normalization"] = False
	CONST_DICT["deconvolution"] = False
	CONST_DICT["permute"] = False
	CONST_DICT["flip"] = False
	CONST_DICT["distort"] = None
	CONST_DICT["augment"] = CONST_DICT["flip"] or CONST_DICT["distort"]
	CONST_DICT["skip_blank"] = False
