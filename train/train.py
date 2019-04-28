#!/bin/python

import sys
sys.path.insert(0,"..\\")

import os
import medseg.generators as gs
import medseg.dataManager as dm
import medseg.modelManager as mm

from constants.const import CONST_DICT

from medseg.model import unet_model

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

def main():

	if os.path.isfile(CONST_DICT["training_data_path"]):
		print("")
		print("")

		print("Opening: " + CONST_DICT["training_data_path"])
		training_data_file = dm.open_data_file(CONST_DICT["training_data_path"])

		print("")
		print("")
		if CONST_DICT("model_2dimensional"):
			model = mm.get_new_model(input_shape=CONST_DICT["input_shape"],pool_size=CONST_DICT["pool_size"],kernel_size=CONST_DICT["kernel_size"],
									 n_labels=CONST_DICT["num_labels"],deconvolution=CONST_DICT["deconvolution"],depth=CONST_DICT["depth_of_model"],
									 n_base_filters=CONST_DICT["base_filters"],include_label_wise_dice_coefficients=CONST_DICT["include_label_coeffs"],batch_normalization=CONST_DICT["batch_normalization"],
									 activation_conv=CONST_DICT["activation_conv"],pooling_strides=CONST_DICT["pooling_strides"],
									 convolution_strides=CONST_DICT["convolution_strides"], model_2dimensional=CONST_DICT["model_2dimensional"])
		else:
			model = unet_model(input_shape=CONST_DICT["input_shape"],n_labels=CONST_DICT["num_labels"],
									initial_learning_rate=CONST_DICT["initial_learning_rate"],
									n_base_filters=CONST_DICT["base_filters"])

		print("")
		print("")
		print(model.summary())
		print("")
		print("")
		
		train_generator, validation_generator, n_train_steps, n_validation_steps = gs.get_generator_and_steps(training_data_file,
								batch_size=CONST_DICT["batch_size"],data_split=CONST_DICT["validation_split"],
								n_labels=CONST_DICT["num_labels"],labels=CONST_DICT["labels"],patch_shape=CONST_DICT["patch_shape"],
								validation_batch_size=CONST_DICT["validation_batch_size"],validation_patch_overlap=CONST_DICT["validation_patch_overlap"],
								training_patch_start_offset=CONST_DICT["training_patch_start_offset"],permute=CONST_DICT["permute"],
								augment=CONST_DICT["augment"],skip_blank=CONST_DICT["skip_blank"],augment_flip=CONST_DICT["flip"],
								augment_distortion_factor=CONST_DICT["distort"],model_2dimensional=CONST_DICT["model_2dimensional"])

		if CONST_DICT["model_2dimensional"]:
			#n_train_steps = (40 * data_shape[0])/ batch_size
			#n_validation_steps = (10 * data_shape[0])/ batch_size
			n_train_steps = ((CONST_DICT["num_images"]*0.8) * CONST_DICT["data_shape"][2])/ CONST_DICT["batch_size"]
			n_validation_steps = ((CONST_DICT["num_images"]*0.2) * CONST_DICT["data_shape"][2])/ CONST_DICT["batch_size"]
			print("Adjusted training steps: " + str(n_train_steps))

		print("\n\n**********************************")
		print("Training new model. Please wait...")
		print("**********************************\n\n")

		
		if CONST_DICT["validation_split"] == 1:
			model.fit_generator(generator=train_generator,steps_per_epoch=n_train_steps,epochs=CONST_DICT["n_epochs"])
		else:
			callbacks = list()
			callbacks.append(CSVLogger(CONST_DICT["csv_log_path"]))
			callbacks.append(ReduceLROnPlateau(monitor='val_loss',factor=CONST_DICT["learning_rate_drop"], patience=CONST_DICT["learning_rate_patience"],verbose=1))
			callbacks.append(ModelCheckpoint(CONST_DICT["model_checkpoint_path"], monitor='val_loss', save_best_only=True))
			if CONST_DICT["early_stopping_patience"]:
				callbacks.append(EarlyStopping(monitor='val_loss',verbose=0, patience=CONST_DICT["early_stopping_patience"]))
			model.fit_generator(generator=train_generator,steps_per_epoch=n_train_steps,
					epochs=CONST_DICT["n_epochs"],validation_data=validation_generator,
					validation_steps=n_validation_steps,callbacks=callbacks)

		print("Saving model...")
		model.save(CONST_DICT["model_path"])
		print("Model saved in the following location: " + CONST_DICT["model_path"])

		training_data_file.close()
	else:
		print("")
		print("")
		print("Could not find: " + CONST_DICT["training_data_path"])
		print("")
		print("")

if __name__ == "__main__":
	main()