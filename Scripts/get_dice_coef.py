#!/bin/python

import numpy as np
import nibabel as nib
import sys, getopt




def dice(img1, img2, label=1, smooth=1.0):
	"""
	Computes the Dice coefficient, a measure of set similarity.
	"""
	image1 = nib.load(img1)
	data1 = np.asarray(image1.get_data() == label )
	image2 = nib.load(img2)
	data2 = np.asarray(image2.get_data() == label)
	y_true_f = data1.flatten()
	y_pred_f = data2.flatten()
	intersection = np.sum(y_true_f * y_pred_f)
	return ( (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth) )


def main(argv):
	inputfile = ''
	inputfile2 = ''
	label = 1
	try:
		opts, args = getopt.getopt(argv,"hi:c:l:",["help","ifile=","ifile2=","label="])
	except getopt.GetoptError:
		print('\nPlease input two images.\nUsage: get_dice_coef.py -i <inputfile1> -c <inputfile2> -l <int>\n\n') 
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print("\n\nUsage: get_dice_coef.py -i <inputfile1> -c <inputfile2> -l <int>\n\n")
			print("\n\nCalculates the similarity of two binary masks. Ouputs a value between 0 and 1 (1 means the images are equivalent).\n\n")
			sys.exit()
		elif opt in ("-i", "--ifile1"):
			inputfile = arg
		elif opt in ("-c", "--ifile2"):
			inputfile2 = arg
		elif opt in ("-l", "--label"):
			label = int(arg)
	if inputfile != '' and inputfile2 != '':
		print(dice(inputfile,inputfile2,label))
	else:
		print("Two images are required") 


if __name__ == "__main__":
   main(sys.argv[1:])