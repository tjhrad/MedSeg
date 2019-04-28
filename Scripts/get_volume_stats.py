#!/bin/python

import os
import numpy as np
import nibabel as nib
import sys, getopt


def get_volume_stats(image_path,label): 
	"""
	Returns the total volume of a binary label in mm3.

	Inputs:

	image_path - A .nii image.

	label - The value of the label (usually and integer)
	"""
	image = nib.load(image_path)
	data = np.asarray(image.get_data())
	header = image.header
	dim1 = header.get_zooms()[0]
	dim2 = header.get_zooms()[1]
	dim3 = header.get_zooms()[2]
	voxel_size = dim1*dim2*dim3
	num_voxels = np.count_nonzero(data == label)
	roi_total_volume = voxel_size * num_voxels
	truncated_vol = round(roi_total_volume,2)
	return truncated_vol


def main(argv):
	inputfile = ''
	label = 1
	try:
		opts, args = getopt.getopt(argv,"hi:t:",["help","ifile=","label="])
	except getopt.GetoptError:
		print '\nPlease give and input image and a label value.\nUsage: get_volume_stats.py -i <inputfile> -l <label>\n\n'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print "\n\nUsage: get_volume_stats.py -i <inputfile> -l <label>\n\n"
			print "If not specified, ROI value set to "
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-l", "--label"):
			label = float(arg)
	if inputfile != '':
		print(get_volume_stats(inputfile,label))
	else:
		print "Please specify an image"


if __name__ == "__main__":
   main(sys.argv[1:])





