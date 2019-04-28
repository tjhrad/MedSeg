#!/bin/python

import numpy as np
import nibabel as nib
from sklearn.metrics import jaccard_similarity_score
import sys, getopt




def jaccard_nii(img1, img2, label):
	"""
	Computes the Jaccard coefficient between two .nii images, a measure of set similarity.
	"""
	image1 = nib.load(img1)
	data1 = np.asarray(image1.get_data() == label)
	image2 = nib.load(img2)
	data2 = np.asarray(image2.get_data() == label)
	im1 = np.asarray(data1).astype(np.bool)
	im2 = np.asarray(data2).astype(np.bool)

	if im1.shape != im2.shape:
		raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

	intersection = np.logical_and(im1, im2)

	union = np.logical_or(im1, im2)

	return intersection.sum() / float(union.sum())


def main(argv):
	inputfile = ''
	inputfile2 = ''
	label = 1
	try:
		opts, args = getopt.getopt(argv,"hi:c:l:",["help","ifile=","ifile2=","label="])
	except getopt.GetoptError:
		print '\nPlease input two images.\nUsage: jaccard_similarity_nii.py -i <inputfile1> -c <inputfile2> -l <int>\n\n'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print "\n\nUsage: jaccard_similarity_nii.py -i <inputfile1> -c <inputfile2> -l <int>\n\n"
			print "\n\nCalculates the similarity of two binary masks. Ouputs a value between 0 and 1 (1 means the images are equivalent).\n\n"
			sys.exit()
		elif opt in ("-i", "--ifile1"):
			inputfile = arg
		elif opt in ("-c", "--ifile2"):
			inputfile2 = arg
		elif opt in ("-l", "--label"):
			label = int(arg)
	if inputfile != '' and inputfile2 != '':
		print(jaccard_nii(inputfile,inputfile2,label))
	else:
		print "Two images are required"


if __name__ == "__main__":
   main(sys.argv[1:])