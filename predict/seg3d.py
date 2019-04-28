#!/bin/python

import os
import sys, getopt
sys.path.insert(0,"..")

import time
import medseg.dataManager as dm
import medseg.modelManager as mm
import medseg.predict as p

def main(argv):
	inputfile = ''
	model_path = ''
	outfile = ''
	dimensions = list()
	crop_shape = list()
	labels = list()
	try:
		opts, args = getopt.getopt(argv,"hi:m:o:d:l:",["help","ifile=","model=","outfile","dimensions=","labels="])
	except getopt.GetoptError:
		print ('\n\nUsage: predict3d.py -i <inputfile.nii> -m <model.h5> -o <outfile.nii.gz> -d \"dimension1,dimension2,dimension3\" -l \"label1,label2...,labeln\"\n\n')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print ('\n\nUsage: predict3d.py -i <inputfile.nii> -m <model.h5> -o <outfile.nii.gz> -d \"dimension1,dimension2,dimension3\" -l \"label1,label2...,labeln\"\n\n')
			print ("NOTE: Dimensions should the same as the 3D cropping parameters used when training the original model.")
			print ("Example: -d \"144,144,144\" \nIf no cropping was used to train your model, don't specify any dimensions.")
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-m", "--model"):
			model_path = arg
		elif opt in ("-o", "--outfile"):
			outfile = arg
		elif opt in ("-d", "--dimensions"):
			dimensions = arg.split(",")
			crop_shape = (int(dimensions[0]),int(dimensions[1]),int(dimensions[2]))
		elif opt in ("-l", "--labels"):
			labels = arg.split(",")
	if inputfile != '' and model_path != '' and outfile != '':
		print("")
		print("***loading model***")
		print("")
		model = mm.load_old_model(model_path)
		p.predict3d(inputfile,model,outfile,crop_shape=crop_shape,labels=labels)
	else:
		print ('\n\nUsage: predict3d.py -i <inputfile.nii> -m <model.h5> -o <outfile.nii.gz> -d \"dimension1,dimension2,dimension3\" -l \"label1,label2...,labeln\"\n\n')


if __name__ == "__main__":
   main(sys.argv[1:])
