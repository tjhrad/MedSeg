#!/bin/python

import numpy as np
import nibabel as nib
from nibabel.processing import smooth_image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from stl import mesh
import sys, getopt


def nii2stl(input_nii, out_file,smooth=True):
	image = nib.load(input_nii)
	if smooth:
		print("Smoothing image..")
		image_smoothed = smooth_image(image,fwhm=3)
		data = np.asarray(image_smoothed.get_data())
	else:
		data = np.asarray(image.get_data())
	print("Converting to stl..")
	verts, faces, normals, values = measure.marching_cubes_lewiner(data)
	solid = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
	for i, f in enumerate(faces):
			for j in range(3):
				solid.vectors[i][j] = verts[f[j],:]
	solid.save(out_file)

	print("STL saved to: " + out_file)

def main(argv):
	input_nii = ''
	out_file = ''
	smooth = True
	try:
		opts, args = getopt.getopt(argv,"hi:o:s:",["help","in=","out=","smooth="])
	except getopt.GetoptError:
		print '\n\nUsage: nii2stl.py -i <in.nii> -o <out.stl> -s (y or n)\n\n'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print "\n\nUsage: nii2stl.py -i <in.nii> -o <out.stl> -s (y or n)\n\n"
			sys.exit()
		elif opt in ("-i", "--in"):
			input_nii = arg
		elif opt in ("-o", "--out"):
			out_file = arg
		elif opt in ("-s", "--smooth"):
			smooth_arg = arg
			if smooth_arg == "n":
				smooth = False
	if input_nii != '' and out_file != '':
		nii2stl(input_nii,out_file,smooth)
	else:
		print "\n\nUsage: nii2stl.py -i <in.nii> -o <out.stl> -s (y or n)\n\n"


if __name__ == "__main__":
   main(sys.argv[1:])