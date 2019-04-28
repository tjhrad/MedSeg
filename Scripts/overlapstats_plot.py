#!/bin/python

import os
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import numpy as np 
from scipy import stats

csv_dir = "C:\Users\Trevor\Desktop\data_files\\"
outfile = "overlapstats_connectome.pdf"
title = "Predicted ventricular volumes: "
x_label = "Batch"
y_label = "???"

labels = ["Total","Union (jaccard)","Mean (dice)","False negative","False positive"]
index = range(1,)

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
			 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
			 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
			 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
			 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):    
			r, g, b = tableau20[i]    
			tableau20[i] = (r / 255., g / 255., b / 255.)

csv_data = pd.read_csv(csv_dir + "test.csv")

with PdfPages(outfile) as pdf:
	x = 1

	for label in labels:
		plt.figure(figsize=(15,11))  
		ax = plt.subplot(111)  
		ax.spines["top"].set_visible(False)    
		ax.spines["bottom"].set_visible(False)    
		ax.spines["right"].set_visible(False)    
		ax.spines["left"].set_visible(False) 
	   
		ax.get_xaxis().tick_bottom()    
		ax.get_yaxis().tick_left()    

		plt.yticks(fontsize=14)     
		plt.xticks(fontsize=14)   
		plt.xticks(np.arange(0, 20, 2))

		plt.tick_params(axis="both", which="both", bottom="off", top="off",    
							labelbottom="on", left="off", right="off", labelleft="on")    
			   
		plt.plot(csv_data["Batch"],csv_data[label],lw=2.5, color=tableau20[x])
		plt.title("Average " + label + " vs. Batch",fontsize=14) 
		plt.xlabel(x_label,fontsize=14)
		plt.ylabel(label,fontsize=14)
		#plt.show()
		pdf.savefig()
		plt.close()
		x = x + 1

