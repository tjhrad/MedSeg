#!/bin/python

import os
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd  
from scipy import stats

csv_path = "C:\Users\Trevor\Desktop\connectome_volumetric_data.csv"
outfile = "linear_volumes_connectome.pdf"
title = "Predicted ventricular volumes: "
x_label = "FreeSurfer"

batches = ["FreeSurfer","Batch1","Batch2","Batch3","Batch4","Batch5",
"Batch6","Batch7","Batch8","Batch9","Batch10","Batch11",
"Batch12","Batch13","Batch14","Batch15","Batch16","Batch17","Batch19"]


tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
			 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
			 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
			 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
			 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):    
			r, g, b = tableau20[i]    
			tableau20[i] = (r / 255., g / 255., b / 255.)

with PdfPages(outfile) as pdf:
	for batch in batches:
		csv_data = pd.read_csv(csv_path)

		slope, intercept, r_value, p_value, std_err = stats.linregress(csv_data["FreeSurfer"],csv_data[batch])
		line = slope*csv_data["FreeSurfer"]+intercept

		plt.figure(figsize=(15,11))
		plt.plot(csv_data["FreeSurfer"],csv_data[batch],'o', csv_data["FreeSurfer"], line)
		plt.title(title + " FreeSurfer vs. " + batch,fontsize=25) 
		plt.text(25000, 5000, "R value: " + str(round(r_value,4)), fontsize=20, color=tableau20[0])
		plt.xlabel(x_label + " mm3",fontsize=15)
		plt.ylabel(batch+ " mm3",fontsize=15)
		#plt.show()
		pdf.savefig()
		plt.close()
