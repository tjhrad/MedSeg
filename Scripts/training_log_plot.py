#!/bin/python

import os
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd  

csv_dir = "C:\Users\Trevor\Desktop\log_files\\"
outfile = "training_data.pdf"
title = "Training Results: "
x_label = "Epoch"
y_label = ""

with PdfPages(outfile) as pdf:
	x = 1
	for csv_file_path in os.listdir(csv_dir):
		csv_data = pd.read_csv(os.path.join(csv_dir,csv_file_path))

		# These are the "Tableau 20" colors as RGB.    
		tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
					 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
					 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
					 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
					 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

		# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
		for i in range(len(tableau20)):    
			r, g, b = tableau20[i]    
			tableau20[i] = (r / 255., g / 255., b / 255.)    

		plt.figure(figsize=(15,11))

		# Remove the plot frame lines. They are unnecessary chartjunk.    
		ax = plt.subplot(111)    
		ax.spines["top"].set_visible(False)    
		ax.spines["bottom"].set_visible(False)    
		ax.spines["right"].set_visible(False)    
		ax.spines["left"].set_visible(False) 
   
		ax.get_xaxis().tick_bottom()    
		ax.get_yaxis().tick_left()    
 
		plt.xlim(0, 25)   
		plt.yticks(fontsize=14)     
		plt.xticks(fontsize=14)   

		for y in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:    
			plt.plot(range(0, len(csv_data)), [y] * len(range(0, len(csv_data))), "--", lw=0.8, color="black", alpha=0.3)    

		plt.tick_params(axis="both", which="both", bottom="off", top="off",    
						labelbottom="on", left="off", right="off", labelleft="on")    

		columns = ['dice_coefficient','loss','val_dice_coefficient','val_loss']
		  
		for rank, column in enumerate(columns):    
			plt.plot(csv_data.epoch.values,    
					csv_data[column.replace("\n", " ")].values,    
					lw=2.5, color=tableau20[rank])    
			y_pos = 0.5
			if column == "dice_coefficient":
				y_pos += 0.17
			elif column == "val_dice_coefficient":
				y_pos += 0.12
			elif column == "loss":
				y_pos -= 0.13
			elif column == "val_loss":
				y_pos -= 0.18
			plt.text(20, y_pos, column, fontsize=14, color=tableau20[rank])  
		plt.title(title + str(x*50) + " images. File: " + csv_file_path,fontsize=14) 
		plt.xlabel(x_label,fontsize=10)
		plt.ylabel(y_label)
		pdf.savefig()
		plt.close()
		x = x + 1