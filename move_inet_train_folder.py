import os
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
train_out='/media/ssd/lichi/VQA1_0_01/Datasets/Blp_inet_train'   #output directory

files = [f for f in os.listdir(train_out) if os.path.isfile(os.path.join(train_out, f))]

for file in files:
	if file.endswith(".JPEG"):
		sub_dir=file.split('_')[0]
		#print sub_dir
		#print file
		if not os.path.isdir(train_out + '/' + sub_dir):
		        os.makedirs(train_out + '/' + sub_dir)					
		out_filepath= train_out + '/' + sub_dir + '/' +file
		print out_filepath
		os.rename(train_out + '/'+ file,out_filepath)
print 'finished create train dataset'
