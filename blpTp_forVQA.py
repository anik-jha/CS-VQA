# To creaate the train and val dataset for blocked based approach
# This code is also used to compute the execution time

import os
from blocked_pTp import blocked_pTp
from scipy import misc
import time
train_in='/media/twotb/VQA_images/VQA/val2014' #VQA dataset - val2014 for val data 
#train_in='/media/twotb/imagenet/ILSVRC2012_val' #imagenet dataset - _val for val data

#train_out='/media/ssd/lichi/VQA1_0_01/Datasets/Blp_inet_val' #output imagenet directory
train_out='/media/ssd/lichi/VQA1_0_01/Datasets/test'   #output directory
print train_out
times_1000=[]
for root, dirs, files in os.walk(train_in):
	i=0
	for file in files:
		if file.endswith(".jpg"): 	#.jpg for vqa dataset
			print 'image_no = ' + str(i) + ' file= '+ file
			
			#print root
			sub_dir=file.split('_')[0]

			#print sub_dir
			
			file_path= root + '/'+file
			tstart=time.time()
			out_img=blocked_pTp(file_path) # take original image(file_path) as input
			tend=time.time()
			times_1000.append(tend - tstart)
			if i==1000:
				print "average time cost is ", round((sum(times_1000)/1000.0),4), "s"
				times_1000=[]
				raw_input('please enter')
			#print out_img
			out_filepath=train_out+'/'+file
			#print out_filepath
			misc.imsave(out_filepath,out_img)
			#raw_input()
 			i=i+1
print 'finished create train dataset'
