# This code is used to generate the imagenet and vqa datasets for train and val
# This code is also used to calculate the execution time
import os
#from blocked_pTp import blocked_pTp
from blpTp_recon import blpTp_recon
from scipy import misc
import time
train_in='/media/twotb/VQA_images/VQA/val2014' #VQA dataset - (val/train)2014 for val/train data respectively
#train_in='/media/twotb/imagenet/ILSVRC2012_img_train' #imagenet dataset - _val/_train for val/train data respectively

#train_out='/media/ssd/lichi/VQA1_0_01/Datasets/recon_inet_train' #output imagenet directory
train_out='/media/ssd/lichi/VQA1_0_1/datasets/recon_vqa_val'   #output vqa directory

try:
	os.mkdir(train_out)
except:
	pass
print train_out
times_1000=[]
for root, dirs, files in os.walk(train_in):
	i=0
	for file in files:
		if file.endswith(".jpg"):  #.JPEG for imagenet and .jpg for vqa
			print file
			print root
			sub_dir=file.split('_')[0]

			print sub_dir
			
			file_path= root + '/'+file
			tstart=time.time()
			out_img=blpTp_recon(file_path)
			tend=time.time()
			#print "time cost is ", round(tend - tstart,4), "s"
			times_1000.append(tend - tstart)
			if i==1000:
				print "average time cost is ", round((sum(times_1000)/1000.0),4), "s"
				#raw_input('please enter')
			#print out_img
			out_filepath=train_out+'/'+file
			#print out_filepath
			misc.imsave(out_filepath,out_img)
			#raw_input()
			i=i+1
print 'finished create val dataset'
