import numpy as np
from skimage import io; 
io.use_plugin('matplotlib')
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import json
import h5py
import argparse
#json_file=json.load(open('vqa_data_prepro.json','r'))
GPU_ID = 0 # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

def generate_hdf5(params):
	json_file=json.load(open(params['json_file'],'r'))
	# net = caffe.Net('/home/mayank/caffe/models/bvlc_googlenet/deploy.prototxt',
	#                 '/media/ssd/lichi/imagenet/googlenet_pi_mean_dlr_ss160000_f2200000_resume_iter_5500000.caffemodel',
	#                 caffe.TEST)
	net = caffe.Net(params['deploy_proto'],
	                params['model'],
	                caffe.TEST)

	#dim_img=1024 #pool
	#dim_img=1000 #classifier


	batch_size= 1
	# load input and configure preprocessing
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	#transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
	transformer.set_transpose('data', (2,0,1))
	transformer.set_channel_swap('data', (2,1,0))
	transformer.set_raw_scale('data', 255.0)

	#note we can change the batch size on-the-fly
	#since we classify only one image, we change batch size from 10 to 1
	net.blobs['data'].reshape(batch_size,3,224,224)
	train_data=json_file['unique_img_train']
	test_data=json_file['unique_img_test']

	#load the image in the data layer

	# for im_path in train_data:
	# 	im = caffe.io.load_image(im_path)
	# 	net.blobs['data'].data[...] = transformer.preprocess('data', im)

	# 	#compute
	# 	out = net.forward(end='loss3/classifier')
	# 	print out
	# 	feat_train[im_path]=out
	# input_train=(82460,1,1000)
	# input_test=(40504,1,1000)
	#incpt5b
	#input_train=(82460,1024,7,7) #train images SAN
	#input_test=(40504,1024,7,7) 
	# input_train=(82460,1,1024) 
	# input_test=(40504,1,1024)
	input_train=(82783,1024) #LSTM
	input_test=(40504,1024)
	#feat_train=np.zeros((len(train_data),dim_img),dtype=np.float32) #len(train_data)=82460
	#feat_test=np.zeros((len(test_data),dim_img),dtype=np.float32)  #len(test_data)=40504
	feat_train=np.zeros(input_train,dtype=np.float32) #len(train_data)=82460
	feat_test=np.zeros(input_test,dtype=np.float32)  #len(test_data)=40504

	
	# end_layer='inception_5b/output' #SAN
	end_layer='pool5/7x7_s1' #LSTM
	#end_layer='loss3/classifier'	
	img_feat= params['outfeat']
	print 'processing image feature',img_feat
	i=0
	for im_path in test_data:
		im = caffe.io.load_image(im_path)
		net.blobs['data'].data[...] = transformer.preprocess('data', im)

		#compute
		#out = net.forward(end='loss3/classifier')
		#vec_out=out['loss3/classifier']
		
		#for pooling layer
		out = net.forward(end=end_layer)
		vec_out=out[end_layer]
		if vec_out.shape == (1,1024,1,1): # for pool5/7x7_s1
			vec_out=np.reshape(vec_out,(1,1024))
		print vec_out
		print vec_out.shape

		feat_test[i][:]=vec_out
		i=i+1

	i=0
	for im_path in train_data:
		im = caffe.io.load_image(im_path)
		net.blobs['data'].data[...] = transformer.preprocess('data', im)

		#compute
		#out = net.forward(end='loss3/classifier')
		#vec_out=out['loss3/classifier']
		
		#for pooling layer
		out = net.forward(end=end_layer)
		vec_out=out[end_layer]
		#vec_out=np.squeeze(vec_out)
		
		if vec_out.shape == (1,1024,1,1): # for pool5/7x7_s1
			vec_out=np.reshape(vec_out,(1,1024))
		feat_train[i][:]=vec_out
		print vec_out
		print vec_out.shape
		i=i+1

	f = h5py.File(img_feat, "w")

	f.create_dataset("imgs_train", dtype='float32', data=feat_train)
	f.create_dataset("imgs_test", dtype='float32', data=feat_test)
	#f=h5py.File('feat_imgs_487.h5', "a")  # for read and write existing hdf5
	#data= f['imgs_test']
	#data[...]=feat_test
	f.close()




if __name__ == '__main__':
    #with tf.device('/gpu:'+str(0)):
    #    train()
    #with tf.device('/gpu:'+str(1)):
    #    test()
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', default='recon_data_prepro.json', help='json file')
    parser.add_argument('--deploy_proto', default='/home/anik/caffe/models/bvlc_googlenet/deploy.prototxt', help='caffe_prototxt_file')
    parser.add_argument('--model', default='/media/ssd/lichi/VQA2/vqa2_25/caffe_models/resume_recon_gg_a0.0001_s05_iter_400000.caffemodel', help='finetuned caffe model')
    parser.add_argument('--outfeat', default='feat_imgs_recon_google485_fc_nobatch.h5', help='output h5 file')
    args = parser.parse_args()
    params = vars(args)
    generate_hdf5(params)
