import caffe
import numpy as np
from scipy import misc
import scipy.io

import matplotlib.pyplot as plt
GPU_ID = 1 # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)
net = caffe.Net('/media/ssd/lichi/VQA1_0_1/codes/deploy_prototxt_files/reconnet_0_10.prototxt',
                '/media/ssd/lichi/VQA1_0_1/codes/caffemodels/reconnet_0_10.caffemodel',
                caffe.TEST)
batch_size= 1
# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
# transformer.set_transpose('data', (2,0,1))
# transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

#note we can change the batch size on-the-fly
#since we classify only one image, we change batch size from 10 to 1
net.blobs['data'].reshape(batch_size,1,109,1) # 10 for .01 and 272 for .25 measurement ratio 

mat=scipy.io.loadmat('phi_0_10_1089.mat')
phi=mat['phi']
phiT=np.transpose(phi)
pTp=np.dot(phiT,phi) #1089 by 1089
end_layer='conv6'
def block_recon_apply(im_pad,pTp,stride):
	for x in range(8):
		for y in range(8):
			#sub_img=np.zeros([33,33])
			sub_img=im_pad[x*stride:33*(x+1),y*stride:33*(y+1)]
			sub_img=np.reshape(sub_img,[1089,1])
			sub_img = np.array(sub_img, dtype=float)
			#sub_img_cs=np.dot(pTp,sub_img)
			sub_img_cs=np.dot(phi,sub_img)
			print sub_img_cs.shape
			net.blobs['data'].data[...] = transformer.preprocess('data', sub_img_cs)
			#apply reconet
			out = net.forward(end=end_layer)
			vec_out=out[end_layer]
			
			#plt.imshow(sub_img_cs)
			#plt.show()
			#raw_input("Press the <ENTER> key to continue...")
			im_pad[x*stride:33*(x+1),y*stride:33*(y+1)]=vec_out



#pTp = np.identity(1089)
def blpTp_recon(img_file):
	#img = misc.face()
	#img=misc.imread('n03709823_424.JPEG')
	img=misc.imread(img_file) #numpy array
	re_img=misc.imresize(img,[256,256,3])

	re_img = np.array(re_img, dtype=float)
	#print re_img.shape
	#raw_input()
	if len(re_img.shape)<3: # if image is grayscale shape (256,256)
		#print len(re_img.shape)
		dim=re_img.shape+(3,)
		re_img_exp=np.zeros(list(dim))
		for i in range(3):
			re_img_exp[:,:,i]=re_img
		re_img=re_img_exp

		


	im_r=re_img[:,:,0]
	im_g=re_img[:,:,1]
	im_b=re_img[:,:,2]

	#im_pad[0:256,0:256]=im_r
	stride=33
	for ch in range(3):
		im_pad=np.zeros([264,264])

		im_pad[0:256,0:256]=re_img[:,:,ch]
		im_pad = np.array(im_pad, dtype=float)
		block_recon_apply(im_pad,pTp,stride)
		re_img[:,:,ch]=im_pad[:256,:256]
		#re_img[:,:,ch] = re_img[:,:,ch] - np.min(re_img[:,:,ch])
		#re_img[:,:,ch] = re_img[:,:,ch]/np.max(re_img[:,:,ch])*255
		#im_rcs=im_pad[:256,:256]


	re_img = re_img - np.min(re_img)
	re_img = re_img/np.max(re_img)*255
	return re_img



