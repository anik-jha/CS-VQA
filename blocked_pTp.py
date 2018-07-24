import scipy.io
import numpy as np
from scipy import misc
import time
import matplotlib.pyplot as plt

def block_apply(im_pad,pTp,stride):
	for x in range(8):
		for y in range(8):
			#sub_img=np.zeros([33,33])
			sub_img=im_pad[x*stride:33*(x+1),y*stride:33*(y+1)]
			sub_img=np.reshape(sub_img,[1089,1])
			sub_img = np.array(sub_img, dtype=float)
			sub_img_cs=np.dot(pTp,sub_img)

			sub_img_cs=np.reshape(sub_img_cs,[33,33])
			#plt.imshow(sub_img_cs)
			#plt.show()
			#raw_input("Press the <ENTER> key to continue...")
			im_pad[x*stride:33*(x+1),y*stride:33*(y+1)]=sub_img_cs

	


mat=scipy.io.loadmat('phi_0_25_1089.mat') #random sensing matrix 1089X1
phi=mat['phi']
phiT=np.transpose(phi)
pTp=np.dot(phiT,phi) #1089 by 1089
#pTp = np.identity(1089)	`
def blocked_pTp(img_file):
	#img = misc.face()
	#img=misc.imread('n03709823_424.JPEG')
	img=misc.imread(img_file) #numpy array
	re_img=misc.imresize(img,[256,256,3])

	re_img = np.array(re_img, dtype=float)
	#print re_img.shape
	#raw_input()
	if len(re_img.shape)<3: # if image is grayscale shape (256,256)
	#	print len(re_img.shape)
		dim=re_img.shape+(3,)
		re_img_exp=np.zeros(list(dim))
		for i in range(3):
			re_img_exp[:,:,i]=re_img
		re_img=re_img_exp

		

	#operate each channel r,g,b seperately
	im_r=re_img[:,:,0]
	im_g=re_img[:,:,1]
	im_b=re_img[:,:,2]

	#im_pad[0:256,0:256]=im_r
	stride=33
	
	for ch in range(3):
		im_pad=np.zeros([264,264])

		im_pad[0:256,0:256]=re_img[:,:,ch]
		im_pad = np.array(im_pad, dtype=float)
		block_apply(im_pad,pTp,stride)
		re_img[:,:,ch]=im_pad[:256,:256]
		#re_img[:,:,ch] = re_img[:,:,ch] - np.min(re_img[:,:,ch])
		#re_img[:,:,ch] = re_img[:,:,ch]/np.max(re_img[:,:,ch])*255
		#im_rcs=im_pad[:256,:256]


	re_img = re_img - np.min(re_img)
	re_img = re_img/np.max(re_img)*255
	return re_img
	#output is 256 by 256 blocked phiTphix image 
#im_rcs=im_pad[:256,:256]

#re_img=blocked_pTp('n04296562_38428.JPEG')
#plt.imshow(re_img.astype(np.uint8))
#plt.show()
