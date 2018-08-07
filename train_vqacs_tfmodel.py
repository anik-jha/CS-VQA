#-*- coding: utf-8 -*-
import tensorflow as tf
#import pandas as pd
from LSTM_CNN import LSTM_CNN
import numpy as np
import os, h5py, sys, argparse
import time
import math
import cv2
import codecs, json
#import tf.nn.rnn_cell as rnn_cell
from sklearn.metrics import average_precision_score
import math

from tensorflow.python.client import timeline #for tracing performance

rnn_cell=tf.nn.rnn_cell

def dim_mul(dim_image):
    dim_mul=1
    for i in range(len(dim_image)):
      dim_mul=dim_mul*dim_image[i]
    return dim_mul

   

#####################################################
#                 Global Parameters         #  
#####################################################
print 'Loading parameters ...'
# Data input setting

input_img_h5 = '/path/to/image_features/feat_imgs_blgoogle485_fc_nobatch.h5'
input_ques_h5 = '/path/to/root_folder/bl_data_prepro.h5'
input_json = '/path/to/root_folder/codes/bl_data_prepro.json'


# Train Parameters setting
learning_rate = 0.0003          # learning rate for rmsprop
#starter_learning_rate = 3e-4
learning_rate_decay_start = -1      # at what iteration to start decaying learning rate? (-1 = dont)
batch_size = 1            # batch_size for each iterations
input_embedding_size = 200      # the encoding size of each token in the vocabulary
rnn_size = 512              # size of the rnn in number of hidden nodes in each layer
rnn_layer = 2               # number of the rnn layer
#dim_image = 1000  #loss3/classifier 
#dim_image = 4096  #original image feature
dim_image_emd = 1024  #feat_imgs_487_pool_nobatch.h5
dim_image = 1024  	#pool5/7x7_s1
dim_hidden = 1024 #1024         # size of the common embedding vector
num_output = 1000           # number of output answers
img_norm = 0                # normalize the image feature. 1 = normalize, 0 = not normalize
img_tran =False
restore_lstm= False
restore_googlenet= False
decay_factor = 0.99997592083 # for batch size=500
#decay_factor = 0.99998762083 # for batch size=20
#decay_factor = 1.0
decay_steps=100000
#factor= 0.6

# Check point
checkpoint_path = 'bl_tf_model/'
model_name= 'bl_lstm_0003'
# misc
gpu_id = 0 
max_itr = 1200000
n_epochs = 300
max_words_q = 26
num_answer = 1000

#restore_model= "./recon_tf_model_2/model_blgoogle_incpt5b-80000"
restore_lstm= False  # if restore incpt5a 
restore_googlenet= False
restore_model= False
#####################################################
def get_data():

    dataset = {}
    train_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # load image feature
    print('loading image feature...')
    with h5py.File(input_img_h5,'r') as hf:
        # -----0~82459------
        tem = hf.get('imgs_train')
        #tem = hf.get('images_train') # for original image feature deta_img_ori.h5
        img_feature = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)-1
        # max length is 23
        #tem = hf.get('ques_len_train')
        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)
        # total 82460 img
        tem = hf.get('img_pos_train')
    # convert into 0~82459
        train_data['img_list'] = np.array(tem)-1
        # answer is 1~1000
        tem = hf.get('answers')
        train_data['answers'] = np.array(tem)-1

    print('question aligning')
    #print "length_q =", length_q
    #print "train_data" , train_data
    train_data['question'] = right_align(train_data['question'], train_data['length_q'])

    print('Normalizing image feature')
    if img_norm:
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
        #img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(1000,1))))  #for classifier
        #img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(4096,1)))) #for testing
        img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(dim_image,1)))) #for testing
    return dataset, img_feature, train_data

def get_data_test():
    dataset = {}
    test_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # load image feature
    print('loading image feature...')
    with h5py.File(input_img_h5,'r') as hf:
        tem = hf.get('imgs_test')
        img_feature = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_test')
        test_data['question'] = np.array(tem)-1
        # max length is 23
        #tem = hf.get('ques_len_test')
        tem = hf.get('ques_length_test')
        test_data['length_q'] = np.array(tem)
        # total 82460 img
        tem = hf.get('img_pos_test')
        # convert into 0~82459
        test_data['img_list'] = np.array(tem)-1
        # quiestion id
        tem = hf.get('question_id_test')
        test_data['ques_id'] = np.array(tem)
        # MC_answer_test
        # tem = hf.get('MC_ans_test')
        # test_data['MC_ans_test'] = np.array(tem)


    print('question aligning')
    test_data['question'] = right_align(test_data['question'], test_data['length_q'])

    print('Normalizing image feature')
    if img_norm:
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
        #img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(1000,1)))) #for loss3
        img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(1024,1)))) #for loss3
        #img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(4096,1)))) #for testing
    return dataset, img_feature, test_data
def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    #print "shape =", np.shape(seq)[0], " ", np.shape(seq)[1]
    
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        #print "length = ", lengths 
	v[i][N-lengths[i]:N-1]=seq[i][0:lengths[i]-1]
    return v



def train():
    for d in ['/gpu:0']:
        with tf.device(d):
            tStart_total=time.time()

            #tStart = time.time()
            print 'loading dataset...'
            dataset, img_feature, train_data = get_data()
            
            num_train = train_data['question'].shape[0]
            vocabulary_size = len(dataset['ix_to_word'].keys())
            print 'vocabulary_size : ' + str(vocabulary_size)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
            sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,device_count={'GPU': 0},gpu_options=gpu_options))
        #with tf.device("/gpu:1"):
            print 'constructing  model...'
        #--------------model section------------------------------------#    

           
            model=LSTM_CNN(
                batch_size=batch_size,
                input_embedding_size = input_embedding_size,
                lstm_size=rnn_size,
                lstm_layer=rnn_layer,
                dim_image=dim_image,
                dim_hidden=dim_hidden,
                vocabulary_size=vocabulary_size,
                max_words_q=max_words_q,
                drop_out_rate = 0.5,
                output_size=1000,
                dim_img_emd=1024)
            tf_loss, tf_image, tf_question, tf_label = model.build_model()
            
            '''
            model=stacked_att_model(
                batch_size=batch_size,
                input_embedding_size = input_embedding_size,
                lstm_size=rnn_size,
                lstm_layer=rnn_layer,
                dim_image=dim_image,
                dim_hidden=dim_hidden,
                att_dim=1024,
                vocabulary_size=vocabulary_size,
                max_words_q=max_words_q,
                drop_out_rate = 0.5,
                im_reg=49,
                output_size=1000)
            tf_loss, tf_image, tf_question, tf_label = model.build_model()
	'''

 
            #--------------model section------------------------------------#
            saver = tf.train.Saver(max_to_keep=100)
            
            tvars = tf.trainable_variables()
	    for t in tvars:
		print t.name
            lr = tf.Variable(learning_rate)
            #opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
	    opt = tf.train.AdamOptimizer(learning_rate=lr)
            #sess.run(tf_loss)
            # gradient clipping
            gvs = opt.compute_gradients(tf_loss,tvars)
            clipped_gvs = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs]
	    #pdb.set_trace()
	    #tf.global_variables_initializer().run
            train_op = opt.apply_gradients(clipped_gvs)
        #with tf.device("/gpu:1"): #3 not working
            
	    init_op = tf.global_variables_initializer()
            sess.run(init_op)
	    #tf.global_variables_initializer().run
            
            if restore_lstm:
                model.restore_lstmw(sess)
            if restore_googlenet:
                model.restore_weight('chkpt-0',sess)
            if restore_model:
                saver.restore(sess, restore_model)
                print 'model %s restore' %(restore_model)
            print 'start training...'
            base_itr=0
            base_lr=1.0
            if restore_model:
                base_itr=r_step+1
                #base_lr= factor**(base_itr/100000)
                base_lr=base_lr*(decay_factor**base_itr)

            current_learning_rate = learning_rate*base_lr
            lr.assign(current_learning_rate).eval()
            print 'base learning rate is',current_learning_rate
            run_metadata = tf.RunMetadata()
            print 'model name is ',model_name
            
            for itr in range(base_itr,max_itr):
                
                
                # shuffle the training data
                index = np.random.random_integers(0, num_train-1, batch_size)
                
                current_question = train_data['question'][index,:]
                current_length_q = train_data['length_q'][index]
                current_answers = train_data['answers'][index]
                current_img_list = train_data['img_list'][index]
                #to replace 82459 since shape of original image feature
                #current_img_list= [0 if x==82459 else x for x in current_img_list]
                #---------------------------------------------#
                current_img = img_feature[current_img_list,:]
                tStart = time.time()
                # do the training process!!!
                _, loss = sess.run(
                       [train_op, tf_loss],
                       feed_dict={
                           tf_image: current_img,
                            tf_question: current_question,
                            tf_label: current_answers
                            })
                # tracing the performance

                if np.mod(itr,5000) ==0 and itr is not 0:
                    current_learning_rate = current_learning_rate*pow(decay_factor,5000)
                    #current_learning_rate = current_learning_rate*pow(decay_factor,300) # for batch size 32
                    lr.assign(current_learning_rate).eval()
                #lr.assign(current_learning_rate).eval()
                #print 'lr is ',lr.eval()
                tStop = time.time()
                if np.mod(itr, 100) == 0:
                    print "Iteration: ", itr, " Loss: ", loss, " Learning Rate: ", lr.eval()
                    print ("Time Cost:", round(tStop - tStart,4), "s")
                    #lr.assign(current_learning_rate).eval() #speed up ?(0.4 for one itr) 
                if np.mod(itr, 10000) == 0:
                #if np.mod(itr, 50000) == 0: # for batch size 32
                    print "Iteration ", itr, " is done. Saving the model ..."
                    saver.save(sess, os.path.join(checkpoint_path, model_name), global_step=itr)

            print "Finally, saving the model ..."
            saver.save(sess, os.path.join(checkpoint_path, model_name), global_step=max_itr)
            tStop_total = time.time()
            print "Total Time Cost:", round(tStop_total - tStart_total,2), "s"


def vqatest(params):
    batch_size=1
    print 'loading dataset...'
    model_path='bl_tf_model/'+params['model']
    print 'using model %s' %(model_path)
    dataset, img_feature, test_data = get_data_test()
    num_test = test_data['question'].shape[0]
    vocabulary_size = len(dataset['ix_to_word'].keys())
    print 'vocabulary_size : ' + str(vocabulary_size)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,device_count={'GPU': 0},gpu_options=gpu_options))

    '''
    model=stacked_att_model(
                batch_size=batch_size,
                input_embedding_size = input_embedding_size,
                lstm_size=rnn_size,
                lstm_layer=rnn_layer,
                dim_image=dim_image,
                dim_hidden=dim_hidden,
                att_dim=1024,
                vocabulary_size=vocabulary_size,
                max_words_q=max_words_q,
                drop_out_rate = 0,
                im_reg=49,
                output_size=1000)
    tf_answer, tf_image, tf_question, = model.build_generator()
    
    '''
    # model=stacked_att_conv_model(
    #             batch_size=batch_size,
    #             input_embedding_size = input_embedding_size,
    #             lstm_size=rnn_size,
    #             lstm_layer=rnn_layer,
    #             dim_image=dim_image,
    #             dim_hidden=dim_hidden,
    #             att_dim=1024,
    #             vocabulary_size=vocabulary_size,
    #             max_words_q=max_words_q,
    #             drop_out_rate = 0.5,
    #             im_reg=49,
    #             output_size=1000)
    # tf_answer, tf_image, tf_question, = model.build_generator()
    '''
    '''
    model=LSTM_CNN(
                batch_size=batch_size,
                input_embedding_size = input_embedding_size,
                lstm_size=rnn_size,
                lstm_layer=rnn_layer,
                dim_image=dim_image,
                dim_hidden=dim_hidden,
                vocabulary_size=vocabulary_size,
                max_words_q=max_words_q,
                drop_out_rate = 0,
                output_size=1000,
                dim_img_emd=1024)
    tf_answer, tf_image, tf_question = model.build_generator()
    

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    tStart_total = time.time()
    result = []
    time_1000=[]
    for current_batch_start_idx in xrange(0,num_test,batch_size):
    #for current_batch_start_idx in xrange(0,3,batch_size):
        
        # set data into current*
        if current_batch_start_idx + batch_size < num_test:
            current_batch_file_idx = range(current_batch_start_idx,current_batch_start_idx+batch_size)
        else:
            current_batch_file_idx = range(current_batch_start_idx,num_test)

        current_question = test_data['question'][current_batch_file_idx,:]
        current_length_q = test_data['length_q'][current_batch_file_idx]
        current_img_list = test_data['img_list'][current_batch_file_idx]
        current_ques_id  = test_data['ques_id'][current_batch_file_idx]
        
        current_img = img_feature[current_img_list,:] # (batch_size, dim_image)
        tStart = time.time()
        # deal with the last batch
        if(len(current_img)<batch_size):
                #pad_img = np.zeros((500-len(current_img),dim_image),dtype=np.int)
                if type(dim_image) is list:
                    pad_img = np.zeros(([batch_size-len(current_img)]+dim_image),dtype=np.int)
                else :
                    pad_img = np.zeros((500-len(current_img),dim_image),dtype=np.int)
                pad_q = np.zeros((batch_size-len(current_img),max_words_q),dtype=np.int)
                pad_q_len = np.zeros(batch_size-len(current_length_q),dtype=np.int)
                pad_q_id = np.zeros(batch_size-len(current_length_q),dtype=np.int)
                pad_ques_id = np.zeros(batch_size-len(current_length_q),dtype=np.int)
                pad_img_list = np.zeros(batch_size-len(current_length_q),dtype=np.int)
                current_img = np.concatenate((current_img, pad_img))
                current_question = np.concatenate((current_question, pad_q))
                current_length_q = np.concatenate((current_length_q, pad_q_len))
                current_ques_id = np.concatenate((current_ques_id, pad_q_id))

                current_img_list = np.concatenate((current_img_list, pad_img_list))


        generated_ans = sess.run(
                tf_answer,
                feed_dict={
                    tf_image: current_img,
                    tf_question: current_question
                    })

        top_ans = np.argmax(generated_ans, axis=1)


        # initialize json list
        for i in xrange(0,batch_size):
            ans = dataset['ix_to_ans'][str(top_ans[i]+1)]
            if(current_ques_id[i] == 0):
                continue
            result.append({u'answer': ans, u'question_id': str(current_ques_id[i])})

        tStop = time.time()
        itr=current_batch_file_idx[0]
        if np.mod(itr, 1000) == 0:
            print (" average Time Cost:", round((sum(time_1000)/1000.0),4), "s")
            time_1000=[]
        print ("Testing batch: ", current_batch_file_idx[0])
        #print ("Time Cost:", round(tStop - tStart,4), "s")
        time_1000.append(tStop - tStart)
    print ("Testing done.")
    tStop_total = time.time()
    print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")
    # Save to JSON
    print 'Saving result...'
    my_list = list(result)
    dd = json.dump(my_list,open('data.json','w'))

if __name__ == '__main__':
    #with tf.device("/gpu:1"):
    train()
    #with tf.device('/gpu:'+str(1)):
    #    test()
    #with tf.device('/cpu:0'):
    #    test()
    
