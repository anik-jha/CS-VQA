#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os, h5py, sys, argparse
import time
import math
import codecs, json
from LSTM_CNN import LSTM_CNN
from try_vqacs_tfmodel_1 import vqatest
from sklearn.metrics import average_precision_score


def parse_ans(params):
    with open('data.json') as data_file:
        data = json.load(data_file)

    for i in xrange(0, 121512):
        #print i
        data[i]['question_id'] = int(data[i]['question_id'])

    #dd = json.dump(data,open('OpenEnded_mscoco_lstmori_results.json','w'))
    out_path= '/media/ssd/lichi/VQA1_0_01/OutputFiles/Recon_01/OpenEnded_mscoco_val2014_'+params['output_anstype']+'_results.json'
    dd = json.dump(data,open(out_path,'w'))
    print 'finished generated %s' %(out_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_anstype', default='recon_01_lr_0003_test_03', help='output openended file')
    parser.add_argument('--model', default='recon_lstm_0003-1200000', help='trained_tf_model')
    args = parser.parse_args()
    params = vars(args)
    with tf.device('/gpu:0'):
    #with tf.device('cpu'):
        vqatest(params)
    parse_ans(params)
    
