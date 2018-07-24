# Following is the sequence in which the codes needs to be executed
 ## Using blocked-based phiTphix
  + blocked_pTp.py:                  operate phiTphiX for one file
  + blpTp_forVQA.py:                 iterate over directory for files

 ## Using phiTphiX
   + make_validation_hadamard_old.m

 ## Using ReconNet
  + test_imagenet_cc.m:              reconstruct ImageNet
  + test_imagenet_val.m
  + reconet_forVQA.py:               iterate over directory for files
  + blpTp_recon.py:                  reconstruct VQA 

 ## Create LMDB
  + create_imagenet.sh

 ## Training CNN
  + blpsu_googlenet.prototxt:        specify CNN architecture and input data
  + blpsu_googlenet_solver.prototxt: specify hyperparameter for training and snapshot

 ## Run caffe training
  + train_caffe.sh

 ## VQA preprocessing
  + vqa_cs_preprocessing.py:          generate vqa_raw.json
  + prepro.py:                        take json files from above and output a data_prepro.json and data_prepro.h5
  + prepro_img_param_nobatch.py:      feedforward caffemodel to generate hdf5 image feature

 ## VQA processing and results
  + train_vqacs_tfmodel.py:           training the tensorflow model
  + vqacs_tfmodel_results.py:         testing and output json file for answer
  + vqaEvalDemo.py:                   evaluating json file 
