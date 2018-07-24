
#!/usr/bin/env sh
# train caffe network

LOG=/media/ssd/lichi/VQA1_0_01/caffe_model/log_e1_a0_0001_b64_wd0_001_dor_09.txt
DATA=/media/ssd/lichi/VQA1_0_01/Datasets
ROOT=/home/anik/caffe
CAFFE=/home/anik/caffe/build/tools
PROTO=blpsu_googlenet_solver.prototxt
CODE= /media/ssd/lichi/VQA1_0_01/codes

$CAFFE/caffe train --solver=$ROOT/examples/imagenet/$PROTO -gpu 0 2>&1|tee $LOG

echo "Done."
