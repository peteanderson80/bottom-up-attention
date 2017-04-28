#!/bin/bash
# Usage:
# ./experiments/scripts/test.sh GPU NET MODEL DATASET [options args to test_net.py]
# MODEL is either obj, attr, attr_rel, attr_rel_softmax_primed, attr_rel_softmax_primed_heatmap, attr_rel_softmax_primed
# DATASET is either pascal_voc or coco or vg.
#
# Example:
# ./experiments/scripts/test.sh 0 VGG16 attr vg \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
MODEL=$3
DATASET=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=70000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=490000
    ;;
  vg)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="vg_train"
    TEST_IMDB="vg_val"
    PT_DIR="vg"
    ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/test_${NET}_${MODEL}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
  
NET_FINAL="/work/code/py-R-FCN-multiGPU/output/faster_rcnn_end2end/vg_train/vgg16_faster_rcnn_${MODEL}_iter_490000.caffemodel"
time ./tools/test_net.py --gpu ${GPU_ID:0:1} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end_${MODEL}/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}
