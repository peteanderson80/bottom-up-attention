#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end_multi_gpu_resnet_final.sh GPU DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco or vg.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end_multi_gpu_resnet_final.sh 0,1,2,3,4,5,6,7 vg
#

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET="ResNet-101"
NET_lc=${NET,,}
DATASET=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
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
    # This is a very conservative training schedule
    TRAIN_IMDB="vg_1600-400-20_train"
    TEST_IMDB="vg_1600-400-20_val"
    PT_DIR="vg"
    ITERS=380000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/faster_rcnn_end2end_final_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net_multi_gpu.py --gpu ${GPU_ID} \
 --solver models/${PT_DIR}/${NET}/faster_rcnn_end2end_final/solver.prototxt \
 --weights data/imagenet_models/${NET}-model.caffemodel \
 --imdb ${TRAIN_IMDB} \
 --iters ${ITERS} \
 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml \
 ${EXTRA_ARGS}

 set +x
 NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
 set -x

time ./tools/test_net.py --gpu ${GPU_ID:0:1} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end_rel/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml \
  ${EXTRA_ARGS}

