#!/usr/bin/env python


"""Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a 
   separate tsv file that can be merged later (e.g. by using merge_tsv function). 
   Modify the load_image_ids script as necessary for your data location. """


# Example:
# ./tools/extract_features_to_npz.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --out test2014_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split coco_test2014


import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import caffe
import argparse
import pprint
import time, os, sys
import base64
import numpy as np
import cv2
import csv
from multiprocessing import Process
import random
import json

csv.field_size_limit(sys.maxsize)


# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36. 
MIN_BOXES = 10
MAX_BOXES = 100

def load_image_ids(split_name):
  ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
  split = []
  if split_name == 'coco_test2014':
    with open('/data/coco/annotations/image_info_test2014.json') as f:
      data = json.load(f)
      for item in data['images']:
        image_id = int(item['id'])
        filepath = os.path.join('/data/test2014/', item['file_name'])
        split.append((filepath,image_id))
  elif split_name == 'coco_test2015':
    with open('/data/coco/annotations/image_info_test2015.json') as f:
      data = json.load(f)
      for item in data['images']:
        image_id = int(item['id'])
        filepath = os.path.join('/data/test2015/', item['file_name'])
        split.append((filepath,image_id))
  elif split_name == 'genome':
    with open('/data/visualgenome/image_data.json') as f:
      for item in json.load(f):
        image_id = int(item['image_id'])
        filepath = os.path.join('/data/visualgenome/', item['url'].split('rak248/')[-1])
        split.append((filepath,image_id))
  elif split_name == 'vizwiz_train':
    with open('/data/datasets/original/VizWiz_final/Annotations/train.json') as f:
      for item in json.load(f):
        image_id = item['image']
        filepath = os.path.join('/data/datasets/original/VizWiz_final/train', image_id)
        split.append((filepath,image_id))
  elif split_name == 'vizwiz_val':
    with open('/data/datasets/original/VizWiz_final/Annotations/val.json') as f:
      for item in json.load(f):
        image_id = item['image']
        filepath = os.path.join('/data/datasets/original/VizWiz_final/val', image_id)
        split.append((filepath,image_id))
  elif split_name == 'vizwiz_test':
    with open('/data/datasets/original/VizWiz_final/Annotations/test.json') as f:
      for item in json.load(f):
        image_id = item['image']
        filepath = os.path.join('/data/datasets/original/VizWiz_final/test', image_id)
        split.append((filepath,image_id))
  else:
    print('Unknown split')
  return split

    
def get_detections_from_im(net, im_file, image_id, conf_thresh=0.2):
  im = cv2.imread(im_file)
  scores, boxes, attr_scores, rel_scores = im_detect(net, im)

  # Keep the original boxes, don't worry about the regresssion bbox outputs
  rois = net.blobs['rois'].data.copy()
  # unscale back to raw image space
  blobs, im_scales = _get_blobs(im, None)

  cls_boxes = rois[:, 1:5] / im_scales[0]
  cls_prob = net.blobs['cls_prob'].data
  pool5 = net.blobs['pool5_flat'].data

  # Keep only the best detections
  max_conf = np.zeros((rois.shape[0]))
  for cls_ind in range(1,cls_prob.shape[1]):
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    keep = np.array(nms(dets, cfg.TEST.NMS))
    max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

  keep_boxes = np.where(max_conf >= conf_thresh)[0]
  if len(keep_boxes) < MIN_BOXES:
    keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
  elif len(keep_boxes) > MAX_BOXES:
    keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
  
  return {
    'image_id': image_id,
    'image_h': np.size(im, 0),
    'image_w': np.size(im, 1),
    'num_boxes' : len(keep_boxes),
    'boxes': base64.b64encode(cls_boxes[keep_boxes]),
    'features': base64.b64encode(pool5[keep_boxes])
  }


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default=None, type=str)
    parser.add_argument('--out', dest='outdir',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--split', dest='data_split',
                        help='dataset to use',
                        default='karpathy_train', type=str)

    if len(sys.argv) == 1:
      parser.print_help()
      sys.exit(1)

    args = parser.parse_args()
    return args


def write_npz(gpu_id, prototxt, weights, image_ids, outdir):
    # First check if file exists, and if it is complete
    if not os.path.exists(outdir):
      os.makedirs(outdir)
    
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffe.TEST, weights=weights)
    for im_file, image_id in image_ids:
      out_id = image_id.replace(".jpg", ".npz")
      out_file = os.path.join(outdir, out_id)
      if os.path.exists(out_file):
        print("{} existed, so skipping...".format(out_file))
        continue

      ret = get_detections_from_im(net, im_file, image_id)
      x = np.transpose(ret["features"])
      print("Writing image features to {}".format(out_file))
      np.savez_compressed(out_file, x=x, num_bbox=ret["num_boxes"],
          bbox=ret["boxes"], image_h=ret["image_h"], image_w=ret["image_w"])

     
if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
      cfg_from_file(args.cfg_file)

    gpu_id = args.gpu_id
    gpu_list = gpu_id.split(',')
    gpus = [int(i) for i in gpu_list]

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    image_ids = load_image_ids(args.data_split)
    random.seed(10)
    random.shuffle(image_ids)
    # Split image ids between gpus
    image_ids = [image_ids[i::len(gpus)] for i in range(len(gpus))]
    
    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))
    procs = []    
    
    for i,gpu_id in enumerate(gpus):
      p = Process(target=write_npz,
                  args=(gpu_id, args.prototxt, args.caffemodel, image_ids[i], args.outdir))
      p.daemon = True
      p.start()
      procs.append(p)
    for p in procs:
      p.join()
                  
