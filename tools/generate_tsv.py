#!/usr/bin/env python


"""Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a 
   separate tsv file that can be merged later (e.g. by using merge_tsv function). 
   Modify the load_image_ids script as necessary for your data location. """


# Example:
# ./tools/generate_tsv.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --out test2014_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split coco_test2014


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


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

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
    else:
      print 'Unknown split'
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
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--split', dest='data_split',
                        help='dataset to use',
                        default='karpathy_train', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

    
def generate_tsv(gpu_id, prototxt, weights, image_ids, outfile):
    # First check if file exists, and if it is complete
    wanted_ids = set([int(image_id[1]) for image_id in image_ids])
    found_ids = set()
    if os.path.exists(outfile):
        with open(outfile) as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = FIELDNAMES)
            for item in reader:
                found_ids.add(int(item['image_id']))
    missing = wanted_ids - found_ids
    if len(missing) == 0:
        print 'GPU {:d}: already completed {:d}'.format(gpu_id, len(image_ids))
    else:
        print 'GPU {:d}: missing {:d}/{:d}'.format(gpu_id, len(missing), len(image_ids))
    if len(missing) > 0:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        net = caffe.Net(prototxt, caffe.TEST, weights=weights)
        with open(outfile, 'ab') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)   
            _t = {'misc' : Timer()}
            count = 0
            for im_file,image_id in image_ids:
                if int(image_id) in missing:
                    _t['misc'].tic()
                    writer.writerow(get_detections_from_im(net, im_file, image_id))
                    _t['misc'].toc()
                    if (count % 100) == 0:
                        print 'GPU {:d}: {:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                              .format(gpu_id, count+1, len(missing), _t['misc'].average_time, 
                              _t['misc'].average_time*(len(missing)-count)/3600)
                    count += 1

                    


def merge_tsvs():
    test = ['/work/data/tsv/test2015/resnet101_faster_rcnn_final_test.tsv.%d' % i for i in range(8)]

    outfile = '/work/data/tsv/merged.tsv'
    with open(outfile, 'ab') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)   
        
        for infile in test:
            with open(infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                for item in reader:
                    try:
                      writer.writerow(item)
                    except Exception as e:
                      print e                           

                      
     
if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

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
        outfile = '%s.%d' % (args.outfile, gpu_id)
        p = Process(target=generate_tsv,
                    args=(gpu_id, args.prototxt, args.caffemodel, image_ids[i], outfile))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()            
                  
