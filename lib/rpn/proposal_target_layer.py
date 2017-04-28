# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
DEBUG = False
DEBUG_SHAPE = False

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        self._count = 0
        self._fg_num = 0
        self._bg_num = 0
        layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params['num_classes']
        if 'num_attr_classes' in layer_params:
            self._num_attr_classes = layer_params['num_attr_classes']
        else:
            self._num_attr_classes = 0
        if 'num_rel_classes' in layer_params:
            self._num_rel_classes = layer_params['num_rel_classes']
        else:
            self._num_rel_classes = 0    
        if 'ignore_label' in layer_params:
            self._ignore_label = layer_params['ignore_label']
        else:
            self._ignore_label = -1

        rois_per_image = 1 if cfg.TRAIN.BATCH_SIZE == -1 else cfg.TRAIN.BATCH_SIZE           
        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(rois_per_image, 5, 1, 1)
        # labels
        top[1].reshape(rois_per_image, 1, 1, 1)
        # bbox_targets
        top[2].reshape(rois_per_image, self._num_classes * 4, 1, 1)
        # bbox_inside_weights
        top[3].reshape(rois_per_image, self._num_classes * 4, 1, 1)
        # bbox_outside_weights
        top[4].reshape(rois_per_image, self._num_classes * 4, 1, 1)
        ix = 5
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image).astype(int)
        if self._num_attr_classes > 0:
            # attribute labels
            top[ix].reshape(fg_rois_per_image, 16)
            ix += 1
        if self._num_rel_classes > 0:
            # relation labels
            top[ix].reshape(fg_rois_per_image*fg_rois_per_image, 1, 1, 1)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label, attributes[16], relations[num_objs])
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data
        gt_boxes = gt_boxes.reshape(gt_boxes.shape[0], gt_boxes.shape[1])
        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :4])))
        )

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        rois_per_image = np.inf if cfg.TRAIN.BATCH_SIZE == -1 else cfg.TRAIN.BATCH_SIZE
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

        # Sample rois with classification labels and bounding box regression
        # targets
        # print 'proposal_target_layer:', fg_rois_per_image
        labels, rois, bbox_targets, bbox_inside_weights, attributes, relations = _sample_rois(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes, self._num_attr_classes, 
            self._num_rel_classes, self._ignore_label)
        if self._num_attr_classes > 0:
            assert attributes is not None
        if self._num_rel_classes > 0:
            assert relations is not None

        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        # sampled rois
        # modified by ywxiong
        rois = rois.reshape((rois.shape[0], rois.shape[1], 1, 1))
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        # classification labels
        # modified by ywxiong
        labels = labels.reshape((labels.shape[0], 1, 1, 1))
        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

        # bbox_targets
        # modified by ywxiong
        bbox_targets = bbox_targets.reshape((bbox_targets.shape[0], bbox_targets.shape[1], 1, 1))
        top[2].reshape(*bbox_targets.shape)
        top[2].data[...] = bbox_targets

        # bbox_inside_weights
        # modified by ywxiong
        bbox_inside_weights = bbox_inside_weights.reshape((bbox_inside_weights.shape[0], bbox_inside_weights.shape[1], 1, 1))
        top[3].reshape(*bbox_inside_weights.shape)
        top[3].data[...] = bbox_inside_weights

        # bbox_outside_weights
        # modified by ywxiong
        bbox_inside_weights = bbox_inside_weights.reshape((bbox_inside_weights.shape[0], bbox_inside_weights.shape[1], 1, 1))
        top[4].reshape(*bbox_inside_weights.shape)
        top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)
        
        #attribute labels
        ix = 5
        if self._num_attr_classes > 0:
            attributes[:,1:][attributes[:,1:]==0] = self._ignore_label
            top[ix].reshape(*attributes.shape)
            top[ix].data[...] = attributes
            ix += 1
        
        # relation labels
        if self._num_rel_classes > 0:
            top[ix].reshape(*relations.shape)
            top[ix].data[...] = relations
            
        if DEBUG_SHAPE:
            for i in range(len(top)):
                print 'ProposalTargetLayer top[{}] size: {}'.format(i, top[i].data.shape)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    # print 'proposal_target_layer:', bbox_targets.shape
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    if cfg.TRAIN.AGNOSTIC:
        for ind in inds:
            cls = clss[ind]
            start = 4 * (1 if cls > 0 else 0)
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    else:
        for ind in inds:
            cls = clss[ind]
            start = int(4 * cls)
            end = int(start + 4)
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes, 
          num_attr_classes, num_rel_classes, ignore_label):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
        
    # GT boxes (x1, y1, x2, y2, label, attributes[16], relations[num_objs]) 
    has_attributes = num_attr_classes > 0
    if has_attributes:
        assert gt_boxes.shape[1] >= 21
    has_relations = num_rel_classes > 0
    if has_relations:
        assert gt_boxes.shape[0] == gt_boxes.shape[1]-21, \
            'relationships not found in gt_boxes, item length is only %d' % gt_boxes.shape[1]
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
 
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # print 'proposal_target_layer:', keep_inds
    
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0 / ignore_label
    labels[fg_rois_per_this_image:] = 0
    fg_gt = np.array(gt_assignment[fg_inds])
    if has_attributes:
        attributes = np.ones((fg_rois_per_image,16))*ignore_label
        attributes[:fg_rois_per_this_image,:] = gt_boxes[fg_gt, 5:21]
        np.place(attributes[:,1:],attributes[:,1:] == 0, ignore_label)
    else:
        attributes = None
    if has_relations:
        expand_rels = gt_boxes[fg_gt, 21:].T[fg_gt].T
        num_relations_per_this_image = np.count_nonzero(expand_rels)
        # Keep an equal number of 'no relation' outputs, the rest can be ignore
        expand_rels = expand_rels.flatten()
        no_rel_inds = np.where(expand_rels==0)[0]
        if len(no_rel_inds) > num_relations_per_this_image:
          no_rel_inds = npr.choice(no_rel_inds, size=num_relations_per_this_image, replace=False)
        np.place(expand_rels,expand_rels==0,ignore_label)
        expand_rels[no_rel_inds] = 0
        relations = np.ones((fg_rois_per_image,fg_rois_per_image),dtype=np.float)*ignore_label
        relations[:fg_rois_per_this_image,:fg_rois_per_this_image] = expand_rels.reshape((fg_rois_per_this_image,fg_rois_per_this_image))
        relations = relations.reshape((relations.size, 1, 1, 1))   
    else:
        relations = None
    rois = all_rois[keep_inds]
    
    # print 'proposal_target_layer:', rois
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    # print 'proposal_target_layer:', bbox_target_data
    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights, attributes, relations
