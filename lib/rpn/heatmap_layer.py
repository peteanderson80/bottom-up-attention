
import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
DEBUG = False

class HeatmapLayer(caffe.Layer):
    """
    Takes regions of interest (rois) and outputs heatmaps.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._output_w = layer_params['output_w']
        self._output_h = layer_params['output_h']
        self._out_size = np.array([self._output_w, self._output_h, 
              self._output_w, self._output_h],dtype=float)
        top[0].reshape(bottom[0].data.shape[0], 1, self._output_h, self._output_w)

    def forward(self, bottom, top):
        # im_info (height, width, scaling)
        assert bottom[1].data.shape[0] == 1, 'Batch size == 1 only'
        image_h = bottom[1].data[0][0]
        image_w = bottom[1].data[0][1]
        image_size = np.array([image_w, image_h, image_w, image_h],dtype=float)
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        rois = bottom[0].data
        rois = rois.reshape(rois.shape[0], rois.shape[1])
        rois = rois[:,1:]*self._out_size/image_size
        # This will fill occupied pixels in an approximate (dilated) fashion
        rois_int = np.round(np.hstack((
            np.floor(rois[:,[0]]), 
            np.floor(rois[:,[1]]),
            np.minimum(self._output_w-1,np.ceil(rois[:,[2]])), 
            np.minimum(self._output_h-1,np.ceil(rois[:,[3]]))
        ))).astype(int)
        top[0].reshape(bottom[0].data.shape[0], 1, self._output_h, self._output_w)
        top[0].data[...] = -1
        for i in range(rois.shape[0]):
            top[0].data[i, 0, rois_int[i,1]:rois_int[i,3], rois_int[i,0]:rois_int[i,2]] = 1

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

