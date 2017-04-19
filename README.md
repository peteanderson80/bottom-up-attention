# py-R-FCN-multiGPU
R-FCN: Object Detection via Region-based Fully Convolutional Networks

py-R-FCN now supports both joint training and alternative optimization. 

py-R-FCN-multiGPU supports multiGPU training of detectors like faster-rcnn and py-R-FCN

## [Soft-NMS](https://github.com/bharatsingh430/soft-nms/) repository gets better results on COCO, so please refer to it.

### Disclaimer

The official R-FCN code (written in MATLAB) is available [here](https://github.com/daijifeng001/R-FCN).

py-R-FCN is modified from [the offcial R-FCN implementation](https://github.com/daijifeng001/R-FCN) and  [py-faster-rcnn code](https://github.com/rbgirshick/py-faster-rcnn ), and the usage is quite similar to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn ).

py-R-FCN-multiGPU is a modified version of [py-R-FCN](https://github.com/Orpine/py-R-FCN). I have heavily reused it's README file as it contains most of the necessary information for running this branch.

There are slight differences between py-R-FCN and the official R-FCN implementation.
 - py-R-FCN is ~10% slower at test-time, because some operations execute on the CPU in Python layers (e.g., 90ms / image vs. 99ms / image for ResNet-50)
 - py-R-FCN supports both join training and alternative optimization of R-FCN.
 
### Multi-GPU Training R-FCN
```
python ./tools/train_net_multi_gpu.py --gpu 0,1 --solver models/pascal_voc/ResNet-101/rfcn_end2end/solver_ohem.prototxt --weights data/imagenet_models/ResNet-101-model.caffemodel  --imdb  voc_2007_trainval+voc_2012_trainval --iters 110000 --cfg experiments/cfgs/rfcn_end2end_ohem.yml
```
or
```
./experiments/scripts/rfcn_end2end_ohem_multi_gpu.sh 0 pascal_voc
```

### Multi-GPU Training Faster-RCNN
```
./experiments/scripts/faster_rcnn_end2end_multi_gpu.sh 0 VGG16 pascal_voc
```

This will use 2 GPUs to perform training. I have set iter_size to 1, so in this case, which is using 2 GPUs, results should be similar. Note that as more GPUs are added, batch size will increase, as it happens in the default multiGPU training in Caffe. The GPU_ID flag in the shell script is only used for testing and if you intent to use more GPUs, please edit it inside the script.

#### Some modification

The original py-faster-rcnn uses class-aware bounding box regression. However, R-FCN use class-agnostic bounding box regression to reduce model complexity. So I add a configuration AGNOSTIC into fast_rcnn/config.py, and the default value is False. You should set it to True both on train and test phase if you want to use class-agnostic training and test. 

OHEM need all rois to select the hard examples, so I changed the sample strategy, set `BATCH_SIZE: -1` for OHEM, otherwise OHEM would not take effect.

In conclusion:

`AGNOSTIC: True` is required for class-agnostic bounding box regression

`BATCH_SIZE: -1` is required for OHEM

And I've already provided two configuration files for you(w/ OHEM and w/o OHEM) under `experiments/cfgs` folder, you could just 
m and needn't change anything.

#### Results on MS-COCO

|                   | training data       | test data          | mAP@[0.5:0.95]   | 
|-------------------|:-------------------:|:-----------------------------:|:-----:|
|R-FCN, ResNet-101  | COCO 2014 train+val -minival | COCO 2014 minival    | 30.8% |
|R-FCN, ResNet-101  | COCO 2014 train+val -minival | COCO 2015 test-dev    | 31.1% |

If you want to use/train this model, please use the coco branch of this repository. The trained model can be found [here](https://drive.google.com/file/d/0B6T5quL13CdHMGtMUWFFSXd2Ym8). Use the config files from the coco branch for this model. Multi-scale training or testing was not done for obtaining this number. Image size was set to 800 and max size was 1200, RPN used 5 scales. This alone obtains 1.6% better than what was reported in the original paper. Training was done on 8 GPUs, with an iter_size of 2.

### License

R-FCN is released under the MIT License (refer to the LICENSE file for details).

### Citing R-FCN

If you find R-FCN useful in your research, please consider citing:

    @article{dai16rfcn,
        Author = {Jifeng Dai, Yi Li, Kaiming He, Jian Sun},
        Title = {{R-FCN}: Object Detection via Region-based Fully Convolutional Networks},
        Journal = {arXiv preprint arXiv:1605.06409},
        Year = {2016}
    }
    


0. **`Important`** Please use the version of caffe uploaded with this repository. I have merged many files between the latest version of Caffe and py-R-FCN.

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  USE_NCCL := 1
  ```
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`
3. Nvidia's NCCL library which is used for multi-GPU training https://github.com/NVIDIA/nccl
4. [Optional] MATLAB is required for **official** PASCAL VOC evaluation only. The code now includes unofficial Python evaluation code.

### Requirements: hardware

Any NVIDIA GPU with 6GB or larger memory is OK(4GB is enough for ResNet-50).


### Installation
1. Clone the R-FCN repository
  ```Shell
  git clone --recursive https://github.com/bharatsingh430/py-R-FCN-multiGPU/
  ```
  (I only test on this commit, and I'm not sure whether this Caffe is still compatible with the prototxt in this repository in the future)
  
  If you followed the above instruction, python code will add `$RFCN_ROOT/caffe/python` to `PYTHONPATH` automatically, otherwise you need to add `$CAFFE_ROOT/python` by your own, you could check `$RFCN_ROOT/tools/_init_paths.py` for more details.

3. Build the Cython modules
    ```Shell
    cd $RFCN_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $RFCN_ROOT/caffe
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
   ```

### Demo
1.  To use demo you need to download the pretrained R-FCN model, please download the model manually from [OneDrive](https://1drv.ms/u/s!AoN7vygOjLIQqUWHpY67oaC7mopf), and put it under `$RFCN/data`. 

    Make sure it looks like this:
    ```Shell
    $RFCN/data/rfcn_models/resnet50_rfcn_final.caffemodel
    $RFCN/data/rfcn_models/resnet101_rfcn_final.caffemodel
    ```

2.  To run the demo
  
    ```Shell
    $RFCN/tools/demo_rfcn.py
    ```
    
  The demo performs detection using a ResNet-101 network trained for detection on PASCAL VOC 2007.


### Preparation for Training & Testing

1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	tar xvf VOCtrainval_11-May-2012.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	$VOCdevkit/VOC2012                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Since py-faster-rcnn does not support multiple training datasets, we need to merge VOC 2007 data and VOC 2012 data manually. Just make a new directory named `VOC0712`, put all subfolders except `ImageSets` in `VOC2007` and `VOC2012` into `VOC0712`(you'll merge some folders). I provide a merged-version `ImageSets` folder for you, please put it into `VOCdevkit/VOC0712/`.

5. Then the folder structure should look like this
  ```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	$VOCdevkit/VOC2012                    # image sets, annotations, etc.
  	$VOCdevkit/VOC0712                    # you just created this folder
  	# ... and several other directories ...
  ```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $RFCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit0712
    ```

5.  Please download ImageNet-pre-trained ResNet-50 and ResNet-100 model manually, and put them into `$RFCN_ROOT/data/imagenet_models`
8.  Then everything is done, you could train your own model.

### Usage

To train and test a R-FCN detector using the **approximate joint training** method, use `experiments/scripts/rfcn_end2end.sh`.
Output is written underneath `$RFCN_ROOT/output`.

To train and test a R-FCN detector using the **approximate joint training** method **with OHEM**, use `experiments/scripts/rfcn_end2end_ohem.sh`.
Output is written underneath `$RFCN_ROOT/output`.

To train and test a R-FCN detector using the **alternative optimization** method **with OHEM**, use `experiments/scripts/rfcn_alt_opt_5stage_ohem.sh`.
Output is written underneath `$RFCN_ROOT/output`

```Shell
cd $RFCN_ROOT
./experiments/scripts/rfcn_end2end[_ohem].sh [GPU_ID] [NET] [DATASET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ResNet-50, ResNet-101} is the network arch to use
# DATASET in {pascal_voc, coco} is the dataset to use(I only tested on pascal_voc)
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

Trained R-FCN networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```

### Misc

Tested on Red Hat with Titan X and Intel Xeon CPU E5-2683 v4 @ 2.10GHz 
