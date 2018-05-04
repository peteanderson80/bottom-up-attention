# bottom-up-attention

This code implements a bottom-up attention model, based on multi-gpu training of Faster R-CNN with ResNet-101, using object and attribute annotations from [Visual Genome](http://visualgenome.org/).

The pretrained model generates output features corresponding to salient image regions. These bottom-up attention features can typically be used as a drop-in replacement for CNN features in attention-based image captioning and visual question answering (VQA) models. This approach was used to achieve state-of-the-art image captioning performance on [MSCOCO](https://competitions.codalab.org/competitions/3221#results) (**CIDEr 117.9**, **BLEU_4 36.9**) and to win the [2017 VQA Challenge](http://www.visualqa.org/workshop.html) (**70.3%** overall accuracy), as described in:
- [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998), and 
- [Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge](https://arxiv.org/abs/1708.02711). 

Some example object and attribute predictions for salient image regions are illustrated below.

![teaser-bike](data/demo/rcnn_example.png?raw=true)
![teaser-oven](data/demo/rcnn_example_2.png?raw=true)

Note: This repo only includes code for training the bottom-up attention / Faster R-CNN model (section 3.1 of the [paper](https://arxiv.org/abs/1707.07998)). The actual captioning model (section 3.2) is available in a separate repo [here](https://github.com/peteanderson80/Up-Down-Captioner). 

### Reference
If you use our code or features, please cite our paper:
```
@inproceedings{Anderson2017up-down,
  author = {Peter Anderson and Xiaodong He and Chris Buehler and Damien Teney and Mark Johnson and Stephen Gould and Lei Zhang},
  title = {Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering},
  booktitle={CVPR},
  year = {2018}
}
```

### Disclaimer

This code is modified from [py-R-FCN-multiGPU](https://github.com/bharatsingh430/py-R-FCN-multiGPU), which is in turn modified from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) code. Please refer to these links for further README information (for example, relating to other models and datasets included in the repo) and appropriate citations for these works. This README only relates to Faster R-CNN trained on Visual Genome.

### License

bottom-up-attention is released under the MIT License (refer to the LICENSE file for details).

### Pretrained features

For ease-of-use, we make pretrained features available for the entire [MSCOCO dataset](http://mscoco.org/dataset/#download). It is not necessary to clone or build this repo to use features downloaded from the links below. Features are stored in tsv (tab-separated-values) format that can be read with `tools/read_tsv.py`. 

**LINKS HAVE BEEN UPDATED**

10 to 100 features per image (adaptive):
- [2014 Train/Val Image Features (120K / 23GB)](https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip)
- [2014 Testing Image Features (40K / 7.3GB)](https://imagecaption.blob.core.windows.net/imagecaption/test2014.zip)
- [2015 Testing Image Features (80K / 15GB)](https://imagecaption.blob.core.windows.net/imagecaption/test2015.zip)

36 features per image (fixed):
- [2014 Train/Val Image Features (120K / 25GB)](https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip)
- [2014 Testing Image Features (40K / 9GB)](https://imagecaption.blob.core.windows.net/imagecaption/test2014_36.zip)
- [2015 Testing Image Features (80K / 17GB)](https://imagecaption.blob.core.windows.net/imagecaption/test2015_36.zip)

Both sets of features can be recreated by using `tools/genenerate_tsv.py` with the appropriate pretrained model and with MIN_BOXES/MAX_BOXES set to either 10/100 or 36/36 respectively - refer [Demo](#demo). 

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation)
4. [Demo](#demo)
5. [Training](#training)
6. [Testing](#testing)

### Requirements: software

0. **`Important`** Please use the version of caffe contained within this repository.

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers and NCCL!

  ```make
  # In your Makefile.config, make sure to have these lines uncommented
  WITH_PYTHON_LAYER := 1
  USE_NCCL := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`
3. Nvidia's NCCL library which is used for multi-GPU training https://github.com/NVIDIA/nccl

### Requirements: hardware

Any NVIDIA GPU with 12GB or larger memory is OK for training Faster R-CNN ResNet-101.

### Installation
1. Clone the repository
  ```Shell
  git clone https://github.com/peteanderson80/bottom-up-attention/
  ```

3. Build the Cython modules
    ```Shell
    cd $REPO_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $REPO_ROOT/caffe
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
   ```

### Demo

1.  Download [pretrained model](https://storage.googleapis.com/bottom-up-attention/resnet101_faster_rcnn_final.caffemodel), and put it under `data\faster_rcnn_models`.
   
2.  Run `tools/demo.ipynb` to show object and attribute detections on demo images.

3.  Run `tools/genenerate_tsv.py` to extract bounding box features to a tab-separated-values (tsv) file. This will require modifying the `load_image_ids` function to suit your data locations. To recreate the pretrained feature files with 10 to 100 features per image, set MIN_BOXES=10 and MAX_BOXES=100. To recreate the pretrained feature files with 36 features per image, set MIN_BOXES=36 and MAX_BOXES=36 use this [alternative pretrained model](https://storage.googleapis.com/bottom-up-attention/resnet101_faster_rcnn_final_iter_320000.caffemodel) instead. The alternative pretrained model was trained for fewer iterations but performance is similar.
  

### Training

1. Download the Visual Genome dataset. Extract all the json files, as well as the image directories `VG_100K` and `VG_100K_2` into one folder `$VGdata`.

2. Create symlinks for the Visual Genome dataset

    ```Shell
    cd $REPO_ROOT/data
    ln -s $VGdata vg
    ``` 

3. Generate xml files for each image in the pascal voc format (this will take some time). This script will extract the top 2500/1000/500 objects/attributes/relations and also does basic cleanup of the visual genome data. Note however, that our training code actually only uses a subset of the annotations in the xml files, i.e., only 1600 object classes and 400 attribute classes, based on the hand-filtered vocabs found in `data/genome/1600-400-20`. The relevant part of the codebase is `lib/datasets/vg.py`. Relation labels can be included in the data layers but are currently not used.

    ```Shell
    cd $REPO_ROOT
    ./data/genome/setup_vg.py
    ``` 

4.  Please download the ImageNet-pre-trained ResNet-100 model manually, and put it into `$REPO_ROOT/data/imagenet_models`

5.  You can train your own model using `./experiments/scripts/faster_rcnn_end2end_multi_gpu_resnet_final.sh` (see instructions in file). The train (95k) / val (5k) / test (5k) splits are in `data/genome/{split}.txt` and have been determined using `data/genome/create_splits.py`. To avoid val / test set contamination when pre-training for MSCOCO tasks, for images in both datasets these splits match the 'Karpathy' COCO splits. 
  

    Trained Faster-RCNN snapshots are saved under:

    ```
    output/faster_rcnn_resnet/vg/
    ```

    Logging outputs are saved under:

    ```
    experiments/logs/
    ```

6.  Run `tools/review_training.ipynb` to visualize the training data and predictions.

### Testing 

1.  The model will be tested on the validation set at the end of training, or models can be tested directly using `tools/test_net.py`, e.g.:

    ```
    ./tools/test_net.py --gpu 0 --imdb vg_1600-400-20_val --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel > experiments/logs/eval.log 2<&1
    ```

    Mean AP is reported separately for object prediction and attibute prediction (given ground-truth object detections). Test outputs are saved under:

    ```
    output/faster_rcnn_resnet/vg_1600-400-20_val/<network snapshot name>/
    ```

#### Expected detection results for the pretrained model

|                   | objects mAP@0.5     | objects weighted mAP@0.5 | attributes mAP@0.5    | attributes weighted mAP@0.5 |
|-------------------|:-------------------:|:------------------------:|:---------------------:|:---------------------------:|
|Faster R-CNN, ResNet-101 | 10.2%  | 15.1% | 7.8%  | 27.8% |


Note that mAP is relatively low because many classes overlap (e.g. person / man / guy), some classes can't be precisely located (e.g. street, field) and separate classes exist for singular and plural objects (e.g. person / people). We focus on performance in downstream tasks (e.g. image captioning, VQA) rather than detection performance. 


