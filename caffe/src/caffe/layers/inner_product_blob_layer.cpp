#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_blob_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductBlobLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  transpose_ = this->layer_param_.inner_product_param().transpose();
}

template <typename Dtype>
void InnerProductBlobLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  M_ = bottom[0]->count(0,1);
  N_ = bottom[1]->count(0,1);
  K_ = bottom[0]->count(1,2);

  // Figure out the dimensions
  top[0]->Reshape(M_, N_, 1, 1);
}

template <typename Dtype>
void InnerProductBlobLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void InnerProductBlobLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductBlobLayer);
#endif

INSTANTIATE_CLASS(InnerProductBlobLayer);
REGISTER_LAYER_CLASS(InnerProductBlob);

}  // namespace caffe
