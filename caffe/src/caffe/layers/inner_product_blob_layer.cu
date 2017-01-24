#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_blob_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductBlobLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = bottom[1]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
  } else {
  //    printf("%d %d %d", M_, N_, K_);
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
  }
}

template <typename Dtype>
void InnerProductBlobLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}


INSTANTIATE_LAYER_GPU_FUNCS(InnerProductBlobLayer);

}  // namespace caffe
