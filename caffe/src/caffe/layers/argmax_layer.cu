#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/argmax_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ArgMaxForward(const int n, const int axis_dist,
    const int dim, const bool has_axis, const bool out_max_val, const Dtype* bottom_data, 
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(i, n) {
    int max_index = 0;
    Dtype max_val = bottom_data[(i / axis_dist * dim) * axis_dist + i % axis_dist];
    for (int j = 0; j < dim; ++j) {
      Dtype curr_val = bottom_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist];
      if (curr_val > max_val) {
        max_val = curr_val;
        max_index = j;
      }
    }
    if (out_max_val) {
      if (has_axis) {
        // Produces max_val per axis
        int index = (i / axis_dist) * axis_dist + i % axis_dist;
        top_data[index]= max_val;
      } else {
        // Produces max_ind and max_val
        top_data[2 * i] = Dtype(max_index);
        top_data[2 * i + 1] = max_val;
      }
    } else {
      // Produces max_ind per axis
      int index = (i / axis_dist) * axis_dist + i % axis_dist;
      top_data[index] = Dtype(max_index);
    }
  }
}

template <typename Dtype>
void ArgMaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int dim, axis_dist;
  if (has_axis_) {
    dim = bottom[0]->shape(axis_);
    // Distance between values of axis in blob
    axis_dist = bottom[0]->count(axis_) / dim;
  } else {
    dim = bottom[0]->count(1);
    axis_dist = 1;
  }
  int num = bottom[0]->count() / dim;
  if (top_k_ == 1) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ArgMaxForward<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
        num, axis_dist, dim, has_axis_, out_max_val_, bottom_data, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    std::vector<std::pair<Dtype, int> > bottom_data_vector(dim);
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < dim; ++j) {
        bottom_data_vector[j] = std::make_pair(
          bottom_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      for (int j = 0; j < top_k_; ++j) {
        if (out_max_val_) {
          if (has_axis_) {
            // Produces max_val per axis
            top_data[(i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist]
              = bottom_data_vector[j].first;
          } else {
            // Produces max_ind and max_val
            top_data[2 * i * top_k_ + j] = bottom_data_vector[j].second;
            top_data[2 * i * top_k_ + top_k_ + j] = bottom_data_vector[j].first;
          }
        } else {
          // Produces max_ind per axis
          top_data[(i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist]
            = bottom_data_vector[j].second;
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FORWARD(ArgMaxLayer);

}  // namespace caffe
