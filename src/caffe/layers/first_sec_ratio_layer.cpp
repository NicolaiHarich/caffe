#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void FirstSecRatioLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const FirstSecRatioParameter& first_sec_ratio_param = this->layer_param_.first_sec_ratio_param();

  has_axis_ = first_sec_ratio_param.has_axis();

  if (has_axis_) {
    axis_ = bottom[0]->CanonicalAxisIndex(first_sec_ratio_param.axis());
    CHECK_GE(axis_, 0) << "axis must not be less than 0.";
    CHECK_LE(axis_, bottom[0]->num_axes()) <<
      "axis must be less than or equal to the number of axis.";
    CHECK_LE(2, bottom[0]->shape(axis_))
      << "top_k must be less than or equal to the dimension of the axis.";
  } else {
    CHECK_LE(2, bottom[0]->count(1))
      << "top_k must be less than or equal to"
        " the dimension of the flattened bottom blob per instance.";
  }
}

template <typename Dtype>
void FirstSecRatioLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  std::vector<int> shape(bottom[0]->num_axes(), 1);
  if (has_axis_) {
    // Produces ratio per axis
    shape = bottom[0]->shape();
    shape[axis_] = 2;
  } else {
    shape[0] = bottom[0]->shape(0);
    // Produces ratio
    shape[2] = 2;
  }
  top[0]->Reshape(shape);
}

template <typename Dtype>
void FirstSecRatioLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
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
  std::vector<std::pair<Dtype, int> > bottom_data_vector(dim);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      bottom_data_vector[j] = std::make_pair(
        bottom_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
    }
    std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + 2,
        bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());

    top_data[2 * i * 2 + 2 + 0] = 1 - bottom_data_vector[1].first / bottom_data_vector[0].first;  // 1 - sec/first

  }
}

INSTANTIATE_CLASS(FirstSecRatioLayer);
REGISTER_LAYER_CLASS(FirstSecRatio);

}  // namespace caffe
