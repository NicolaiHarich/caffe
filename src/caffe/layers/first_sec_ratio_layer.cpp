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
    //nothing to do
}

template <typename Dtype>
void FirstSecRatioLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_LE(2, bottom[0]->count() / bottom[0]->num())
        << "number class must be more than or equal to 2.";

  top[0]->Reshape(bottom[0]->num(), 1,
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void FirstSecRatioLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  int channels = bottom[0]->channels();


  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++){

      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < channels; ++k) {
        bottom_data_vector.push_back(
          std::make_pair(bottom_data[i * dim + k * spatial_dim + j], k));
      }
      // Top-2
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + 2,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());

      top_data[i * dim + 0 + j] = 1 - bottom_data_vector[1].first / bottom_data_vector[0].first;  // 1 - sec/first

    }
  }

}

INSTANTIATE_CLASS(FirstSecRatioLayer);
REGISTER_LAYER_CLASS(FirstSecRatio);

}  // namespace caffe
