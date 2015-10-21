#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void VoidThresholdLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

//  softmax_bottom_vec_.clear();
//  softmax_bottom_vec_.push_back(bottom[0]);
//  softmax_top_vec_.clear();
//  softmax_top_vec_.push_back(&prob_);
//  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  void_label_ = this->layer_param_.void_threshold_param().void_label();

  thresh_ = this->layer_param_.void_threshold_param().threshold();

}

template <typename Dtype>
void VoidThresholdLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);

  // output
  top[0]->ReshapeLike(*bottom[0]);
  //top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
  //      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void VoidThresholdLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  Dtype* prob_data = bottom[0]->cpu_data();
  //Dtype* top_data = top[0]->mutable_cpu_data();

  // Softmax normalization
  //softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  //const Dtype* prob_data = prob_.cpu_data();

//  int num = prob_.num();
//  int dim = prob_.count() / num;
//  int spatial_dim = prob_.height() * prob_.width();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  int channels = bottom[0]->channels();

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {

        // find highest prob
        std::vector<std::pair<Dtype, int> > prob_data_vector;
        for (int k = 0; k < channels; ++k) {
          if(k = void_label_){
              continue;
          }
          prob_data_vector.push_back(
            std::make_pair(prob_data[i * dim + k * spatial_dim + j], k));
        }
        std::partial_sort(
            prob_data_vector.begin(), prob_data_vector.begin() + 1,
            prob_data_vector.end(), std::greater<std::pair<Dtype, int> >());

        // check if max. prob is greater than threshold
        if (prob_data_vector[0].first < thresh_) {
            //set void_label-probabilty to 1
            prob_data[i * dim + void_label_ * spatial_dim + j] = 1;
            //const Dtype best_score = bottom_data[i * dim + prob_data_vector[0].second * spatial_dim + j];
            //top_data[i * dim + void_label_ * spatial_dim + j] = best_score + 1;
        }
    }
  }

  top[0]->ShareData(prob_data);
}

//template <typename Dtype>
//void VoidThresholdLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
//    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

//    vector<bool> propagate_down(2, true);
//    softmax_layer_->Backward(softmax_top_vec_, propagate_down, softmax_bottom_vec_);

//}

INSTANTIATE_CLASS(VoidThresholdLayer);
REGISTER_LAYER_CLASS(VoidThreshold);

}  // namespace caffe
