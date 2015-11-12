#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLoss2GtLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  //TODO has_ignore_label_2
  has_ignore_label_ =  this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
}

template <typename Dtype>
void SoftmaxWithLoss2GtLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithLoss2GtLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* label2 = bottom[2]->cpu_data();

  int dim = prob_.count() / outer_num_;
  int count = 0;
  int count2 = 0;
  Dtype loss = 0;
  Dtype loss2 = 0;

  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      const int label2_value = static_cast<int>(label2[i * inner_num_ + j]);

      //update first loss
      if ( !has_ignore_label_ || (has_ignore_label_ && label_value != ignore_label_) ) {

          DCHECK_GE(label_value, 0);
          DCHECK_LT(label_value, prob_.shape(softmax_axis_));
          loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                               Dtype(FLT_MIN)));
          ++count;
      }
      //update second loss
      if ( !has_ignore_label_ || (has_ignore_label_ && label2_value != ignore_label_) ) {
          DCHECK_GE(label2_value, 0);
          DCHECK_LT(label2_value, prob_.shape(softmax_axis_));
          loss2 -= log(std::max(prob_data[i * dim + label2_value * inner_num_ + j],
                               Dtype(FLT_MIN)));
          ++count2;
      }

    }
  }
  if (normalize_) {
    loss /= count;
    loss2 /= count2;
  } else {
    loss /= outer_num_;
    loss2 /= outer_num_;
  }
  top[0]->mutable_cpu_data()[0] = loss;
  top[1]->mutable_cpu_data()[0] = loss2;

  if (top.size() == 3) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLoss2GtLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* label2 = bottom[2]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    int count2 = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        const int label2_value = static_cast<int>(label2[i * inner_num_ + j]);

        if ( has_ignore_label_ && label_value == ignore_label_ && label2_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {

            if ( !has_ignore_label_ || (has_ignore_label_ && label_value != ignore_label_) ) {
                bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
                ++count;
            }
            if ( !has_ignore_label_ || (has_ignore_label_ && label2_value != ignore_label_) ) {
                bottom_diff[i * dim + label2_value * inner_num_ + j] -= 1;
                ++count2;
            }

        }

      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    const Dtype loss2_weight = top[1]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(prob_.count(), (loss_weight/count + loss2_weight/count2) / 2, bottom_diff);
    } else {
      caffe_scal(prob_.count(), (loss_weight + loss2_weight) / (2*outer_num_), bottom_diff);
    }
  }
}

//TODO
//#ifdef CPU_ONLY
//STUB_GPU(SoftmaxWithLoss2GtLayer);
//#endif

INSTANTIATE_CLASS(SoftmaxWithLoss2GtLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss2Gt);

}  // namespace caffe
