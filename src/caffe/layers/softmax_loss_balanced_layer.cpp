#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossBalancedLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
}

template <typename Dtype>
void SoftmaxWithLossBalancedLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithLossBalancedLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  Dtype loss = 0;
  int count = 0;
  
  std::vector<Dtype> accum_loss(dim);
  std::vector<Dtype> class_scale_vec;

  std::fill (class_scale_vec.begin(),class_scale_vec.end(),0); //reset
  int num_classes_in_batch = 0;

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      const int label_value = static_cast<int>(label[i * spatial_dim + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.channels());
      accum_loss[label_value] -= log(std::max(prob_data[i * dim + label_value * spatial_dim + j],
                           Dtype(FLT_MIN)));
      ++count;

      if(class_scale_vec.size()<=label_value){
          //int size_prev = class_scale_vec.size();
          class_scale_vec.resize(label_value+1);
          //std::fill (class_scale_vec.begin()+size_prev,class_scale_vec.end(),0); //fill with 0
      }
      ++class_scale_vec[label_value];
    }
  }

  //normalize loss based on class-frequency
  for (int i = 0; i < class_scale_vec.size(); ++i) {
    if( class_scale_vec[i] > 0 ){
        class_scale_vec[i] = 1/class_scale_vec[i];    //invert
        loss += accum_loss[i] * class_scale_vec[i];	//scale by class frequency
        ++num_classes_in_batch;
	} else {
        class_scale_vec[i] = 0;
	}
  }
  
  if (normalize_) {
    if( num_classes_in_batch> 0 ){
        //LOG(INFO) << "[SOFTMAX_BALANCED] num_classes_in_batch: " << num_classes_in_batch;
        top[0]->mutable_cpu_data()[0] = loss / num_classes_in_batch;
    }else{
        LOG(INFO) << "[SOFTMAX_BALANCED] num_classes_in_batch is zero! This should not happen! ";
        top[0]->mutable_cpu_data()[0] = 0;
    }
  } else {
    top[0]->mutable_cpu_data()[0] = loss / num;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }

}

template <typename Dtype>
void SoftmaxWithLossBalancedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    int spatial_dim = prob_.height() * prob_.width();
    int count = 0;

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        const int label_value = static_cast<int>(label[i * spatial_dim + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->channels(); ++c) {
            bottom_diff[i * dim + c * spatial_dim + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * spatial_dim + j] -= 1;  //loss_weight * class_scale_vec[label_value];  //scale with class frequency
          ++count;
        }
      }
    }

    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
	if (normalize_) {
      //if( num_classes_in_batch_>0 ) caffe_scal(prob_.count(), loss_weight / num_classes_in_batch_, bottom_diff);
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / num, bottom_diff);
    }
  }
  //LOG(INFO) << "[SOFTMAX_BALANCED] run: " << ct_run_;
}

INSTANTIATE_CLASS(SoftmaxWithLossBalancedLayer);
REGISTER_LAYER_CLASS(SOFTMAX_LOSS_BALANCED, SoftmaxWithLossBalancedLayer);

}  // namespace caffe
