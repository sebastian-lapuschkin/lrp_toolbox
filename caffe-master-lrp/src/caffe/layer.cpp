#include <boost/thread.hpp>
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void Layer<Dtype>::InitMutex() {
  forward_mutex_.reset(new boost::mutex());
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Backward_Relevance_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom, 
  	  const int layerindex, 
	  const relpropopts & ro , 
	  const std::vector<int> & classinds, const bool thenightstartshere  )
{
	LOG(FATAL) << "not implemented: void Backward_Relevance_cpu (...) "  << std::endl;
	exit(1);
}

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
