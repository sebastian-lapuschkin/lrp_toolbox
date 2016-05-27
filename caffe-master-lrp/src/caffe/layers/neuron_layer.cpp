#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NeuronLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}



template <typename Dtype>
void NeuronLayer<Dtype>::Backward_Relevance_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom,
	  const int layerindex, const relpropopts & ro , const std::vector<int> & classinds, const bool thenightstartshere)
{

	//LOG(WARN)<< "using generic void NeuronLayer<Dtype>::Backward_Relevance_cpu(...) other more optimal implementations might be available";
	switch(ro.relpropformulatype)
	{
		case 0:
		case 2:
		{
			  //if (propagate_down[0]) {
			   // const Dtype* bottom_data = bottom[0]->cpu_data();
			    const Dtype* top_diff = top[0]->cpu_diff();
			    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			    const int count = bottom[0]->count();
			    //Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
			    for (int i = 0; i < count; ++i) {
			      bottom_diff[i] = top_diff[i];
			    }
			 // }

		}
		break;
		default:
		{
			LOG(FATAL) << "unknown value for ro.relpropformulatype " << ro.relpropformulatype << std::endl;
			exit(1);
		}
		break;

	}

}



INSTANTIATE_CLASS(NeuronLayer);

}  // namespace caffe
