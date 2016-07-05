#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void LossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// LossLayers have a non-zero (1) loss by default.
	if (this->layer_param_.loss_weight_size() == 0) {
		this->layer_param_.add_loss_weight(Dtype(1));
	}
}

template<typename Dtype>
void LossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
			<< "The data and label should have the same number." << bottom[0]->shape_string() << " vs " << bottom[1]->shape_string();
	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
}

template<typename Dtype>
void LossLayer<Dtype>::Backward_Relevance_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom, const int layerindex,
		const relpropopts & ro, const std::vector<int> & classinds,
		const bool thenightstartshere) {

	if (true == thenightstartshere) {
		LOG(INFO) << "bottom.size() " << bottom.size();

		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		const Dtype* bottom_data = bottom[0]->cpu_data();

		LOG(INFO) << "softmaxlayer bottom[0]->count()"
				<< bottom[0]->count() << std::endl;
		memset(bottom_diff, 0, sizeof(Dtype) * bottom[0]->count());

		for (int i = 0; i < (int) classinds.size(); ++i) {
			int classindex = classinds[i];
			bottom_diff[classindex] = bottom_data[classindex];
			LOG(INFO) << "softmaxlayer " << bottom_diff[classindex]
					<< std::endl;
		}
	}

}

INSTANTIATE_CLASS (LossLayer);

}  // namespace caffe
