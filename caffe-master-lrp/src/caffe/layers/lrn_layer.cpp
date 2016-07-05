#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void LRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	size_ = this->layer_param_.lrn_param().local_size();
	CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
	pre_pad_ = (size_ - 1) / 2;
	alpha_ = this->layer_param_.lrn_param().alpha();
	beta_ = this->layer_param_.lrn_param().beta();
	k_ = this->layer_param_.lrn_param().k();
	if (this->layer_param_.lrn_param().norm_region()
			== LRNParameter_NormRegion_WITHIN_CHANNEL) {
		// Set up split_layer_ to use inputs in the numerator and denominator.
		split_top_vec_.clear();
		split_top_vec_.push_back(&product_input_);
		split_top_vec_.push_back(&square_input_);
		LayerParameter split_param;
		split_layer_.reset(new SplitLayer<Dtype>(split_param));
		split_layer_->SetUp(bottom, split_top_vec_);
		// Set up square_layer_ to square the inputs.
		square_bottom_vec_.clear();
		square_top_vec_.clear();
		square_bottom_vec_.push_back(&square_input_);
		square_top_vec_.push_back(&square_output_);
		LayerParameter square_param;
		square_param.mutable_power_param()->set_power(Dtype(2));
		square_layer_.reset(new PowerLayer<Dtype>(square_param));
		square_layer_->SetUp(square_bottom_vec_, square_top_vec_);
		// Set up pool_layer_ to sum over square neighborhoods of the input.
		pool_top_vec_.clear();
		pool_top_vec_.push_back(&pool_output_);
		LayerParameter pool_param;
		pool_param.mutable_pooling_param()->set_pool(
				PoolingParameter_PoolMethod_AVE);
		pool_param.mutable_pooling_param()->set_pad(pre_pad_);
		pool_param.mutable_pooling_param()->set_kernel_size(size_);
		pool_layer_.reset(new PoolingLayer<Dtype>(pool_param));
		pool_layer_->SetUp(square_top_vec_, pool_top_vec_);
		// Set up power_layer_ to compute (1 + alpha_/N^2 s)^-beta_, where s is
		// the sum of a squared neighborhood (the output of pool_layer_).
		power_top_vec_.clear();
		power_top_vec_.push_back(&power_output_);
		LayerParameter power_param;
		power_param.mutable_power_param()->set_power(-beta_);
		power_param.mutable_power_param()->set_scale(alpha_);
		power_param.mutable_power_param()->set_shift(Dtype(1));
		power_layer_.reset(new PowerLayer<Dtype>(power_param));
		power_layer_->SetUp(pool_top_vec_, power_top_vec_);
		// Set up a product_layer_ to compute outputs by multiplying inputs by the
		// inverse demoninator computed by the power layer.
		product_bottom_vec_.clear();
		product_bottom_vec_.push_back(&product_input_);
		product_bottom_vec_.push_back(&power_output_);
		LayerParameter product_param;
		EltwiseParameter* eltwise_param = product_param.mutable_eltwise_param();
		eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
		product_layer_.reset(new EltwiseLayer<Dtype>(product_param));
		product_layer_->SetUp(product_bottom_vec_, top);
	}
}

template<typename Dtype>
void LRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
			<< "corresponding to (num, channels, height, width)";
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	switch (this->layer_param_.lrn_param().norm_region()) {
	case LRNParameter_NormRegion_ACROSS_CHANNELS:
		top[0]->Reshape(num_, channels_, height_, width_);
		scale_.Reshape(num_, channels_, height_, width_);
		break;
	case LRNParameter_NormRegion_WITHIN_CHANNEL:
		split_layer_->Reshape(bottom, split_top_vec_);
		square_layer_->Reshape(square_bottom_vec_, square_top_vec_);
		pool_layer_->Reshape(square_top_vec_, pool_top_vec_);
		power_layer_->Reshape(pool_top_vec_, power_top_vec_);
		product_layer_->Reshape(product_bottom_vec_, top);
		break;
	}
}

template<typename Dtype>
void LRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	switch (this->layer_param_.lrn_param().norm_region()) {
	case LRNParameter_NormRegion_ACROSS_CHANNELS:
		LOG(INFO) << "LRNParameter_NormRegion_ACROSS_CHANNELS";
		CrossChannelForward_cpu(bottom, top);
		break;
	case LRNParameter_NormRegion_WITHIN_CHANNEL:
		LOG(INFO) << "LRNParameter_NormRegion_WITHIN_CHANNEL";
		WithinChannelForward(bottom, top);
		break;
	default:
		LOG(FATAL) << "Unknown normalization region.";
	}
}

template<typename Dtype>
void LRNLayer<Dtype>::CrossChannelForward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	bool bypass=false;
	if(true==bypass)
	{
		caffe_copy(bottom[0]->count(),bottom_data,top_data);
		return;
	}



	Dtype* scale_data = scale_.mutable_cpu_data();
	// start with the constant value
	for (int i = 0; i < scale_.count(); ++i) {
		scale_data[i] = k_;
	}

/*
for (int i = 0; i < scale_.count(); ++i) {
if(isnan(scale_data[i]))
{
LOG(FATAL) << "0 isnan at scale_data[i] ";
}
}
*/
	Blob < Dtype > padded_square(1, channels_ + size_ - 1, height_, width_);
	Dtype* padded_square_data = padded_square.mutable_cpu_data();
	caffe_set(padded_square.count(), Dtype(0), padded_square_data);
	Dtype alpha_over_size = alpha_ / size_;
	// go through the images
	for (int n = 0; n < num_; ++n) {
		// compute the padded square
		caffe_sqr(channels_ * height_ * width_,
				bottom_data + bottom[0]->offset(n),
				padded_square_data + padded_square.offset(0, pre_pad_));
		// Create the first channel scale
		for (int c = 0; c < size_; ++c) {
			caffe_axpy < Dtype
					> (height_ * width_, alpha_over_size, padded_square_data
							+ padded_square.offset(0, c), scale_data
							+ scale_.offset(n, 0));
		}

/*
for (int i = 0; i < scale_.count(); ++i) {
if(isnan(scale_data[i]))
{
LOG(FATAL) << "0A isnan at scale_data[i] ";
}
}
*/
		for (int c = 1; c < channels_; ++c) {
			// copy previous scale
			caffe_copy < Dtype
					> (height_ * width_, scale_data + scale_.offset(n, c - 1), scale_data
							+ scale_.offset(n, c));
/*
for (int i = 0; i < scale_.count(); ++i) {
if(isnan(scale_data[i]))
{
LOG(FATAL) << "0B isnan at scale_data[i] ";
}
}

Dtype tmpdata[height_ * width_];
			caffe_copy < Dtype
					> (height_ * width_, scale_data + scale_.offset(n, c ), tmpdata);
*/
			// add head
			caffe_axpy < Dtype
					> (height_ * width_, alpha_over_size, padded_square_data
							+ padded_square.offset(0, c + size_ - 1), scale_data
							+ scale_.offset(n, c));
/*
for (int i = 0; i < scale_.count(); ++i) {
if(isnan(scale_data[i]))
{
LOG(FATAL) << "0C isnan at scale_data[i] ";
}
}
*/

//LOG(ERROR) << ;

			// subtract tail
			caffe_axpy < Dtype
					> (height_ * width_, Dtype(-alpha_over_size), padded_square_data
							+ padded_square.offset(0, c - 1), scale_data
							+ scale_.offset(n, c));
/*
for (int i = 0; i < height_ * width_; ++i) {
if(isnan( padded_square_data[ padded_square.offset(0, c - 1)+i]  ))
{
LOG(FATAL) << "0C2 isnan at scale_data[i] " <<  c-1 << " " << i << " v= " << *(padded_square_data+ padded_square.offset(0, c - 1)+i);
}
}

for (int i = 0; i < height_ * width_; ++i) {
if(isnan( scale_data[ scale_.offset(n, c)+i]  ))
{

LOG(ERROR) <<"tmpdata[i] " << tmpdata[i];
LOG(ERROR) << " padded_square_data[ padded_square.offset(0, c - 1)+i] "<< padded_square_data[ padded_square.offset(0, c - 1)+i];

LOG(ERROR) << " -alpha_over_size*padded_square_data[ padded_square.offset(0, c - 1)+i] "<< -alpha_over_size*padded_square_data[ padded_square.offset(0, c - 1)+i];

LOG(ERROR) << "res " << -alpha_over_size*padded_square_data[ padded_square.offset(0, c - 1)+i] + tmpdata[i];

LOG(ERROR) << "num_ " << num_ << " channels_ " << channels_ <<" height_ * width_ "<< height_ * width_; 
LOG(FATAL) << "0C3 isnan at scale_data[i] nc offset i" <<  n << " " << c << " off " <<scale_.offset(n, c) << " "  << i << " v= " << scale_data[ scale_.offset(n, c)+i];
}
}


for (int i = 0; i < scale_.count(); ++i) {
if(isnan(scale_data[i]))
{
LOG(FATAL) << "0D isnan at scale_data[i] " << -alpha_over_size << " " << i << " v= " <<scale_data[i];
}
}
*/

		}
	}

	//for (int i = 0; i < scale_.count(); ++i) {
	//	scale_data[i] = std::max(k_,scale_data[i]);
	//}

/*
for (int i = 0; i < scale_.count(); ++i) {
if(isnan(scale_data[i]))
{
LOG(FATAL) << "1 isnan at scale_data[i] ";
}
}
*/
	// In the end, compute output
	caffe_powx < Dtype > (scale_.count(), scale_data, -beta_, top_data);

for (int i = 0; i < top[0]->count(); ++i) {
if(isnan(top_data[i]))
{
LOG(FATAL) << "2 isnan at top_data[i] ";
}
}

for (int i = 0; i < bottom[0]->count(); ++i) {
if(isnan(bottom_data[i]))
{
LOG(FATAL) << "3 isnan at bottom_data[i] ";
}
}

	caffe_mul < Dtype > (scale_.count(), top_data, bottom_data, top_data);

for (int i = 0; i < top[0]->count(); ++i) {
if(isnan(top_data[i]))
{
LOG(FATAL) << "4 isnan at top_data[i] ";
}
}


}

template<typename Dtype>
void LRNLayer<Dtype>::WithinChannelForward(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	split_layer_->Forward(bottom, split_top_vec_);
	square_layer_->Forward(square_bottom_vec_, square_top_vec_);
	pool_layer_->Forward(square_top_vec_, pool_top_vec_);
	power_layer_->Forward(pool_top_vec_, power_top_vec_);
	product_layer_->Forward(product_bottom_vec_, top);
}

template<typename Dtype>
void LRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	switch (this->layer_param_.lrn_param().norm_region()) {
	case LRNParameter_NormRegion_ACROSS_CHANNELS:
		CrossChannelBackward_cpu(top, propagate_down, bottom);
		break;
	case LRNParameter_NormRegion_WITHIN_CHANNEL:
		WithinChannelBackward(top, propagate_down, bottom);
		break;
	default:
		LOG(FATAL) << "Unknown normalization region.";
	}
}

template<typename Dtype>
void LRNLayer<Dtype>::CrossChannelBackward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* top_data = top[0]->cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* scale_data = scale_.cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Blob < Dtype > padded_ratio(1, channels_ + size_ - 1, height_, width_);
	Blob < Dtype > accum_ratio(1, 1, height_, width_);
	Dtype* padded_ratio_data = padded_ratio.mutable_cpu_data();
	Dtype* accum_ratio_data = accum_ratio.mutable_cpu_data();
	// We hack a little bit by using the diff() to store an additional result
	Dtype* accum_ratio_times_bottom = accum_ratio.mutable_cpu_diff();
	caffe_set(padded_ratio.count(), Dtype(0), padded_ratio_data);
	Dtype cache_ratio_value = 2. * alpha_ * beta_ / size_;

	caffe_powx < Dtype > (scale_.count(), scale_data, -beta_, bottom_diff);
	caffe_mul < Dtype > (scale_.count(), top_diff, bottom_diff, bottom_diff);

	// go through individual data
	int inverse_pre_pad = size_ - (size_ + 1) / 2;
	for (int n = 0; n < num_; ++n) {
		int block_offset = scale_.offset(n);
		// first, compute diff_i * y_i / s_i
		caffe_mul < Dtype
				> (channels_ * height_ * width_, top_diff + block_offset, top_data
						+ block_offset, padded_ratio_data
						+ padded_ratio.offset(0, inverse_pre_pad));
		caffe_div < Dtype
				> (channels_ * height_ * width_, padded_ratio_data
						+ padded_ratio.offset(0, inverse_pre_pad), scale_data
						+ block_offset, padded_ratio_data
						+ padded_ratio.offset(0, inverse_pre_pad));
		// Now, compute the accumulated ratios and the bottom diff
		caffe_set(accum_ratio.count(), Dtype(0), accum_ratio_data);
		for (int c = 0; c < size_ - 1; ++c) {
			caffe_axpy < Dtype
					> (height_ * width_, 1., padded_ratio_data
							+ padded_ratio.offset(0, c), accum_ratio_data);
		}
		for (int c = 0; c < channels_; ++c) {
			caffe_axpy < Dtype
					> (height_ * width_, 1., padded_ratio_data
							+ padded_ratio.offset(0, c + size_ - 1), accum_ratio_data);
			// compute bottom diff
			caffe_mul < Dtype
					> (height_ * width_, bottom_data + top[0]->offset(n, c), accum_ratio_data, accum_ratio_times_bottom);
			caffe_axpy < Dtype
					> (height_ * width_, -cache_ratio_value, accum_ratio_times_bottom, bottom_diff
							+ top[0]->offset(n, c));
			caffe_axpy < Dtype
					> (height_ * width_, -1., padded_ratio_data
							+ padded_ratio.offset(0, c), accum_ratio_data);
		}
	}
}

template<typename Dtype>
void LRNLayer<Dtype>::WithinChannelBackward(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		vector<bool> product_propagate_down(2, true);
		product_layer_->Backward(top, product_propagate_down,
				product_bottom_vec_);
		power_layer_->Backward(power_top_vec_, propagate_down, pool_top_vec_);
		pool_layer_->Backward(pool_top_vec_, propagate_down, square_top_vec_);
		square_layer_->Backward(square_top_vec_, propagate_down,
				square_bottom_vec_);
		split_layer_->Backward(split_top_vec_, propagate_down, bottom);
	}
}

template<typename Dtype>
void LRNLayer<Dtype>::Backward_Relevance_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom,
		const int layerindex, const relpropopts & ro,
		const std::vector<int> & classinds, const bool thenightstartshere) {

	switch (ro.lrn_backward_type) {
	case 0: 
	case -1:
		{
		//identity
		//if (propagate_down[0]) {
		//const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const int count = bottom[0]->count();
		for (int i = 0; i < count; ++i) {
			bottom_diff[i] = top_diff[i];
		}
		// }

		/*
		 switch(ro.codeexectype)
		 {
		 case 0:
		 {
		 //slowneasy
		 Backward_Relevance_cpu_taylor1_slowneasy(top,
		 propagate_down, bottom,
		 layerindex, ro );
		 }
		 break;
		 default:
		 {
		 LOG(FATAL) << "unknown value for ro.codeexectype " << ro.codeexectype << std::endl;
		 exit(1);
		 }
		 break;
		 } //	switch(ro.codeexectype)
		 */

	}
		break;

	case 1: {
		Dtype alpha_over_size = alpha_ / size_;

		const Dtype* top_diff = top[0]->cpu_diff();
		//const Dtype* top_data = top[0]->cpu_data();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* scale_data = scale_.cpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		//for (int i = 0; i < scale_.count(); ++i) {
		//	scale_data[i] = k_;
		//}
		Blob < Dtype > padded_square(1, channels_ + size_ , height_, width_);
		Dtype* padded_square_data = padded_square.mutable_cpu_data();

		Dtype* kplusalphatimesxcd2only = padded_square.mutable_cpu_diff();

		Blob < Dtype
				> padded_tri_one(1, channels_ + size_ , height_, width_);

		Dtype* padded_one = padded_tri_one.mutable_cpu_data();

		Dtype* padded_tri = padded_tri_one.mutable_cpu_diff();

		Blob < Dtype
				> padded_tri_pre(1, channels_ + size_ , height_, width_);
		Dtype* padded_tri_a = padded_tri_pre.mutable_cpu_data();
		Dtype* padded_tri_b = padded_tri_pre.mutable_cpu_diff();


		// go through the images
		for (int n = 0; n < num_; ++n) {

			caffe_set(padded_square.count(), Dtype(0), padded_square_data);

			// compute the padded square
			caffe_sqr(channels_ * height_ * width_,
					bottom_data + bottom[0]->offset(n),
					padded_square_data + padded_square.offset(0, pre_pad_));

			//caffe_set(padded_tri_one.count(), Dtype(0.), padded_one);
			// x_cd  ^2
			caffe_copy(padded_tri_one.count(), padded_square_data, padded_one);
			  //* alpha /size_
			caffe_scal(padded_tri_one.count(), alpha_over_size, padded_one);
			// k_ + this
			caffe_add_scalar(padded_tri_one.count(), k_, padded_one);
			caffe_powx(padded_tri_one.count(), padded_one, -beta_,
					kplusalphatimesxcd2only);

			caffe_set(padded_tri_one.count(), Dtype(0.), padded_one);

			//bottom data
			caffe_mul(channels_ * height_ * width_, kplusalphatimesxcd2only+ padded_square.offset(0, pre_pad_),
					bottom_data + bottom[0]->offset(n) , padded_one+ padded_square.offset(0, pre_pad_) );
			caffe_copy(padded_tri_one.count(), padded_one, kplusalphatimesxcd2only);
			//top diff
			caffe_mul(channels_ * height_ * width_, kplusalphatimesxcd2only+ padded_square.offset(0, pre_pad_),
					top_diff + top[0]->offset(n) , padded_one+ padded_square.offset(0, pre_pad_) );



			caffe_set(padded_tri_one.count(), Dtype(0.), padded_tri);
			caffe_set(padded_tri_one.count(), Dtype(0.), padded_tri_a);
			caffe_set(padded_tri_one.count(), Dtype(0.), padded_tri_b);


			//caffe_set(channels_ * height_ * width_, Dtype(1.), padded_tri +  padded_square.offset(0, pre_pad_) );


			//for(int k= pre_pad_; k < pre_pad +channels_  ;++k)
			//{
			caffe_powx( channels_ * height_ * width_ , scale_data + scale_.offset(n, 0), - beta_ - 1,
					padded_tri_a + padded_square.offset(0, pre_pad_ ) );
			//}
			caffe_scal(channels_ * height_ * width_ , -1 * Dtype(2.) * alpha_over_size * beta_,
					padded_tri_a +  padded_square.offset(0, pre_pad_));
			caffe_mul(channels_ * height_ * width_, padded_tri_a + padded_square.offset(0, pre_pad_),
					bottom_data + bottom[0]->offset(n) , padded_tri_b + padded_square.offset(0, pre_pad_) );
			caffe_mul(channels_ * height_ * width_, padded_tri_b + padded_square.offset(0, pre_pad_),
					top_diff + top[0]->offset(n) , padded_tri_a + padded_square.offset(0, pre_pad_) );

			for(int curc=pre_pad_;curc < pre_pad_ + channels_ ; ++curc)
			{
				for(int k= std::max(pre_pad_, curc- pre_pad_); k < std::min( curc + 1 + pre_pad_, channels_ + size_  ) ;++k)
				{
					if(k != curc )
					{
					caffe_cpu_axpby(height_ * width_, Dtype(1.), padded_tri_a + padded_square.offset(0, k),
					                            Dtype(1.), padded_tri + padded_square.offset(0, curc)) ;
					}
				}
			}

			caffe_mul(channels_ * height_ * width_, padded_tri + padded_square.offset(0, pre_pad_),
					padded_square_data + padded_square.offset(0, pre_pad_) , padded_tri_a + padded_square.offset(0, pre_pad_) );

			caffe_copy(channels_ * height_ * width_, padded_tri_a + padded_square.offset(0, pre_pad_) , bottom_diff + bottom[0]->offset(n));

			caffe_cpu_axpby(channels_ * height_ * width_, Dtype(1.), padded_one + padded_square.offset(0, pre_pad_),
			                            Dtype(1.), bottom_diff + bottom[0]->offset(n)) ;

			//scale_data + scale_.offset(n, c)

			// Create the first channel scale

			//for (int c = 0; c < size_; ++c) {
			// caffe_axpy<Dtype>(height_ * width_, alpha_over_size,
			//     padded_square_data + padded_square.offset(0, c),
			//    scale_data + scale_.offset(n, 0));
			//}

		} //			  for (int n = 0; n < num_; ++n) {

	}
		break;
	default: {
		LOG(FATAL) << "unknown value for ro.lrn_backward_type "
				<< ro.lrn_backward_type << std::endl;
		exit(1);
	}
		break;

	}

}

#ifdef CPU_ONLY
STUB_GPU(LRNLayer);
STUB_GPU_FORWARD(LRNLayer, CrossChannelForward);
STUB_GPU_BACKWARD(LRNLayer, CrossChannelBackward);
#endif

INSTANTIATE_CLASS (LRNLayer);
REGISTER_LAYER_CLASS (LRN);

}  // namespace caffe
