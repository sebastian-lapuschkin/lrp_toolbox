#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// Configure the kernel size, padding, stride, and inputs.
	ConvolutionParameter conv_param = this->layer_param_.convolution_param();
	force_nd_im2col_ = conv_param.force_nd_im2col();
	channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
	const int first_spatial_axis = channel_axis_ + 1;
	const int num_axes = bottom[0]->num_axes();
	num_spatial_axes_ = num_axes - first_spatial_axis;
	CHECK_GE(num_spatial_axes_, 0);
	vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
	vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
	// Setup filter kernel dimensions (kernel_shape_).
	kernel_shape_.Reshape(spatial_dim_blob_shape);
	int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
	if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
		CHECK_EQ(num_spatial_axes_, 2)
				<< "kernel_h & kernel_w can only be used for 2D convolution.";
		CHECK_EQ(0, conv_param.kernel_size_size())
				<< "Either kernel_size or kernel_h/w should be specified; not both.";
		kernel_shape_data[0] = conv_param.kernel_h();
		kernel_shape_data[1] = conv_param.kernel_w();
	} else {
		const int num_kernel_dims = conv_param.kernel_size_size();
		CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
				<< "kernel_size must be specified once, or once per spatial dimension "
				<< "(kernel_size specified " << num_kernel_dims << " times; "
				<< num_spatial_axes_ << " spatial dims);";
		for (int i = 0; i < num_spatial_axes_; ++i) {
			kernel_shape_data[i] = conv_param.kernel_size(
					(num_kernel_dims == 1) ? 0 : i);
		}
	}
	for (int i = 0; i < num_spatial_axes_; ++i) {
		CHECK_GT(kernel_shape_data[i], 0)
				<< "Filter dimensions must be nonzero.";
	}
	// Setup stride dimensions (stride_).
	stride_.Reshape(spatial_dim_blob_shape);
	int* stride_data = stride_.mutable_cpu_data();
	if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
		CHECK_EQ(num_spatial_axes_, 2)
				<< "stride_h & stride_w can only be used for 2D convolution.";
		CHECK_EQ(0, conv_param.stride_size())
				<< "Either stride or stride_h/w should be specified; not both.";
		stride_data[0] = conv_param.stride_h();
		stride_data[1] = conv_param.stride_w();
	} else {
		const int num_stride_dims = conv_param.stride_size();
		CHECK(
				num_stride_dims == 0 || num_stride_dims == 1
						|| num_stride_dims == num_spatial_axes_)
				<< "stride must be specified once, or once per spatial dimension "
				<< "(stride specified " << num_stride_dims << " times; "
				<< num_spatial_axes_ << " spatial dims);";
		const int kDefaultStride = 1;
		for (int i = 0; i < num_spatial_axes_; ++i) {
			stride_data[i] =
					(num_stride_dims == 0) ?
							kDefaultStride :
							conv_param.stride((num_stride_dims == 1) ? 0 : i);
			CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
		}
	}
	// Setup pad dimensions (pad_).
	pad_.Reshape(spatial_dim_blob_shape);
	int* pad_data = pad_.mutable_cpu_data();
	if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
		CHECK_EQ(num_spatial_axes_, 2)
				<< "pad_h & pad_w can only be used for 2D convolution.";
		CHECK_EQ(0, conv_param.pad_size())
				<< "Either pad or pad_h/w should be specified; not both.";
		pad_data[0] = conv_param.pad_h();
		pad_data[1] = conv_param.pad_w();
	} else {
		const int num_pad_dims = conv_param.pad_size();
		CHECK(
				num_pad_dims == 0 || num_pad_dims == 1
						|| num_pad_dims == num_spatial_axes_)
				<< "pad must be specified once, or once per spatial dimension "
				<< "(pad specified " << num_pad_dims << " times; "
				<< num_spatial_axes_ << " spatial dims);";
		const int kDefaultPad = 0;
		for (int i = 0; i < num_spatial_axes_; ++i) {
			pad_data[i] =
					(num_pad_dims == 0) ?
							kDefaultPad :
							conv_param.pad((num_pad_dims == 1) ? 0 : i);
		}
	}
	// Special case: im2col is the identity for 1x1 convolution with stride 1
	// and no padding, so flag for skipping the buffer and transformation.
	is_1x1_ = true;
	for (int i = 0; i < num_spatial_axes_; ++i) {
		is_1x1_ &= kernel_shape_data[i] == 1 && stride_data[i] == 1
				&& pad_data[i] == 0;
		if (!is_1x1_) {
			break;
		}
	}
	// Configure output channels and groups.
	channels_ = bottom[0]->shape(channel_axis_);
	num_output_ = this->layer_param_.convolution_param().num_output();
	CHECK_GT(num_output_, 0);
	group_ = this->layer_param_.convolution_param().group();
	CHECK_EQ(channels_ % group_, 0);
	CHECK_EQ(num_output_ % group_, 0)
			<< "Number of output should be multiples of group.";
	if (reverse_dimensions()) {
		conv_out_channels_ = channels_;
		conv_in_channels_ = num_output_;
	} else {
		conv_out_channels_ = num_output_;
		conv_in_channels_ = channels_;
	}
	// Handle the parameters: weights and biases.
	// - blobs_[0] holds the filter weights
	// - blobs_[1] holds the biases (optional)
	vector<int> weight_shape(2);
	weight_shape[0] = conv_out_channels_;
	weight_shape[1] = conv_in_channels_ / group_;
	for (int i = 0; i < num_spatial_axes_; ++i) {
		weight_shape.push_back(kernel_shape_data[i]);
	}
	bias_term_ = this->layer_param_.convolution_param().bias_term();
	vector<int> bias_shape(bias_term_, num_output_);
	if (this->blobs_.size() > 0) {
		CHECK_EQ(1 + bias_term_, this->blobs_.size())
				<< "Incorrect number of weight blobs.";
		if (weight_shape != this->blobs_[0]->shape()) {
			Blob < Dtype > weight_shaped_blob(weight_shape);
			LOG(FATAL) << "Incorrect weight shape: expected shape "
					<< weight_shaped_blob.shape_string()
					<< "; instead, shape was "
					<< this->blobs_[0]->shape_string();
		}
		if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
			Blob < Dtype > bias_shaped_blob(bias_shape);
			LOG(FATAL) << "Incorrect bias shape: expected shape "
					<< bias_shaped_blob.shape_string()
					<< "; instead, shape was "
					<< this->blobs_[1]->shape_string();
		}
		LOG(INFO) << "Skipping parameter initialization";
	} else {
		if (bias_term_) {
			this->blobs_.resize(2);
		} else {
			this->blobs_.resize(1);
		}
		// Initialize and fill the weights:
		// output channels x input channels per-group x kernel height x kernel width
		this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
		shared_ptr < Filler<Dtype>
				> weight_filler(
						GetFiller < Dtype
								> (this->layer_param_.convolution_param().weight_filler()));
		weight_filler->Fill(this->blobs_[0].get());
		// If necessary, initialize and fill the biases.
		if (bias_term_) {
			this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
			shared_ptr < Filler<Dtype>
					> bias_filler(
							GetFiller < Dtype
									> (this->layer_param_.convolution_param().bias_filler()));
			bias_filler->Fill(this->blobs_[1].get());
		}
	}
	kernel_dim_ = this->blobs_[0]->count(1);
	weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
	// Propagate gradients to the parameters (as directed by backward pass).
	this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const int first_spatial_axis = channel_axis_ + 1;
	CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
			<< "bottom num_axes may not change.";
	num_ = bottom[0]->count(0, channel_axis_);
	CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
			<< "Input size incompatible with convolution kernel.";
	// TODO: generalize to handle inputs of different shapes.
	for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
		CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
				<< "All inputs must have the same shape.";
	}
	// Shape the tops.
	bottom_shape_ = &bottom[0]->shape();
	compute_output_shape();
	vector<int> top_shape(bottom[0]->shape().begin(),
			bottom[0]->shape().begin() + channel_axis_);
	top_shape.push_back(num_output_);
	for (int i = 0; i < num_spatial_axes_; ++i) {
		top_shape.push_back(output_shape_[i]);
	}
	for (int top_id = 0; top_id < top.size(); ++top_id) {
		top[top_id]->Reshape(top_shape);
	}
	if (reverse_dimensions()) {
		conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
	} else {
		conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
	}
	col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
	output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
	// Setup input dimensions (conv_input_shape_).
	vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
	conv_input_shape_.Reshape(bottom_dim_blob_shape);
	int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
	for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
		if (reverse_dimensions()) {
			conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
		} else {
			conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
		}
	}
	// The im2col result buffer will only hold one image at a time to avoid
	// overly large memory usage. In the special case of 1x1 convolution
	// it goes lazily unused to save memory.
	col_buffer_shape_.clear();
	col_buffer_shape_.push_back(kernel_dim_ * group_);
	for (int i = 0; i < num_spatial_axes_; ++i) {
		if (reverse_dimensions()) {
			col_buffer_shape_.push_back(input_shape(i + 1));
		} else {
			col_buffer_shape_.push_back(output_shape_[i]);
		}
	}
	col_buffer_.Reshape(col_buffer_shape_);
	bottom_dim_ = bottom[0]->count(channel_axis_);
	top_dim_ = top[0]->count(channel_axis_);
	num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
	num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
	// Set up the all ones "bias multiplier" for adding biases by BLAS
	out_spatial_dim_ = top[0]->count(first_spatial_axis);
	if (bias_term_) {
		vector<int> bias_multiplier_shape(1, out_spatial_dim_);
		bias_multiplier_.Reshape(bias_multiplier_shape);
		caffe_set(bias_multiplier_.count(), Dtype(1),
				bias_multiplier_.mutable_cpu_data());
	}
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
		const Dtype* weights, Dtype* output, bool skip_im2col) {
	const Dtype* col_buff = input;
	if (!is_1x1_) {
		if (!skip_im2col) {
			conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
		}
		col_buff = col_buffer_.cpu_data();
	}
	for (int g = 0; g < group_; ++g) {
		caffe_cpu_gemm < Dtype
				> (CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_, (Dtype) 1., weights
						+ weight_offset_ * g, col_buff + col_offset_ * g, (Dtype) 0., output
						+ output_offset_ * g);
	}
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
		const Dtype* bias) {
	caffe_cpu_gemm < Dtype
			> (CblasNoTrans, CblasNoTrans, num_output_, out_spatial_dim_, 1, (Dtype) 1., bias, bias_multiplier_.cpu_data(), (Dtype) 1., output);
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
		const Dtype* weights, Dtype* input) {
	Dtype* col_buff = col_buffer_.mutable_cpu_data();
	if (is_1x1_) {
		col_buff = input;
	}
	for (int g = 0; g < group_; ++g) {
		caffe_cpu_gemm < Dtype
				> (CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_
						/ group_, (Dtype) 1., weights + weight_offset_ * g, output
						+ output_offset_ * g, (Dtype) 0., col_buff
						+ col_offset_ * g);
	}
	if (!is_1x1_) {
		conv_col2im_cpu(col_buff, input);
	}
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::alphabeta(
		const Dtype* upperlayerrelevances, //ex output,
		const Dtype* weights, const Dtype* input,
		Dtype * new_lowerlayerrelevances, const relpropopts & ro,
		bool skip_im2col) {
	int K = kernel_dim_; //xi gets summed along this with wij
	int Iin = conv_out_spatial_dim_;
	int Rout = conv_out_channels_ / group_;

	float beta =ro.alphabeta_beta;
	float alpha = 1.0 + beta;
	// inputs col_buff are K x Iin
	//weights are Rout x K
	// upper relevances are Rout x Iin

	const Dtype* col_buff = input;
	if (!is_1x1_) {
		if (!skip_im2col) {
			//memset(col_buffer_.mutable_cpu_data(), 0,
			//		sizeof(Dtype) * col_buffer_.count());
			conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
		}
		col_buff = col_buffer_.cpu_data();
	}

	Dtype* col_buff_new = col_buffer_.mutable_cpu_diff();
	if (is_1x1_) {
		col_buff_new = new_lowerlayerrelevances;
	}
	//memset(col_buff_new, 0,
	//		sizeof(Dtype) * conv_out_spatial_dim_ * kernel_dim_);
	memset(col_buff_new, 0,
			sizeof(Dtype) * col_buffer_.count());

	Blob < Dtype > pos_sums(1, 1, Rout, Iin);
	Blob < Dtype > neg_sums(1, 1, Rout, Iin);
	Dtype* pos_sums_data = pos_sums.mutable_cpu_data();
	Dtype* neg_sums_data = neg_sums.mutable_cpu_data();

	for (int g = 0; g < group_; ++g) {
		/*
		 caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
		 conv_out_spatial_dim_, conv_out_channels_ / group_,
		 (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
		 (Dtype)0., col_buff + col_offset_ * g);
		 */
		memset(pos_sums_data, 0, sizeof(Dtype) * Rout * Iin);
		memset(neg_sums_data, 0, sizeof(Dtype) * Rout * Iin);

		if (beta > 0) {
			if (this->bias_term_) {
				switch (ro.biastreatmenttype) {
				case 0: //ignore bias
				{
					for (long iinind = 0; iinind < Iin; ++iinind) {
						for (long routind = 0; routind < Rout; ++routind) {
							//pos_sums_data[routind,iinind]

							// x is col_buff[kernelind,iinind]

							//w is weights[routind,kernelind]

							for (long kernelind = 0; kernelind < K; ++kernelind) {
								//if or max/min ?
								pos_sums_data[routind * Iin + iinind] += std::max(
										Dtype(0.),
										col_buff[col_offset_ * g + kernelind * Iin
												+ iinind]
												* weights[weight_offset_ * g
														+ routind * K + kernelind]);
								neg_sums_data[routind * Iin + iinind] += std::min(
										Dtype(0.),
										col_buff[col_offset_ * g + kernelind * Iin
												+ iinind]
												* weights[weight_offset_ * g
														+ routind * K + kernelind]);

							}
							
								Dtype bterm = this->blobs_[1]->cpu_data()[routind]
										* bias_multiplier_.cpu_data()[iinind];
								pos_sums_data[routind * Iin + iinind] += std::max(
										Dtype(0.), bterm);
								neg_sums_data[routind * Iin + iinind] += std::min(
										Dtype(0.), bterm);
							

						} //for (long routind = 0; routind < Rout; ++routind) {
					} //for (long iinind = 0; iinind < Iin; ++iinind) {
				} // case 0:
				break;
				
				case 1: //distrib bias
				{
					for (long iinind = 0; iinind < Iin; ++iinind) {
						for (long routind = 0; routind < Rout; ++routind) {
							//pos_sums_data[routind,iinind]

							// x is col_buff[kernelind,iinind]

							//w is weights[routind,kernelind]

							Dtype bterm = this->blobs_[1]->cpu_data()[routind]
									* bias_multiplier_.cpu_data()[iinind];
							
							for (long kernelind = 0; kernelind < K; ++kernelind) {
								//if or max/min ?
								pos_sums_data[routind * Iin + iinind] += std::max(
										Dtype(0.),
										col_buff[col_offset_ * g + kernelind * Iin
												+ iinind]
												* weights[weight_offset_ * g
														+ routind * K + kernelind] + bterm / K);
								neg_sums_data[routind * Iin + iinind] += std::min(
										Dtype(0.),
										col_buff[col_offset_ * g + kernelind * Iin
												+ iinind]
												* weights[weight_offset_ * g
														+ routind * K + kernelind]  + bterm / K);

							}
							

						} //for (long routind = 0; routind < Rout; ++routind) {
					} //for (long iinind = 0; iinind < Iin; ++iinind) {
				} // case 1:
				break;
				
				default: {
					LOG(FATAL) << "uknown value for: ro.biastreatmenttype "
							<< ro.biastreatmenttype;
				}
					break;
				}//				switch (ro.biastreatmenttype) {

			}
			else //if (this->bias_term_) {
			{	
			for (long iinind = 0; iinind < Iin; ++iinind) {
				for (long routind = 0; routind < Rout; ++routind) {
					//pos_sums_data[routind,iinind]

					// x is col_buff[kernelind,iinind]

					//w is weights[routind,kernelind]

					for (long kernelind = 0; kernelind < K; ++kernelind) {
						//if or max/min ?
						pos_sums_data[routind * Iin + iinind] += std::max(
								Dtype(0.),
								col_buff[col_offset_ * g + kernelind * Iin
										+ iinind]
										* weights[weight_offset_ * g
												+ routind * K + kernelind]);
						neg_sums_data[routind * Iin + iinind] += std::min(
								Dtype(0.),
								col_buff[col_offset_ * g + kernelind * Iin
										+ iinind]
										* weights[weight_offset_ * g
												+ routind * K + kernelind]);

					}

				} //for (long routind = 0; routind < Rout; ++routind) {
			} //for (long iinind = 0; iinind < Iin; ++iinind) {
			} // else of if (this->bias_term_) {
		} else // if(beta>0)
		{
			
			if (this->bias_term_) {
				switch (ro.biastreatmenttype) {
				case 0: //ignore bias
				{
					for (long iinind = 0; iinind < Iin; ++iinind) {
						for (long routind = 0; routind < Rout; ++routind) {
							//pos_sums_data[routind,iinind]

							// x is col_buff[kernelind,iinind]

							//w is weights[routind,kernelind]

							for (long kernelind = 0; kernelind < K; ++kernelind) {
								//if or max/min ?
								pos_sums_data[routind * Iin + iinind] += std::max(
										Dtype(0.),
										col_buff[col_offset_ * g + kernelind * Iin
												+ iinind]
												* weights[weight_offset_ * g
														+ routind * K + kernelind]);
								//neg_sums_data[ routind*Iin+iinind]+=std::min(Dtype(0.), col_buff[col_offset_ * g + kernelind*Iin+iinind]*weights[weight_offset_ * g + routind*K+kernelind] );

							}

							
								Dtype bterm = this->blobs_[1]->cpu_data()[routind]
										* bias_multiplier_.cpu_data()[iinind];
								pos_sums_data[routind * Iin + iinind] += std::max(
										Dtype(0.), bterm);
								neg_sums_data[routind * Iin + iinind] += std::min(
										Dtype(0.), bterm);
							

						} //for (long routind = 0; routind < Rout; ++routind) {
					} //for (long iinind = 0; iinind < Iin; ++iinind) {
				}//case 0:
				break;
				case 1: //distrib bias
				{
					for (long iinind = 0; iinind < Iin; ++iinind) {
						for (long routind = 0; routind < Rout; ++routind) {
							//pos_sums_data[routind,iinind]

							// x is col_buff[kernelind,iinind]

							//w is weights[routind,kernelind]
							Dtype bterm = this->blobs_[1]->cpu_data()[routind]
									* bias_multiplier_.cpu_data()[iinind];

							for (long kernelind = 0; kernelind < K; ++kernelind) {
								//if or max/min ?
								pos_sums_data[routind * Iin + iinind] += std::max(
										Dtype(0.),
										col_buff[col_offset_ * g + kernelind * Iin
												+ iinind]
												* weights[weight_offset_ * g
														+ routind * K + kernelind] + bterm / K );
								//neg_sums_data[ routind*Iin+iinind]+=std::min(Dtype(0.), col_buff[col_offset_ * g + kernelind*Iin+iinind]*weights[weight_offset_ * g + routind*K+kernelind] );

							}

						} //for (long routind = 0; routind < Rout; ++routind) {
					} //for (long iinind = 0; iinind < Iin; ++iinind) {
				}//case 1:
				break;
				default: {
					LOG(FATAL) << "uknown value for: ro.biastreatmenttype "
							<< ro.biastreatmenttype;
				}
					break;
				}//				switch (ro.biastreatmenttype) {

			}
			else //if (this->bias_term_) {
			{
				for (long iinind = 0; iinind < Iin; ++iinind) {
					for (long routind = 0; routind < Rout; ++routind) {
						//pos_sums_data[routind,iinind]

						// x is col_buff[kernelind,iinind]

						//w is weights[routind,kernelind]

						for (long kernelind = 0; kernelind < K; ++kernelind) {
							//if or max/min ?
							pos_sums_data[routind * Iin + iinind] += std::max(
									Dtype(0.),
									col_buff[col_offset_ * g + kernelind * Iin
											+ iinind]
											* weights[weight_offset_ * g
													+ routind * K + kernelind]);
							//neg_sums_data[ routind*Iin+iinind]+=std::min(Dtype(0.), col_buff[col_offset_ * g + kernelind*Iin+iinind]*weights[weight_offset_ * g + routind*K+kernelind] );

						}


					} //for (long routind = 0; routind < Rout; ++routind) {
				} //for (long iinind = 0; iinind < Iin; ++iinind) {
			} // else of //if (this->bias_term_) {
		} // else of  if(beta>0)

		//if(beta>0)
		//{	  

		if (beta > 0) {
			if (this->bias_term_) {
				switch (ro.biastreatmenttype) {
				case 0: //ignore bias
				{
					for (long iinind = 0; iinind < Iin; ++iinind) {
						for (long routind = 0; routind < Rout; ++routind) {
							Dtype z1 = 0;
							if (pos_sums_data[routind * Iin + iinind] > 0) {
								z1 = upperlayerrelevances[output_offset_ * g
										+ routind * Iin + iinind]
										/ pos_sums_data[routind * Iin + iinind];
							}
							Dtype z2 = 0;
							if (neg_sums_data[routind * Iin + iinind] < 0) {
								z2 = upperlayerrelevances[output_offset_ * g
										+ routind * Iin + iinind]
										/ neg_sums_data[routind * Iin + iinind];
							}
							for (long kernelind = 0; kernelind < K;
									++kernelind) {

								col_buff_new[col_offset_ * g + kernelind * Iin
										+ iinind] +=
										alpha
												* std::max(Dtype(0.),
														col_buff[col_offset_ * g
																+ kernelind
																		* Iin
																+ iinind]
																* weights[weight_offset_
																		* g
																		+ routind
																				* K
																		+ kernelind])
												* z1
												- beta
														* std::min(Dtype(0.),
																col_buff[col_offset_
																		* g
																		+ kernelind
																				* Iin
																		+ iinind]
																		* weights[weight_offset_
																				* g
																				+ routind
																						* K
																				+ kernelind])
														* z2;
							}
						} //for (long routind = 0; routind < Rout; ++routind) {
					} //for (long iinind = 0; iinind < Iin; ++iinind) {
				} //case 0:
					break;

				case 1: //dist bias as 1/n
				{
					for (long iinind = 0; iinind < Iin; ++iinind) {
						for (long routind = 0; routind < Rout; ++routind) {
							Dtype bterm = this->blobs_[1]->cpu_data()[routind]
									* bias_multiplier_.cpu_data()[iinind];

							Dtype z1 = 0;
							if (pos_sums_data[routind * Iin + iinind] > 0) {
								z1 = upperlayerrelevances[output_offset_ * g
										+ routind * Iin + iinind]
										/ pos_sums_data[routind * Iin + iinind];
							}
							Dtype z2 = 0;
							if (neg_sums_data[routind * Iin + iinind] < 0) {
								z2 = upperlayerrelevances[output_offset_ * g
										+ routind * Iin + iinind]
										/ neg_sums_data[routind * Iin + iinind];
							}
							for (long kernelind = 0; kernelind < K;
									++kernelind) {

								col_buff_new[col_offset_ * g + kernelind * Iin
										+ iinind] +=
										alpha
												* std::max(Dtype(0.),
														col_buff[col_offset_ * g
																+ kernelind
																		* Iin
																+ iinind]
																* weights[weight_offset_
																		* g
																		+ routind
																				* K
																		+ kernelind]
																+ bterm / K)
												* z1
												- beta
														* std::min(Dtype(0.),
																col_buff[col_offset_
																		* g
																		+ kernelind
																				* Iin
																		+ iinind]
																		* weights[weight_offset_
																				* g
																				+ routind
																						* K
																				+ kernelind]
																		+ bterm
																				/ K)
														* z2;
							}
						} //for (long routind = 0; routind < Rout; ++routind) {
					} //for (long iinind = 0; iinind < Iin; ++iinind) {
				} //case 1:
					break;

				default: {
					LOG(FATAL) << "uknown value for: ro.biastreatmenttype "
							<< ro.biastreatmenttype;
				}
					break;
				} //switch(ro.biasbiastreatmenttype)
			} else //if (this->bias_term_) 
			{
				for (long iinind = 0; iinind < Iin; ++iinind) {
					for (long routind = 0; routind < Rout; ++routind) {
						Dtype z1 = 0;
						if (pos_sums_data[routind * Iin + iinind] > 0) {
							z1 = upperlayerrelevances[output_offset_ * g
									+ routind * Iin + iinind]
									/ pos_sums_data[routind * Iin + iinind];
						}
						Dtype z2 = 0;
						if (neg_sums_data[routind * Iin + iinind] < 0) {
							z2 = upperlayerrelevances[output_offset_ * g
									+ routind * Iin + iinind]
									/ neg_sums_data[routind * Iin + iinind];
						}
						for (long kernelind = 0; kernelind < K; ++kernelind) {

							col_buff_new[col_offset_ * g + kernelind * Iin
									+ iinind] +=
									alpha
											* std::max(Dtype(0.),
													col_buff[col_offset_ * g
															+ kernelind * Iin
															+ iinind]
															* weights[weight_offset_
																	* g
																	+ routind
																			* K
																	+ kernelind])
											* z1
											- beta
													* std::min(Dtype(0.),
															col_buff[col_offset_
																	* g
																	+ kernelind
																			* Iin
																	+ iinind]
																	* weights[weight_offset_
																			* g
																			+ routind
																					* K
																			+ kernelind])
													* z2;
						}
					} //for (long routind = 0; routind < Rout; ++routind) {
				} //for (long iinind = 0; iinind < Iin; ++iinind) {

			} //else of if (this->bias_term_) 
		} else // if(beta>0)
		{

			if (this->bias_term_) {
				switch (ro.biastreatmenttype) {
				case 0: //ignore bias
				{
					for (long iinind = 0; iinind < Iin; ++iinind) {
						for (long routind = 0; routind < Rout; ++routind) {
							Dtype z1 = 0;
							if (pos_sums_data[routind * Iin + iinind] > 0) {
								z1 = upperlayerrelevances[output_offset_ * g
										+ routind * Iin + iinind]
										/ pos_sums_data[routind * Iin + iinind];
							}

							for (long kernelind = 0; kernelind < K;
									++kernelind) {

								col_buff_new[col_offset_ * g + kernelind * Iin
										+ iinind] += alpha
										* std::max(Dtype(0.),
												col_buff[col_offset_ * g
														+ kernelind * Iin
														+ iinind]
														* weights[weight_offset_
																* g
																+ routind * K
																+ kernelind])
										* z1;
							}
						} //for (long routind = 0; routind < Rout; ++routind) {
					} //for (long iinind = 0; iinind < Iin; ++iinind) {
				} //case 0:
					break;

				case 1: //dist bias as 1/n
				{
					for (long iinind = 0; iinind < Iin; ++iinind) {
						for (long routind = 0; routind < Rout; ++routind) {
							Dtype bterm = this->blobs_[1]->cpu_data()[routind]
									* bias_multiplier_.cpu_data()[iinind];

							Dtype z1 = 0;
							if (pos_sums_data[routind * Iin + iinind] > 0) {
								z1 = upperlayerrelevances[output_offset_ * g
										+ routind * Iin + iinind]
										/ pos_sums_data[routind * Iin + iinind];
							}

							for (long kernelind = 0; kernelind < K;
									++kernelind) {

								col_buff_new[col_offset_ * g + kernelind * Iin
										+ iinind] += alpha
										* std::max(Dtype(0.),
												col_buff[col_offset_ * g
														+ kernelind * Iin
														+ iinind]
														* weights[weight_offset_
																* g
																+ routind * K
																+ kernelind]
														+ bterm / K) * z1;
							}
						} //for (long routind = 0; routind < Rout; ++routind) {
					} //for (long iinind = 0; iinind < Iin; ++iinind) {
				} //case 1:
					break;
				default: {
					LOG(FATAL) << "uknown value for: ro.biastreatmenttype "
							<< ro.biastreatmenttype;
				}
					break;
				} //switch(ro.biasbiastreatmenttype)
			} else //if (this->bias_term_) 
			{

				for (long iinind = 0; iinind < Iin; ++iinind) {
					for (long routind = 0; routind < Rout; ++routind) {
						Dtype z1 = 0;
						if (pos_sums_data[routind * Iin + iinind] > 0) {
							z1 = upperlayerrelevances[output_offset_ * g
									+ routind * Iin + iinind]
									/ pos_sums_data[routind * Iin + iinind];
						}

						for (long kernelind = 0; kernelind < K; ++kernelind) {

							col_buff_new[col_offset_ * g + kernelind * Iin
									+ iinind] += alpha
									* std::max(Dtype(0.),
											col_buff[col_offset_ * g
													+ kernelind * Iin + iinind]
													* weights[weight_offset_ * g
															+ routind * K
															+ kernelind]) * z1;
						}
					} //for (long routind = 0; routind < Rout; ++routind) {
				} //for (long iinind = 0; iinind < Iin; ++iinind) {

			} // else of //if (this->bias_term_) 
		} // else of  if(beta>0)

	} //for (int g = 0; g < group_; ++g) {
	  //we need to use the diffs , not the data!!
	if (!is_1x1_) {
		conv_col2im_cpu(col_buff_new, new_lowerlayerrelevances);
	}

}




template<typename Dtype>
void BaseConvolutionLayer<Dtype>::alphabeta_4cases(
		const Dtype* upperlayerrelevances, //ex output,
		const Dtype* weights, const Dtype* input,
		Dtype * new_lowerlayerrelevances, const relpropopts & ro,
		bool skip_im2col) {
	int K = kernel_dim_; //xi gets summed along this with wij
	int Iin = conv_out_spatial_dim_;
	int Rout = conv_out_channels_ / group_;

	float beta =  ro.alphabeta_beta;
	float alpha = 1.0 + beta;
	// inputs col_buff are K x Iin
	//weights are Rout x K
	// upper relevances are Rout x Iin

	const Dtype* col_buff = input;
	if (!is_1x1_) {
		if (!skip_im2col) {
			conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
		}
		col_buff = col_buffer_.cpu_data();
	}

	Dtype* col_buff_new = col_buffer_.mutable_cpu_diff();
	if (is_1x1_) {
		col_buff_new = new_lowerlayerrelevances;
	}
	//memset(col_buff_new, 0,
	//		sizeof(Dtype) * conv_out_spatial_dim_ * kernel_dim_);
	memset(col_buff_new, 0,
			sizeof(Dtype) * col_buffer_.count());
	//
	Blob < Dtype > pos_sums(1, 1, Rout, Iin);
	Blob < Dtype > neg_sums(1, 1, Rout, Iin);
	Dtype* pos_sums_data = pos_sums.mutable_cpu_data();
	Dtype* neg_sums_data = neg_sums.mutable_cpu_data();
	

	Blob < Dtype > xpos(col_buffer_.shape());
	Dtype* xpos_data=xpos.mutable_cpu_data();
	caffe_copy(col_buffer_.count(),col_buff, xpos_data);
	for(int i=0;i< xpos.count();++i)
	{
		xpos_data[i]=std::max(Dtype(0.),xpos_data[i]);
	}
	
	Blob < Dtype > xneg(col_buffer_.shape());
	Dtype* xneg_data=xneg.mutable_cpu_data();
	caffe_copy(col_buffer_.count(),col_buff, xneg_data);
	for(int i=0;i< xneg.count();++i)
	{
		xneg_data[i]=std::min(Dtype(0.),xneg_data[i]);
	}
	
	Dtype *wpos=new Dtype[ weight_offset_ * group_ ];
	caffe_copy(weight_offset_ * group_,weights, wpos);
	for(int i=0;i< weight_offset_ * group_;++i)
	{
		wpos[i]=std::max(Dtype(0.),wpos[i]);
	}
	
	Dtype *wneg=new Dtype[ weight_offset_ * group_ ];					  
	caffe_copy(weight_offset_ * group_,weights, wneg);
	for(int i=0;i< weight_offset_ * group_;++i)
	{
		wneg[i]=std::min(Dtype(0.),wneg[i]);
	}
	
	const Dtype* xpos_cdata=xpos.cpu_data();
	const Dtype* xneg_cdata=xneg.cpu_data();
	
	Blob < Dtype > wijplus_tjplus___wijminus_tjplus(col_buffer_.shape());
	Blob < Dtype > wijplus_tjminus___wijminus_tjminus(col_buffer_.shape());

	Dtype * wijplus_tjplus_data=wijplus_tjplus___wijminus_tjplus.mutable_cpu_data();
	Dtype * wijminus_tjplus_data=wijplus_tjplus___wijminus_tjplus.mutable_cpu_diff();
	Dtype * wijplus_tjminus_data=wijplus_tjminus___wijminus_tjminus.mutable_cpu_data();
	Dtype * wijminus_tjminus_data=wijplus_tjminus___wijminus_tjminus.mutable_cpu_diff();

	Dtype* Rj_tjplus = new Dtype[ output_offset_ * group_ ];
	Dtype* Rj_tjminus = new Dtype[ output_offset_ * group_ ];
	
	caffe_copy(output_offset_ * group_, upperlayerrelevances, Rj_tjplus);
	caffe_copy(output_offset_ * group_, upperlayerrelevances, Rj_tjminus);
	
	for (int g = 0; g < group_; ++g) {
		/*
		 caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
		 conv_out_spatial_dim_, conv_out_channels_ / group_,
		 (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
		 (Dtype)0., col_buff + col_offset_ * g);
		 */
		memset(pos_sums_data, 0, sizeof(Dtype) * Rout * Iin);
		memset(neg_sums_data, 0, sizeof(Dtype) * Rout * Iin);

		if (beta > 0) {
			if (this->bias_term_) {
				switch (ro.biastreatmenttype) {
				case 0: //ignore bias
				{
																//Rout						Iin					K_
					caffe_cpu_gemm < Dtype
							> (CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_, (Dtype) 1., wpos
									+ weight_offset_ * g, xpos_cdata + col_offset_ * g, (Dtype) 0., pos_sums_data);
					caffe_cpu_gemm < Dtype
							> (CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_, (Dtype) 1., wneg
									+ weight_offset_ * g, xneg_cdata + col_offset_ * g, (Dtype) 1., pos_sums_data);
					
					caffe_cpu_gemm < Dtype
							> (CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_, (Dtype) 1., wneg
									+ weight_offset_ * g, xpos_cdata + col_offset_ * g, (Dtype) 0., neg_sums_data);
					caffe_cpu_gemm < Dtype
							> (CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_, (Dtype) 1., wpos
									+ weight_offset_ * g, xneg_cdata + col_offset_ * g, (Dtype) 1., neg_sums_data);
					
					//	caffe_cpu_gemm < Dtype
					//> (CblasNoTrans, CblasNoTrans, num_output_, out_spatial_dim_, 1, (Dtype) 1., bias, bias_multiplier_.cpu_data(), (Dtype) 1., output);
					
					for (long iinind = 0; iinind < Iin; ++iinind) {
						for (long routind = 0; routind < Rout; ++routind) {
							//pos_sums_data[routind,iinind]

							// x is col_buff[kernelind,iinind]

							//w is weights[routind,kernelind]
							
								Dtype bterm = this->blobs_[1]->cpu_data()[routind]
										* bias_multiplier_.cpu_data()[iinind];
								pos_sums_data[routind * Iin + iinind] += std::max(
										Dtype(0.), bterm);
								neg_sums_data[routind * Iin + iinind] += std::min(
										Dtype(0.), bterm);
							

						} //for (long routind = 0; routind < Rout; ++routind) {
					} //for (long iinind = 0; iinind < Iin; ++iinind) {
					
				} // case 0:
				break;
				
				default: {
					LOG(FATAL) << "BaseConvolutionLayer<Dtype>::alphabeta_4cases(...) does not support this ro.biastreatmenttype" << ro.biastreatmenttype << " if ro.biastreatmenttype==1, then use ro.codeexectype=1 to call void alphabeta(...) instead ";
				}
					break;
				}//				switch (ro.biastreatmenttype) {

			}
			else //if (this->bias_term_) {
			{	
				caffe_cpu_gemm < Dtype
						> (CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_, (Dtype) 1., wpos
								+ weight_offset_ * g, xpos_cdata + col_offset_ * g, (Dtype) 0., pos_sums_data);
				caffe_cpu_gemm < Dtype
						> (CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_, (Dtype) 1., wneg
								+ weight_offset_ * g, xneg_cdata + col_offset_ * g, (Dtype) 1., pos_sums_data);
				
				caffe_cpu_gemm < Dtype
						> (CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_, (Dtype) 1., wneg
								+ weight_offset_ * g, xpos_cdata + col_offset_ * g, (Dtype) 0., neg_sums_data);
				caffe_cpu_gemm < Dtype
						> (CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_, (Dtype) 1., wpos
								+ weight_offset_ * g, xneg_cdata + col_offset_ * g, (Dtype) 1., neg_sums_data);
			} // else of if (this->bias_term_) {
		} else // if(beta>0)
		{
			
			if (this->bias_term_) {
				switch (ro.biastreatmenttype) {
				case 0: //ignore bias
				{
					
					caffe_cpu_gemm < Dtype
							> (CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_, (Dtype) 1., wpos
									+ weight_offset_ * g, xpos_cdata + col_offset_ * g, (Dtype) 0., pos_sums_data);
					caffe_cpu_gemm < Dtype
							> (CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_, (Dtype) 1., wneg
									+ weight_offset_ * g, xneg_cdata + col_offset_ * g, (Dtype) 1., pos_sums_data);
					
					for (long iinind = 0; iinind < Iin; ++iinind) {
						for (long routind = 0; routind < Rout; ++routind) {
							//pos_sums_data[routind,iinind]

							// x is col_buff[kernelind,iinind]

							//w is weights[routind,kernelind]
							
								Dtype bterm = this->blobs_[1]->cpu_data()[routind]
										* bias_multiplier_.cpu_data()[iinind];
								pos_sums_data[routind * Iin + iinind] += std::max(
										Dtype(0.), bterm);
								neg_sums_data[routind * Iin + iinind] += std::min(
										Dtype(0.), bterm);
							

						} //for (long routind = 0; routind < Rout; ++routind) {
					} //for (long iinind = 0; iinind < Iin; ++iinind) {
				}//case 0:
				break;

				default: {
					LOG(FATAL) << "BaseConvolutionLayer<Dtype>::alphabeta_4cases(...) does not support this ro.biastreatmenttype" << ro.biastreatmenttype << " if ro.biastreatmenttype==1, then use ro.codeexectype=1 to call void alphabeta(...) instead ";

				}
					break;
				}//				switch (ro.biastreatmenttype) {

			}
			else //if (this->bias_term_) {
			{
				caffe_cpu_gemm < Dtype
						> (CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_, (Dtype) 1., wpos
								+ weight_offset_ * g, xpos_cdata + col_offset_ * g, (Dtype) 0., pos_sums_data);
				caffe_cpu_gemm < Dtype
						> (CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_, (Dtype) 1., wneg
								+ weight_offset_ * g, xneg_cdata + col_offset_ * g, (Dtype) 1., pos_sums_data);
			} // else of //if (this->bias_term_) {
		} // else of  if(beta>0)





		for(int i=0;i< pos_sums.count();++i)
		{
			if(pos_sums_data[i]>0)
			{
				Rj_tjplus[output_offset_ * g + i]/= pos_sums_data[i];
			}
		}
		
		if (beta > 0) {
		for(int i=0;i< neg_sums.count();++i)
		{
			if(neg_sums_data[i]<0)
			{
				Rj_tjminus[output_offset_ * g + i]/= neg_sums_data[i];
			}
		}
		}


		
		if (beta > 0) {
			if (this->bias_term_) {
				switch (ro.biastreatmenttype) {
				case 0: //ignore bias
				{
					caffe_cpu_gemm < Dtype
							> (CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_
									/ group_, (Dtype) 1., wpos + weight_offset_ * g, Rj_tjplus
									+ output_offset_ * g, (Dtype) 0., wijplus_tjplus_data
									+ col_offset_ * g);
					
					caffe_cpu_gemm < Dtype
							> (CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_
									/ group_, (Dtype) 1., wneg + weight_offset_ * g, Rj_tjplus
									+ output_offset_ * g, (Dtype) 0., wijminus_tjplus_data
									+ col_offset_ * g);
					
					caffe_cpu_gemm < Dtype
							> (CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_
									/ group_, (Dtype) 1., wpos + weight_offset_ * g, Rj_tjminus
									+ output_offset_ * g, (Dtype) 0., wijplus_tjminus_data
									+ col_offset_ * g);
					
					caffe_cpu_gemm < Dtype
							> (CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_
									/ group_, (Dtype) 1., wneg + weight_offset_ * g, Rj_tjminus
									+ output_offset_ * g, (Dtype) 0., wijminus_tjminus_data
									+ col_offset_ * g);
					
					
					for (int i=0;i< col_offset_ ;++i)
					{
						wijplus_tjplus_data[col_offset_ * g + i]*=xpos_data[col_offset_ * g + i];
					}
					
					for (int i=0;i< col_offset_ ;++i)
					{
						wijminus_tjplus_data[col_offset_ * g + i]*=xneg_data[col_offset_ * g + i];
					}
					
					for (int i=0;i< col_offset_ ;++i)
					{
						wijplus_tjminus_data[col_offset_ * g + i]*=xneg_data[col_offset_ * g + i];
					}
					
					for (int i=0;i< col_offset_ ;++i)
					{
						wijminus_tjminus_data[col_offset_ * g + i]*=xpos_data[col_offset_ * g + i];
					}
					
					caffe_axpy<Dtype>(col_offset_, alpha, wijplus_tjplus_data + col_offset_ * g,
							col_buff_new + col_offset_ * g);
					caffe_axpy<Dtype>(col_offset_, alpha, wijminus_tjplus_data + col_offset_ * g,
							col_buff_new + col_offset_ * g);
					
					caffe_axpy<Dtype>(col_offset_, -beta, wijplus_tjminus_data + col_offset_ * g,
							col_buff_new + col_offset_ * g);
					caffe_axpy<Dtype>(col_offset_, -beta, wijminus_tjminus_data + col_offset_ * g,
							col_buff_new + col_offset_ * g);
					

					/*
					for (long iinind = 0; iinind < Iin; ++iinind) {
						for (long routind = 0; routind < Rout; ++routind) {
							Dtype z1 = 0;
							if (pos_sums_data[routind * Iin + iinind] > 0) {
								z1 = upperlayerrelevances[output_offset_ * g
										+ routind * Iin + iinind]
										/ pos_sums_data[routind * Iin + iinind];
							}
							Dtype z2 = 0;
							if (neg_sums_data[routind * Iin + iinind] < 0) {
								z2 = upperlayerrelevances[output_offset_ * g
										+ routind * Iin + iinind]
										/ neg_sums_data[routind * Iin + iinind];
							}
							for (long kernelind = 0; kernelind < K;
									++kernelind) {

								col_buff_new[col_offset_ * g + kernelind * Iin
										+ iinind] +=
										alpha
												* std::max(Dtype(0.),
														col_buff[col_offset_ * g
																+ kernelind
																		* Iin
																+ iinind]
																* weights[weight_offset_
																		* g
																		+ routind
																				* K
																		+ kernelind])
												* z1
												- beta
														* std::min(Dtype(0.),
																col_buff[col_offset_
																		* g
																		+ kernelind
																				* Iin
																		+ iinind]
																		* weights[weight_offset_
																				* g
																				+ routind
																						* K
																				+ kernelind])
														* z2;
							}
						} //for (long routind = 0; routind < Rout; ++routind) {
					} //for (long iinind = 0; iinind < Iin; ++iinind) {
					*/
				} //case 0:
					break;

				default: {
					LOG(FATAL) << "BaseConvolutionLayer<Dtype>::alphabeta_4cases(...) does not support this ro.biastreatmenttype" << ro.biastreatmenttype << " if ro.biastreatmenttype==1, then use ro.codeexectype=1 to call void alphabeta(...) instead ";

				}
					break;
				} //switch(ro.biasbiastreatmenttype)
			} else //if (this->bias_term_) 
			{
				caffe_cpu_gemm < Dtype
						> (CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_
								/ group_, (Dtype) 1., wpos + weight_offset_ * g, Rj_tjplus
								+ output_offset_ * g, (Dtype) 0., wijplus_tjplus_data
								+ col_offset_ * g);
				
				caffe_cpu_gemm < Dtype
						> (CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_
								/ group_, (Dtype) 1., wneg + weight_offset_ * g, Rj_tjplus
								+ output_offset_ * g, (Dtype) 0., wijminus_tjplus_data
								+ col_offset_ * g);
				
				caffe_cpu_gemm < Dtype
						> (CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_
								/ group_, (Dtype) 1., wpos + weight_offset_ * g, Rj_tjminus
								+ output_offset_ * g, (Dtype) 0., wijplus_tjminus_data
								+ col_offset_ * g);
				
				caffe_cpu_gemm < Dtype
						> (CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_
								/ group_, (Dtype) 1., wneg + weight_offset_ * g, Rj_tjminus
								+ output_offset_ * g, (Dtype) 0., wijminus_tjminus_data
								+ col_offset_ * g);
				
				
				for (int i=0;i< col_offset_ ;++i)
				{
					wijplus_tjplus_data[col_offset_ * g + i]*=xpos_data[col_offset_ * g + i];
				}
				
				for (int i=0;i< col_offset_ ;++i)
				{
					wijminus_tjplus_data[col_offset_ * g + i]*=xneg_data[col_offset_ * g + i];
				}
				
				for (int i=0;i< col_offset_ ;++i)
				{
					wijplus_tjminus_data[col_offset_ * g + i]*=xneg_data[col_offset_ * g + i];
				}
				
				for (int i=0;i< col_offset_ ;++i)
				{
					wijminus_tjminus_data[col_offset_ * g + i]*=xpos_data[col_offset_ * g + i];
				}
				
				caffe_axpy<Dtype>(col_offset_, alpha, wijplus_tjplus_data + col_offset_ * g,
						col_buff_new + col_offset_ * g);
				caffe_axpy<Dtype>(col_offset_, alpha, wijminus_tjplus_data + col_offset_ * g,
						col_buff_new + col_offset_ * g);
				
				caffe_axpy<Dtype>(col_offset_, -beta, wijplus_tjminus_data + col_offset_ * g,
						col_buff_new + col_offset_ * g);
				caffe_axpy<Dtype>(col_offset_, -beta, wijminus_tjminus_data + col_offset_ * g,
						col_buff_new + col_offset_ * g);

			} //else of if (this->bias_term_) 
		} else // if(beta>0)
		{

			if (this->bias_term_) {
				switch (ro.biastreatmenttype) {
				case 0: //ignore bias
				{
					caffe_cpu_gemm < Dtype
							> (CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_
									/ group_, (Dtype) 1., wpos + weight_offset_ * g, Rj_tjplus
									+ output_offset_ * g, (Dtype) 0., wijplus_tjplus_data
									+ col_offset_ * g);
					
					caffe_cpu_gemm < Dtype
							> (CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_
									/ group_, (Dtype) 1., wneg + weight_offset_ * g, Rj_tjplus
									+ output_offset_ * g, (Dtype) 0., wijminus_tjplus_data
									+ col_offset_ * g);
					
					
					for (int i=0;i< col_offset_ ;++i)
					{
						wijplus_tjplus_data[col_offset_ * g + i]*=xpos_data[col_offset_ * g + i];
					}
					
					for (int i=0;i< col_offset_ ;++i)
					{
						wijminus_tjplus_data[col_offset_ * g + i]*=xneg_data[col_offset_ * g + i];
					}
					
					
					caffe_axpy<Dtype>(col_offset_, alpha, wijplus_tjplus_data + col_offset_ * g,
							col_buff_new + col_offset_ * g);
					caffe_axpy<Dtype>(col_offset_, alpha, wijminus_tjplus_data + col_offset_ * g,
							col_buff_new + col_offset_ * g);

				} //case 0:
					break;

				
				default: {
					LOG(FATAL) << "BaseConvolutionLayer<Dtype>::alphabeta_4cases(...) does not support this ro.biastreatmenttype" << ro.biastreatmenttype << " if ro.biastreatmenttype==1, then use ro.codeexectype=1 to call void alphabeta(...) instead ";

				}
					break;
				} //switch(ro.biasbiastreatmenttype)
			} else //if (this->bias_term_) 
			{

				caffe_cpu_gemm < Dtype
						> (CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_
								/ group_, (Dtype) 1., wpos + weight_offset_ * g, Rj_tjplus
								+ output_offset_ * g, (Dtype) 0., wijplus_tjplus_data
								+ col_offset_ * g);
				
				caffe_cpu_gemm < Dtype
						> (CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_
								/ group_, (Dtype) 1., wneg + weight_offset_ * g, Rj_tjplus
								+ output_offset_ * g, (Dtype) 0., wijminus_tjplus_data
								+ col_offset_ * g);
				
				
				for (int i=0;i< col_offset_ ;++i)
				{
					wijplus_tjplus_data[col_offset_ * g + i]*=xpos_data[col_offset_ * g + i];
				}
				
				for (int i=0;i< col_offset_ ;++i)
				{
					wijminus_tjplus_data[col_offset_ * g + i]*=xneg_data[col_offset_ * g + i];
				}
				
				
				caffe_axpy<Dtype>(col_offset_, alpha, wijplus_tjplus_data + col_offset_ * g,
						col_buff_new + col_offset_ * g);
				caffe_axpy<Dtype>(col_offset_, alpha, wijminus_tjplus_data + col_offset_ * g,
						col_buff_new + col_offset_ * g);
			} // else of //if (this->bias_term_) 
		} // else of  if(beta>0)

	} //for (int g = 0; g < group_; ++g) {
	  //we need to use the diffs , not the data!!
	if (!is_1x1_) {
		conv_col2im_cpu(col_buff_new, new_lowerlayerrelevances);
	}

	
	delete[] wpos;
	delete[] wneg;
	delete[] Rj_tjplus;
	delete[] Rj_tjminus;

}



template<typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
		const Dtype* output, Dtype* weights) {
	const Dtype* col_buff = input;
	if (!is_1x1_) {
		conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
		col_buff = col_buffer_.cpu_data();
	}
	for (int g = 0; g < group_; ++g) {
		caffe_cpu_gemm < Dtype
				> (CblasNoTrans, CblasTrans, conv_out_channels_ / group_, kernel_dim_, conv_out_spatial_dim_, (Dtype) 1., output
						+ output_offset_ * g, col_buff + col_offset_ * g, (Dtype) 1., weights
						+ weight_offset_ * g);
	}
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
		const Dtype* input) {
	caffe_cpu_gemv < Dtype
			> (CblasNoTrans, num_output_, out_spatial_dim_, 1., input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
		const Dtype* weights, Dtype* output, bool skip_im2col) {
	const Dtype* col_buff = input;
	if (!is_1x1_) {
		if (!skip_im2col) {
			conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
		}
		col_buff = col_buffer_.gpu_data();
	}
	for (int g = 0; g < group_; ++g) {
		caffe_gpu_gemm < Dtype
				> (CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_, (Dtype) 1., weights
						+ weight_offset_ * g, col_buff + col_offset_ * g, (Dtype) 0., output
						+ output_offset_ * g);
	}
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
		const Dtype* bias) {
	caffe_gpu_gemm < Dtype
			> (CblasNoTrans, CblasNoTrans, num_output_, out_spatial_dim_, 1, (Dtype) 1., bias, bias_multiplier_.gpu_data(), (Dtype) 1., output);
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
		const Dtype* weights, Dtype* input) {
	Dtype* col_buff = col_buffer_.mutable_gpu_data();
	if (is_1x1_) {
		col_buff = input;
	}
	for (int g = 0; g < group_; ++g) {
		caffe_gpu_gemm < Dtype
				> (CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_
						/ group_, (Dtype) 1., weights + weight_offset_ * g, output
						+ output_offset_ * g, (Dtype) 0., col_buff
						+ col_offset_ * g);
	}
	if (!is_1x1_) {
		conv_col2im_gpu(col_buff, input);
	}
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
		const Dtype* output, Dtype* weights) {
	const Dtype* col_buff = input;
	if (!is_1x1_) {
		conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
		col_buff = col_buffer_.gpu_data();
	}
	for (int g = 0; g < group_; ++g) {
		caffe_gpu_gemm < Dtype
				> (CblasNoTrans, CblasTrans, conv_out_channels_ / group_, kernel_dim_, conv_out_spatial_dim_, (Dtype) 1., output
						+ output_offset_ * g, col_buff + col_offset_ * g, (Dtype) 1., weights
						+ weight_offset_ * g);
	}
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
		const Dtype* input) {
	caffe_gpu_gemv < Dtype
			> (CblasNoTrans, num_output_, out_spatial_dim_, 1., input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS (BaseConvolutionLayer);

}  // namespace caffe
