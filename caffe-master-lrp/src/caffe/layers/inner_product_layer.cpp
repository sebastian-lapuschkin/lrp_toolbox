#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const int num_output =
			this->layer_param_.inner_product_param().num_output();
	bias_term_ = this->layer_param_.inner_product_param().bias_term();
	N_ = num_output;
	const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.inner_product_param().axis());
	// Dimensions starting from "axis" are "flattened" into a single
	// length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
	// and axis == 1, N inner products with dimension CHW are performed.
	K_ = bottom[0]->count(axis);
	// Check if we need to set up the weights
	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	} else {
		if (bias_term_) {
			this->blobs_.resize(2);
		} else {
			this->blobs_.resize(1);
		}
		// Intialize the weight
		vector<int> weight_shape(2);
		weight_shape[0] = N_;
		weight_shape[1] = K_;
		this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
		// fill the weights
		shared_ptr < Filler<Dtype>
				> weight_filler(
						GetFiller < Dtype
								> (this->layer_param_.inner_product_param().weight_filler()));
		weight_filler->Fill(this->blobs_[0].get());
		// If necessary, intiialize and fill the bias term
		if (bias_term_) {
			vector<int> bias_shape(1, N_);
			this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
			shared_ptr < Filler<Dtype>
					> bias_filler(
							GetFiller < Dtype
									> (this->layer_param_.inner_product_param().bias_filler()));
			bias_filler->Fill(this->blobs_[1].get());
		}
	}  // parameter initialization
	this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// Figure out the dimensions
	const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.inner_product_param().axis());
	const int new_K = bottom[0]->count(axis);
	CHECK_EQ(K_, new_K)
			<< "Input size incompatible with inner product parameters.";
	// The first "axis" dimensions are independent inner products; the total
	// number of these is M_, the product over these dimensions.
	M_ = bottom[0]->count(0, axis);
	// The top shape will be the bottom shape with the flattened axes dropped,
	// and replaced by a single axis with dimension num_output (N_).
	vector<int> top_shape = bottom[0]->shape();
	top_shape.resize(axis + 1);
	top_shape[axis] = N_;
	top[0]->Reshape(top_shape);
	// Set up the bias multiplier
	if (bias_term_) {
		vector<int> bias_shape(1, M_);
		bias_multiplier_.Reshape(bias_shape);
		caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
	}
}

template<typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const Dtype* weight = this->blobs_[0]->cpu_data();
	caffe_cpu_gemm < Dtype
			> (CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype) 1., bottom_data, weight, (Dtype) 0., top_data);
	if (bias_term_) {
		caffe_cpu_gemm < Dtype
				> (CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype) 1., bias_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(), (Dtype) 1., top_data);
	}
}

template<typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	if (this->param_propagate_down_[0]) {
		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		// Gradient with respect to weight
		caffe_cpu_gemm < Dtype
				> (CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype) 1., top_diff, bottom_data, (Dtype) 1., this->blobs_[0]->mutable_cpu_diff());
	}
	if (bias_term_ && this->param_propagate_down_[1]) {
		const Dtype* top_diff = top[0]->cpu_diff();
		// Gradient with respect to bias
		caffe_cpu_gemv < Dtype
				> (CblasTrans, M_, N_, (Dtype) 1., top_diff, bias_multiplier_.cpu_data(), (Dtype) 1., this->blobs_[1]->mutable_cpu_diff());
	}
	if (propagate_down[0]) {
		const Dtype* top_diff = top[0]->cpu_diff();
		// Gradient with respect to bottom data
		caffe_cpu_gemm < Dtype
				> (CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype) 1., top_diff, this->blobs_[0]->cpu_data(), (Dtype) 0., bottom[0]->mutable_cpu_diff());
	}
}

template<typename Dtype>
void InnerProductLayer<Dtype>::Backward_Relevance_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom, const int layerindex,
		const relpropopts & ro, const std::vector<int> & classinds,
		const bool thenightstartshere) {

	switch (ro.relpropformulatype) {
	case 0: // epsilon-type formula
	{
		switch (ro.codeexectype) {
		case 0: {
			//slowneasy
			LOG(INFO) << "InnerproductLayer ro.relpropformulatype " << ro.relpropformulatype << " (eps)";
			Backward_Relevance_cpu_epsstab_slowneasy(top, propagate_down,
					bottom, layerindex, ro);
		}
		break;
		default: {
			LOG(FATAL) << "unknown value for ro.codeexectype "
					<< ro.codeexectype << std::endl;
			exit(1);
		}
			break;
		} //	switch(ro.codeexectype)

	}
	break;

	case 2: // (alpha-beta)-type formula
	{
		switch (ro.codeexectype) {
		case 0:
		case 1:
		{
			//slowneasy
			LOG(INFO) << "InnerproductLayer ro.relpropformulatype " << ro.relpropformulatype << " (alphabeta)";
			Backward_Relevance_cpu_alphabeta_slowneasy(top, propagate_down,
					bottom, layerindex, ro);
		}
			break;
		default: {
			LOG(FATAL) << "unknown value for ro.codeexectype "
					<< ro.codeexectype << std::endl;
			exit(1);
		}
			break;
		} //	switch(ro.codeexectype)

	}
	break;

	case 54: // epsilon + flat below a given layer index
	{
		if(ro.auxiliaryvariable_maxlayerindexforflatdistinconv<0)
		{
			LOG(FATAL) << "ro.auxiliaryvariable_maxlayerindexforflatdistinconv not set for this case in convlayer";
		}
		if(layerindex <= ro.auxiliaryvariable_maxlayerindexforflatdistinconv)
		{
			LOG(INFO) << "InnerproductLayer ro.relpropformulatype " << ro.relpropformulatype << " (flat)";
			LOG(FATAL) << "DISABLED";
			exit(1);
			//Backward_Relevance_cpu_flatdist_slowneasy(top, propagate_down, bottom,	layerindex, ro );
		}
		else
		{
			LOG(INFO) << "InnerproductLayer ro.relpropformulatype " << ro.relpropformulatype << " (eps)";
			Backward_Relevance_cpu_epsstab_slowneasy(top,
					propagate_down, bottom,
					layerindex, ro );
		}
	}
	break;

	case 56: // epsilon + w^2 below a given layer index
	{
		if(ro.auxiliaryvariable_maxlayerindexforflatdistinconv<0)
		{
			LOG(FATAL) << "ro.auxiliaryvariable_maxlayerindexforflatdistinconv not set for this case in convlayer";
		}

		if(layerindex <= ro.auxiliaryvariable_maxlayerindexforflatdistinconv)
		{
			LOG(INFO) << "InnerproductLayer ro.relpropformulatype " << ro.relpropformulatype << " (wsquare)";
			LOG(FATAL) << "DISABLED";
			exit(1);
			//Backward_Relevance_cpu_wsquare(top, propagate_down, bottom,	layerindex, ro );
		}
		else
		{
			LOG(INFO) << "InnerproductLayer ro.relpropformulatype " << ro.relpropformulatype << " (eps)";
			Backward_Relevance_cpu_epsstab_slowneasy(top,
					propagate_down, bottom,
					layerindex, ro );
		}
	}
	break;

	case 58: // (alpha-beta) + flat below a given layer index
	{
		if(ro.auxiliaryvariable_maxlayerindexforflatdistinconv<0)
		{
			LOG(FATAL) << "ro.auxiliaryvariable_maxlayerindexforflatdistinconv not set for this case in convlayer";
		}
		if(layerindex <= ro.auxiliaryvariable_maxlayerindexforflatdistinconv)
		{
			LOG(INFO) << "InnerproductLayer ro.relpropformulatype " << ro.relpropformulatype << " (flat)";
			//Backward_Relevance_cpu_flatdist_slowneasy(top, propagate_down, bottom,layerindex, ro );
			LOG(FATAL) << "DISABLED";
			exit(1);
		}
		else
		{
			LOG(INFO) << "InnerproductLayer ro.relpropformulatype " << ro.relpropformulatype << " (alphabeta)";
			Backward_Relevance_cpu_alphabeta_slowneasy(top,
					propagate_down, bottom,
					layerindex, ro );
		}
	}
	break;

	case 60: // (alpha-beta) + w^2 below a given layer index
	{
		if(ro.auxiliaryvariable_maxlayerindexforflatdistinconv<0)
		{
			LOG(FATAL) << "ro.auxiliaryvariable_maxlayerindexforflatdistinconv not set for this case in convlayer";
		}

		if(layerindex <= ro.auxiliaryvariable_maxlayerindexforflatdistinconv)
		{
			LOG(INFO) << "InnerproductLayer ro.relpropformulatype " << ro.relpropformulatype << " (wsquare)";
			LOG(FATAL) << "DISABLED";
			exit(1);
			//Backward_Relevance_cpu_wsquare(top,	propagate_down, bottom, layerindex, ro );
		}
		else
		{
			LOG(INFO) << "InnerproductLayer ro.relpropformulatype " << ro.relpropformulatype << " (alphabeta)";
			Backward_Relevance_cpu_alphabeta_slowneasy(top,
					propagate_down, bottom,
					layerindex, ro );
		}
	}
	break;

	case 100: // decomposition type per layer: (alpha-beta) for conv layers, epsilon for inner product layers
	{
		switch (ro.codeexectype) {
		case 0: {
			//slowneasy
			LOG(INFO) << "InnerproductLayer ro.relpropformulatype " << ro.relpropformulatype << " (eps)";
			Backward_Relevance_cpu_epsstab_slowneasy(top, propagate_down,
					bottom, layerindex, ro);
		}
		break;
		default: {
			LOG(FATAL) << "unknown value for ro.codeexectype "
					<< ro.codeexectype << std::endl;
			exit(1);
		}
			break;
		} //	switch(ro.codeexectype)

	}
	break;

	case 102: // composite method + flat below a given layer index
	{
		if(ro.auxiliaryvariable_maxlayerindexforflatdistinconv<0)
		{
			LOG(FATAL) << "ro.auxiliaryvariable_maxlayerindexforflatdistinconv not set for this case in convlayer";
		}
		if(layerindex <= ro.auxiliaryvariable_maxlayerindexforflatdistinconv)
		{
			LOG(INFO) << "InnerproductLayer ro.relpropformulatype " << ro.relpropformulatype << " (flat)";
			LOG(FATAL) << "DISABLED";
			exit(1);
			//Backward_Relevance_cpu_flatdist_slowneasy(top, propagate_down, bottom, layerindex, ro );
		}
		else
		{
			LOG(INFO) << "InnerproductLayer ro.relpropformulatype " << ro.relpropformulatype << " (eps)";
			Backward_Relevance_cpu_epsstab_slowneasy(top,
					propagate_down, bottom,
					layerindex, ro );
		}
	}
	break;

	case 104: // decomposition type per layer + w^2 below a given layer index
	{
		if(ro.auxiliaryvariable_maxlayerindexforflatdistinconv<0)
		{
			LOG(FATAL) << "ro.auxiliaryvariable_maxlayerindexforflatdistinconv not set for this case in convlayer";
		}

		if(layerindex <= ro.auxiliaryvariable_maxlayerindexforflatdistinconv)
		{
			LOG(INFO) << "InnerproductLayer ro.relpropformulatype " << ro.relpropformulatype << " (wsquare)";
			LOG(FATAL) << "DISABLED";
			exit(1);
			//Backward_Relevance_cpu_wsquare(top,propagate_down, bottom,layerindex, ro );
		}
		else
		{
			LOG(INFO) << "InnerproductLayer ro.relpropformulatype " << ro.relpropformulatype << " (eps)";
			Backward_Relevance_cpu_epsstab_slowneasy(top,
					propagate_down, bottom,
					layerindex, ro );
		}
	}
	break;



	// EXPERIMENTAL AND OTHERS BELOW

	case 6: // (alpha-beta) + z^beta on lowest considered layer
	case 8: // epsilon + z^beta on lowest considered layer
	case 11: //gradient in demonstrator
	case 18:
	case 22:
	{
		int fc6layerindex=15;
		if (layerindex > fc6layerindex )
		{
			relpropopts ro2=ro;
			ro2.relpropformulatype=0;
			ro2.alphabeta_beta=0;
			Backward_Relevance_cpu_alphabeta_slowneasy(top, propagate_down,
				bottom, layerindex, ro2);
		}
		else
		{
			relpropopts ro2=ro;
			ro2.relpropformulatype=0;
			ro2.alphabeta_beta=1;
			Backward_Relevance_cpu_alphabeta_slowneasy(top, propagate_down,
				bottom, layerindex, ro2);
		}
	}
	break;


	case 26: //zeiler: deconvolution
	{
		switch (ro.codeexectype) {
		case 0: {
			LOG(INFO) << "InnerproductLayer ro.relpropformulatype " << ro.relpropformulatype << " (Deconvolution: Zeiler)";
			//slowneasy
			Backward_Relevance_cpu_zeilerlike_slowneasy(top, propagate_down,
					bottom, layerindex, ro);
		}
			break;
		default: {
			LOG(FATAL) << "unknown value for ro.codeexectype "
					<< ro.codeexectype << std::endl;
			exit(1);
		}
			break;
		} //	switch(ro.codeexectype)

	}
		break;


	default: {
		LOG(FATAL) << "unknown value for ro.relpropformulatype "
				<< ro.relpropformulatype << std::endl;
		exit(1);
	}
		break;

	}

}

template<typename Dtype>
void InnerProductLayer<Dtype>::Backward_Relevance_cpu_epsstab_slowneasy(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom, const int layerindex,
		const relpropopts & ro) {

        // an inplace relu may alter the layer, thats why the forward here
        Forward_cpu( bottom, top);

	for (int i = 0; i < top.size(); ++i) {
		const Dtype* top_diff = top[i]->cpu_diff();
		const Dtype* bottom_data = bottom[i]->cpu_data();
		Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
		//memset(bottom_diff, 0, sizeof(Dtype) * bottom[i]->count());
	    caffe_set(bottom[i]->count(), Dtype(0.0), bottom_diff);


		const Dtype* top_data = top[i]->cpu_data();
		Blob < Dtype > topdata_witheps((top[i])->shape());

		LOG(INFO) << "top.size()" << top.size() << "part of it:" << i
				<< " weight shape: " << topdata_witheps.shape_string();
		LOG(INFO) << "M_, K_, N_" << M_ << " "<< K_ << " "<< N_;
		int outcount = topdata_witheps.count();
		if (topdata_witheps.count() != M_ * N_) {
			LOG(FATAL) << "Incorrect weight shape: "
					<< topdata_witheps.shape_string()
					<< " Incorrect weight count: " << topdata_witheps.count()
					<< " " << outcount << " expected count " << M_ * N_;

			exit(1);
		}

		Dtype* topdata_witheps_data = topdata_witheps.mutable_cpu_data();
		caffe_copy < Dtype > (outcount, top_diff, topdata_witheps_data);

		for (int c = 0; c < outcount; ++c) {
			//something_J =R_j / (output_j + eps * sign (output_j) )
			if (top_data[c] > 0) {
				topdata_witheps_data[c] /= top_data[c] + ro.epsstab;
			} else if (top_data[c] < 0) {
				topdata_witheps_data[c] /= top_data[c] - ro.epsstab;
			}
		} //for(int c=0;c< M * N ;++c)

	      //for (int n = 0; n < this->num_; ++n) {
		caffe_cpu_gemm < Dtype
				> (CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype) 1., topdata_witheps_data, this->blobs_[0]->cpu_data(), (Dtype) 0., bottom[i]->mutable_cpu_diff());

		// now bottom_diff * bottom_data
		for (int d = 0; d < M_ * K_; ++d) {
			bottom_diff[d] *= bottom_data[d]; // R_i = x_i * stuff_i
		}
	   // } for (int n = 0; n < this->num_; ++n) {

	} //for (int i = 0; i < top.size(); ++i)
}


template<typename Dtype>
void InnerProductLayer<Dtype>::Backward_Relevance_cpu_zeilerlike_slowneasy(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom, const int layerindex,
		const relpropopts & ro) {

        Forward_cpu( bottom, top);

	for (int i = 0; i < top.size(); ++i) {
		const Dtype* top_diff = top[i]->cpu_diff();
		const Dtype* bottom_data = bottom[i]->cpu_data();
		Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
		//memset(bottom_diff, 0, sizeof(Dtype) * bottom[i]->count());
	    caffe_set(bottom[i]->count(), Dtype(0.0), bottom_diff);


		const Dtype* top_data = top[i]->cpu_data();
		Blob < Dtype > topdata_witheps((top[i])->shape());

		LOG(INFO) << "top.size()" << top.size() << "part of it:" << i
				<< " weight shape: " << topdata_witheps.shape_string();
		LOG(INFO) << "M_, K_, N_" << M_ << " "<< K_ << " "<< N_;
		int outcount = topdata_witheps.count();
		if (topdata_witheps.count() != M_ * N_) {
			LOG(FATAL) << "Incorrect weight shape: "
					<< topdata_witheps.shape_string()
					<< " Incorrect weight count: " << topdata_witheps.count()
					<< " " << outcount << " expected count " << M_ * N_;

			exit(1);
		}

		Dtype* topdata_witheps_data = topdata_witheps.mutable_cpu_data();
		caffe_copy < Dtype > (outcount, top_diff, topdata_witheps_data);

	      //for (int n = 0; n < this->num_; ++n) {
		caffe_cpu_gemm < Dtype
				> (CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype) 1., topdata_witheps_data, this->blobs_[0]->cpu_data(), (Dtype) 0., bottom[i]->mutable_cpu_diff());

		// now bottom_diff * bottom_data
		//for (int d = 0; d < M_ * K_; ++d) {
		//	bottom_diff[d] *= bottom_data[d]; // R_i = x_i * stuff_i
		//}
	   // } for (int n = 0; n < this->num_; ++n) {

	} //for (int i = 0; i < top.size(); ++i)
}



template<typename Dtype>
void InnerProductLayer<Dtype>::Backward_Relevance_cpu_alphabeta_slowneasy(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom, const int layerindex,
		const relpropopts & ro) {\

	if(ro.alphabeta_beta <0)
	{
		LOG(FATAL) << "ro.alphabeta_beta <0 should be non-neg" << ro.alphabeta_beta ;
	}

	for (int i = 0; i < top.size(); ++i) {
		const Dtype* top_diff = top[i]->cpu_diff();
		const Dtype* bottom_data = bottom[i]->cpu_data();

		const Dtype* weights = this->blobs_[0]->cpu_data();

		Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
		//memset(bottom_diff, 0, sizeof(Dtype) * bottom[i]->count());
	    caffe_set(bottom[i]->count(), Dtype(0.0), bottom_diff);


		Blob < Dtype > pos_sums(1, 1, M_, N_);
		Blob < Dtype > neg_sums(1, 1, M_, N_);

		Dtype* pos_sums_data = pos_sums.mutable_cpu_data();
		Dtype* neg_sums_data = neg_sums.mutable_cpu_data();
		memset(pos_sums_data, 0, sizeof(Dtype) * M_ * N_);
		memset(neg_sums_data, 0, sizeof(Dtype) * M_ * N_);

		/*
		for (long mind = 0; mind < M_; ++mind) {
			for (long enind = 0; enind < N_; ++enind) {

				for (long kernelind = 0; kernelind < K_; ++kernelind) {
					//if or max/min ?
					pos_sums_data[mind * N_ + enind] += std::max(Dtype(0.),
							bottom_data[mind * K_ + kernelind]
									* weights[enind * K_ + kernelind]);
					neg_sums_data[mind * N_ + enind] += std::min(Dtype(0.),
							bottom_data[mind * K_ + kernelind]
									* weights[enind * K_ + kernelind]);

				}


			}
		}
		*/


		if(bias_term_)
		{
			switch(ro.biastreatmenttype)
			{
			case 0: //ignore bias
			{
			for (long mind = 0; mind < M_; ++mind) {
				for (long enind = 0; enind < N_; ++enind) {

					for (long kernelind = 0; kernelind < K_; ++kernelind) {
						//if or max/min ?
						pos_sums_data[mind * N_ + enind] += std::max(Dtype(0.),
								bottom_data[mind * K_ + kernelind]
										* weights[enind * K_ + kernelind]);
						neg_sums_data[mind * N_ + enind] += std::min(Dtype(0.),
								bottom_data[mind * K_ + kernelind]
										* weights[enind * K_ + kernelind]);

					}

					Dtype bterm=bias_multiplier_.cpu_data()[mind] * this->blobs_[1]->cpu_data()[enind];
					pos_sums_data[mind * N_ + enind] += std::max(Dtype(0.),bterm);
					neg_sums_data[mind * N_ + enind] += std::min(Dtype(0.),bterm);

				}
			}
			}//case 0:
			break;
			case 1: //distrib bias
			{
			for (long mind = 0; mind < M_; ++mind) {
				for (long enind = 0; enind < N_; ++enind) {
					Dtype bterm=bias_multiplier_.cpu_data()[mind] * this->blobs_[1]->cpu_data()[enind];

					for (long kernelind = 0; kernelind < K_; ++kernelind) {
						//if or max/min ?
						pos_sums_data[mind * N_ + enind] += std::max(Dtype(0.),
								bottom_data[mind * K_ + kernelind]
										* weights[enind * K_ + kernelind] + bterm / K_ );
						neg_sums_data[mind * N_ + enind] += std::min(Dtype(0.),
								bottom_data[mind * K_ + kernelind]
										* weights[enind * K_ + kernelind] + bterm / K_ );

					}


				}
			}
			}//case 1:
			break;
			default:
			{
				LOG(FATAL) << "uknown value for: ro.biastreatmenttype " << ro.biastreatmenttype;
			}
			break;
			} //			switch(ro.biastreatmenttype)

		}
		else //if(bias_term_)
		{
			for (long mind = 0; mind < M_; ++mind) {
				for (long enind = 0; enind < N_; ++enind) {

					for (long kernelind = 0; kernelind < K_; ++kernelind) {
						//if or max/min ?
						pos_sums_data[mind * N_ + enind] += std::max(Dtype(0.),
								bottom_data[mind * K_ + kernelind]
										* weights[enind * K_ + kernelind]);
						neg_sums_data[mind * N_ + enind] += std::min(Dtype(0.),
								bottom_data[mind * K_ + kernelind]
										* weights[enind * K_ + kernelind]);

					}

				}
			}
		} // else of if(bias_term_)

		float beta=ro.alphabeta_beta;
		float alpha = 1.0 + beta;

		if (this->bias_term_) {
			switch(ro.biastreatmenttype)
			{
			case 0: //ignore bias
			{


		for (long mind = 0; mind < M_; ++mind) {
			for (long enind = 0; enind < N_; ++enind) {
				Dtype z1 = 0;
				if (pos_sums_data[mind * N_ + enind] > 0) {
					z1 = top_diff[mind * N_ + enind]
							/ pos_sums_data[mind * N_ + enind];
				}
				Dtype z2 = 0;
				if (neg_sums_data[mind * N_ + enind] < 0) {
					z2 = top_diff[mind * N_ + enind]
							/ neg_sums_data[mind * N_ + enind];
				}

				for (long kernelind = 0; kernelind < K_; ++kernelind) {

					bottom_diff[mind * K_ + kernelind] += alpha
							* std::max(Dtype(0.),
									bottom_data[mind * K_ + kernelind]
											* weights[enind * K_ + kernelind])
							* z1
							- beta
									* std::min(Dtype(0.),
											bottom_data[mind * K_ + kernelind]
													* weights[enind * K_
															+ kernelind]) * z2;
				}
			} //			for (long enind = 0; enind < N_; ++enind) {
		} //		for (long mind = 0; mind < M_; ++mind) {

			}//case 0:
			break;
			case 1: //distrib bias as 1/n
			{


		for (long mind = 0; mind < M_; ++mind) {
			for (long enind = 0; enind < N_; ++enind) {
				Dtype bterm=bias_multiplier_.cpu_data()[mind] * this->blobs_[1]->cpu_data()[enind];

				Dtype z1 = 0;
				if (pos_sums_data[mind * N_ + enind] > 0) {
					z1 = top_diff[mind * N_ + enind]
							/ pos_sums_data[mind * N_ + enind];
				}
				Dtype z2 = 0;
				if (neg_sums_data[mind * N_ + enind] < 0) {
					z2 = top_diff[mind * N_ + enind]
							/ neg_sums_data[mind * N_ + enind];
				}

				for (long kernelind = 0; kernelind < K_; ++kernelind) {

					bottom_diff[mind * K_ + kernelind] += alpha
							* std::max(Dtype(0.),
									bottom_data[mind * K_ + kernelind]
											* weights[enind * K_ + kernelind] + bterm / K_ )
							* z1
							- beta
									* std::min(Dtype(0.),
											bottom_data[mind * K_ + kernelind]
													* weights[enind * K_
															+ kernelind] + bterm / K_ ) * z2;
				}
			} //			for (long enind = 0; enind < N_; ++enind) {
		} //		for (long mind = 0; mind < M_; ++mind) {

			}//case 1:
			break;
			default:
			{
				LOG(FATAL) << "uknown value for: ro.biastreatmenttype " << ro.biastreatmenttype;
			}
			break;
			} //			switch(ro.biastreatmenttype)
		} //if (this->bias_term_) {
		else
		{
			for (long mind = 0; mind < M_; ++mind) {
				for (long enind = 0; enind < N_; ++enind) {
					Dtype z1 = 0;
					if (pos_sums_data[mind * N_ + enind] > 0) {
						z1 = top_diff[mind * N_ + enind]
								/ pos_sums_data[mind * N_ + enind];
					}
					Dtype z2 = 0;
					if (neg_sums_data[mind * N_ + enind] < 0) {
						z2 = top_diff[mind * N_ + enind]
								/ neg_sums_data[mind * N_ + enind];
					}

					for (long kernelind = 0; kernelind < K_; ++kernelind) {

						bottom_diff[mind * K_ + kernelind] += alpha
								* std::max(Dtype(0.),
										bottom_data[mind * K_ + kernelind]
												* weights[enind * K_ + kernelind])
								* z1
								- beta
										* std::min(Dtype(0.),
												bottom_data[mind * K_ + kernelind]
														* weights[enind * K_
																+ kernelind]) * z2;
					}
				} //			for (long enind = 0; enind < N_; ++enind) {
			} //		for (long mind = 0; mind < M_; ++mind) {
		} //else of if (this->bias_term_) {

		//  if (bias_term_) {
		//    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
		//        bias_multiplier_.cpu_data(),
		//        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
		//  }

} //for (int i = 0; i < top.size(); ++i)
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS (InnerProductLayer);
REGISTER_LAYER_CLASS (InnerProduct);

}  // namespace caffe
