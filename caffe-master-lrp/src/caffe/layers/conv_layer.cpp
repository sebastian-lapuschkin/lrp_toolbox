#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


#include "caffe/relpropopts.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_shape_data[i])
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_Relevance_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom, 
	  const int layerindex, const relpropopts & ro, const std::vector<int> & classinds, const bool thenightstartshere )
{
	
	switch(ro.relpropformulatype)
	{
		case 0:
		{
			//epsstab
			switch(ro.codeexectype)
			{
				case 0:
				{
					//slowneasy
					Backward_Relevance_cpu_epsstab_slowneasy(top,
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
		
			
		}
		break;
		
		case 2:
		{
			//epsstab
			switch(ro.codeexectype)
			{
				case 0:
				{

					Backward_Relevance_cpu_alphabeta_4cases(top,
						 propagate_down, bottom, 
						 layerindex, ro );
				}
				break;
				case 1:
				{
					//slowneasy
					Backward_Relevance_cpu_alphabeta_slowneasy(top,
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


template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_Relevance_cpu_epsstab_slowneasy(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom, 
	  const int layerindex, const relpropopts & ro ) {
	
	  const Dtype* weight = this->blobs_[0]->cpu_data();

	
	  for (int i = 0; i < top.size(); ++i) {
	    const Dtype* top_diff = top[i]->cpu_diff();
	    const Dtype* bottom_data = bottom[i]->cpu_data();
	    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
		//memset(bottom_diff, (Dtype)0., sizeof(Dtype) * bottom[i]->count());
	    caffe_set(bottom[i]->count(), Dtype(0.0), bottom_diff);
	    
	    
	    const Dtype* top_data = top[i]->cpu_data();
	    
	    
	    int Mfull=this->num_output_; //conv_out_channels_ /group_;
	    
	    const int first_spatial_axis = this->channel_axis_ + 1;
	    int N= bottom[i]->count(first_spatial_axis); //this->conv_out_spatial_dim_;
	    int K= this->blobs_[0]->count(1); //this->kernel_dim_;
	    
	    //Blob<Dtype> topdata_witheps(1, 1, 1, 1);
	    //topdata_witheps.Reshape(top[i]->shape());
	    Blob<Dtype> topdata_witheps((top[i])->shape());
	    
	    int outcount=topdata_witheps.count();
	    
    	LOG(INFO) << "M: this->num_output_"<< this->num_output_ ;
    	LOG(INFO) << "K: this->blobs_[0]->count(1)"<< this->blobs_[0]->count(1) ;
    	LOG(INFO) << "??: this->blobs_[0]->count(0)"<< this->blobs_[0]->count(0) ;
    	LOG(INFO) << "N: bottom[i]->count(first_spatial_axis)"<< bottom[i]->count(first_spatial_axis) ;
    	LOG(INFO) << "N: bottom[i]->count(0)"<< bottom[i]->count(0) ;

	    /*
	    if(topdata_witheps.count()!=Mfull * N)
	    {
	    	
	    	LOG(ERROR) << "top.size()"<< top.size()  ;
	    	LOG(ERROR) << i << " at i ,  top[i]->shape_string() " << top[i]->shape_string() ;
	    	LOG(ERROR) << i << " at i ,  bottom[i]->shape_string() " << bottom[i]->shape_string() ;
	    	
	    	LOG(ERROR) << "this->num_"<< this->num_ ;

	    	LOG(ERROR) << "this->top_dim_"<< this->top_dim_ ;
	    	LOG(ERROR) << "this->bottom_dim_"<< this->bottom_dim_ ;

	    	LOG(ERROR) << "M: this->num_output_"<< this->num_output_ ;
	    	LOG(ERROR) << "K: this->blobs_[0]->count(1)"<< this->blobs_[0]->count(1) ;
	    	LOG(ERROR) << "N: bottom[i]->count(first_spatial_axis)"<< bottom[i]->count(first_spatial_axis) ;

	    	
	    	LOG(FATAL) << "Incorrect weight shape: "<< topdata_witheps.shape_string()
	    	 << " Incorrect weight count: "<< topdata_witheps.count() << " " << outcount
			 << " expected count " << Mfull * N ;

	    }
	    */
	    
	    
	    Dtype* topdata_witheps_data=topdata_witheps.mutable_cpu_data();
	    caffe_copy<Dtype>(outcount, top_diff, topdata_witheps_data);
	    
  
	      for(int c=0;c< outcount ;++c)
	      {
	    	  //something_J =R_j / (output_j + eps * sign (output_j) )
	    	if(top_data[c]>0)
	    	{	
	    	  topdata_witheps_data[c]/= top_data[c]+ro.epsstab;
	    	} 
	    	else if(top_data[c]<0)
	    	{
	    		topdata_witheps_data[c]/= top_data[c]-ro.epsstab;
	    	}
	    	
	    	//if (isnan(topdata_witheps_data[c]))
	    	//{
	    	//	LOG(ERROR) << "have a nan at c=" << c << " top_diff[c]" <<top_diff[c]  <<  " top_data[c] " << top_data[c] << "ro.epsstab" << ro.epsstab; 
	    	//}
	    	
	      } //for(int c=0;c< M * N ;++c)

	      for (int n = 0; n < this->num_; ++n) {
	    	  
	    	  // stuff_i = \sum_j w_{ij} something_j  
	        this->backward_cpu_gemm(topdata_witheps_data + n * this->top_dim_, weight,
	              bottom_diff + n * this->bottom_dim_);
	        		
		      // now bottom_diff * bottom_data
	        	for(int d=0;d< this->bottom_dim_ ;++d)
	        	{
	        		bottom_diff[d + n * this->bottom_dim_]*=bottom_data[d + n * this->bottom_dim_]; // R_i = x_i * stuff_i
	    	    	//if (isnan(bottom_data[d]))
	    	    	//{
	    	    	//	LOG(ERROR) << "have a nan in bottom_data at d=" << d ;
	    	    	//}
	        	}
	        
	      } // for (int n = 0; n < this->num_; ++n)
	      
	    
	    
	  } // for (int i = 0; i < top.size(); ++i) {
}


template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_Relevance_cpu_alphabeta_slowneasy(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom, 
	  const int layerindex, const relpropopts & ro ) {
	
	  const Dtype* weight = this->blobs_[0]->cpu_data();
	  
		if(ro.alphabeta_beta <0)
		{
			LOG(FATAL) << "ro.alphabeta_beta <0 should be non-neg" << ro.alphabeta_beta ; 
		}

	
	  for (int i = 0; i < top.size(); ++i) {
	    const Dtype* top_diff = top[i]->cpu_diff();
	    const Dtype* bottom_data = bottom[i]->cpu_data();
	    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
		//memset(bottom_diff, 0, sizeof(Dtype) * bottom[i]->count());
	    caffe_set(bottom[i]->count(), Dtype(0.0), bottom_diff);

	    //const Dtype* top_data = top[i]->cpu_data();
	    
	    
	    const int first_spatial_axis = this->channel_axis_ + 1;
    	LOG(INFO) << "M: this->num_output_"<< this->num_output_ ;
    	LOG(INFO) << "K: this->blobs_[0]->count(1)"<< this->blobs_[0]->count(1) ;
    	LOG(INFO) << "??: this->blobs_[0]->count(0)"<< this->blobs_[0]->count(0) ;
    	LOG(INFO) << "N: bottom[i]->count(first_spatial_axis)"<< bottom[i]->count(first_spatial_axis) ;
    	LOG(INFO) << "N: bottom[i]->count(0)"<< bottom[i]->count(0) ;
    	LOG(INFO) << "this->num_"<< this->num_ ;



	      for (int n = 0; n < this->num_; ++n) {
	    	  
	       // this->backward_cpu_gemm(topdata_witheps_data + n * this->top_dim_, weight,
	       //       bottom_diff + n * this->bottom_dim_);
	        		
	        this->alphabeta(
	        		top_diff+n * this->top_dim_, //ex output,
	        		weight, bottom_data + n * this->bottom_dim_, 
					bottom_diff + n * this->bottom_dim_, ro);
	        
	      } // for (int n = 0; n < this->num_; ++n)
	      
	    
	    
	  } // for (int i = 0; i < top.size(); ++i) {
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_Relevance_cpu_alphabeta_4cases(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom, 
	  const int layerindex, const relpropopts & ro ) {
	
	  const Dtype* weight = this->blobs_[0]->cpu_data();
	  
		if(ro.alphabeta_beta <0)
		{
			LOG(FATAL) << "ro.alphabeta_beta <0 should be non-neg" << ro.alphabeta_beta ; 
		}

	
	  for (int i = 0; i < top.size(); ++i) {
	    const Dtype* top_diff = top[i]->cpu_diff();
	    const Dtype* bottom_data = bottom[i]->cpu_data();
	    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
		//memset(bottom_diff, 0, sizeof(Dtype) * bottom[i]->count());
	    caffe_set(bottom[i]->count(), Dtype(0.0), bottom_diff);

	    //const Dtype* top_data = top[i]->cpu_data();
	    
	    
	    const int first_spatial_axis = this->channel_axis_ + 1;
  	LOG(INFO) << "M: this->num_output_"<< this->num_output_ ;
  	LOG(INFO) << "K: this->blobs_[0]->count(1)"<< this->blobs_[0]->count(1) ;
  	LOG(INFO) << "??: this->blobs_[0]->count(0)"<< this->blobs_[0]->count(0) ;
  	LOG(INFO) << "N: bottom[i]->count(first_spatial_axis)"<< bottom[i]->count(first_spatial_axis) ;
  	LOG(INFO) << "N: bottom[i]->count(0)"<< bottom[i]->count(0) ;
  	LOG(INFO) << "this->num_"<< this->num_ ;



	      for (int n = 0; n < this->num_; ++n) {
	    	  
	       // this->backward_cpu_gemm(topdata_witheps_data + n * this->top_dim_, weight,
	       //       bottom_diff + n * this->bottom_dim_);
	        		
	        this->alphabeta_4cases(
	        		top_diff+n * this->top_dim_, //ex output,
	        		weight, bottom_data + n * this->bottom_dim_,
					bottom_diff + n * this->bottom_dim_, ro);
	        
	      } // for (int n = 0; n < this->num_; ++n)
	      
	    
	    
	  } // for (int i = 0; i < top.size(); ++i) {
}


#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
