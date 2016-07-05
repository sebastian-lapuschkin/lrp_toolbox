#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageDataMultiLabelLayer<Dtype>::~ImageDataMultiLabelLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageDataMultiLabelLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source_img = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening image file " << source_img;
  std::ifstream infileimg(source_img.c_str());

  LOG(INFO) << "Opening label file " << source_img+"_labelfile.txt";
  std::ifstream infilelabel((source_img+"_labelfile.txt").c_str());

    numlabels_=-1;

    infilelabel >> numlabels_;

    if(numlabels_ <=1)
    {
	LOG(FATAL) << "failed to read labelfile << "<< source_img+"_labelfile.txt"<< " numlabels_ <=1 " << numlabels_ ;
    }	
 vector<int> lbfile(numlabels_);
    std::string imgfile;
  while(! ( infileimg.eof() || infileimg.bad() || infileimg.fail()  ))
  {

    getline(infileimg,imgfile);

    for(int c=0;c<numlabels_;++c)
    {
	infilelabel >> lbfile[c];
    }
    if(imgfile.length()>3)
    {
      imgs_.push_back(imgfile);
	multilabels_.push_back(lbfile);
    }
		
  }
  //string filename;
  //int label;
  //while (infile >> filename >> label) {
  //  lines_.push_back(std::make_pair(filename, label));
  //}
  

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << imgs_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(imgs_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + imgs_[lines_id_],
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << imgs_[lines_id_];
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(2);
//label_shape[0]=numlabels_;
//label_shape[1]=batch_size;
label_shape[0]=batch_size;
label_shape[1]=numlabels_;
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ImageDataMultiLabelLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
   std::vector<int> inds(imgs_.size());
   for(int i=0;i< (int)inds.size();++i)
  {
	inds[i]=i;
  }  
  shuffle(inds.begin(), inds.end(), prefetch_rng);
std::vector<std::string> imgs_copy(imgs_);
  vector< vector<int> > multilabels_copy(multilabels_);
   for(int i=0;i< (int)inds.size();++i)
   {
multilabels_[i]=multilabels_copy[inds[i]];
imgs_[i]=imgs_copy[inds[i]];
   }

}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataMultiLabelLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + imgs_[lines_id_],
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << imgs_[lines_id_];
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = imgs_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + imgs_[lines_id_],
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << imgs_[lines_id_];
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

   if(batch->label_.count()!= numlabels_ * batch_size)
	{
LOG(FATAL) << " batch->label_.count()!= numlabels_ * batch_size " << batch->label_.count() << " vs" << numlabels_ * batch_size;
	}
    for (int lb=0;lb<numlabels_;++lb)
  {
//transpose here ???
    //prefetch_label[lb*batch_size+ item_id] = multilabels_[lines_id_][lb];
    prefetch_label[lb+ item_id * numlabels_] = multilabels_[lines_id_][lb];

 }
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageDataMultiLabelLayer);
REGISTER_LAYER_CLASS(ImageDataMultiLabel);

}  // namespace caffe
#endif  // USE_OPENCV
