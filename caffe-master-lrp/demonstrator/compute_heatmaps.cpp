//

/*

 g++ heatmapcomp_nogui_filelist.cpp -o hmcomp_lrp_filelist -I ../include/ -I ../build/src/  -I /usr/include/ImageMagick/ -I /usr/local/cuda-6.5/include -I /home/binder/installed_sw/open_cv_2_4_11/include/    -L /home/binder/installed_sw/open_cv_2_4_11/lib/ -L/usr/local/cuda-6.5/lib64   -Wl,--whole-archive  ../build/lib/libcaffe.a  -Wl,--no-whole-archive -lcudart -lcublas -lcurand -lpthread -lglog -lgflags -lprotobuf -lleveldb -lsnappy -lboost_system -lhdf5_hl -lhdf5 -llmdb -lopencv_core -lopencv_highgui -lopencv_imgproc -latlas -lcblas  -Wall -lMagick++ -lMagickWand -lMagickCore -lboost_filesystem -lboost_system -lboost_thread  -lpthread

 //`pkg-config gtkmm-3.0 --cflags --libs`
 */

#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <iomanip>

//#include <gtkmm.h>
#include <boost/filesystem.hpp>

#include "Magick++.h"

#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"

using namespace caffe;

void hsv2rgb(std::vector<std::vector<double> > & rgbpix,
		const std::vector<std::vector<double> > & hsvpix);

struct CmpPair {
	bool operator()(const std::pair<double, int> & a,
			const std::pair<double, int> & b) {
		return a.first < b.first;
	}
};

void sortingPermutation(const std::vector<double>& values,
		std::vector<int>& permutation);

void loadimagealex(std::vector<std::vector<double> > & img, int & hei,
		int & wid, const std::string & imgfile, const int maxdim);

//reads a mean file blob into floats
void readablob3d(std::vector<std::vector<float> > & channels, int & imgmeanhei,
		int & imgmeanwid, const std::string & mean_file);

void saveimgasjpg(const std::string & file, const int hei, const int wid,
		const std::vector<std::vector<double> > & img);

void saveimgaspng(const std::string & file, const int hei, const int wid,
		const std::vector<std::vector<double> > & img);

class configstuff {
public:

	std::string guifile;

	std::string param_file;
	std::string model_file;
	std::string synsetfile;
//	std::string imagedefaultfolder;
	std::string mean_file;

	int netinhei;
	int netinwid;

	int use_mean_file_asbinaryprotoblob;

	int lastlayerindex;
	int firstlayerindex;

	int biastreatmenttype;
	int baseimgsize;

	//added over gui version
	std::string standalone_outpath;
	std::string standalone_rootpath;
	float epsstab;
	float alphabeta_beta;
	int relpropformulatype;

	//int classindstype;
	int numclasses;

	void readconfig2(const std::string & configfile);

	bool filexists_check(const std::string & file);

	template<typename vartype>
	bool readattributefromstring(vartype & variable,
			const ::std::string & contents,
			const ::std::string & attributename);
	void deblankbeginandend(::std::string & str);

//	int outputheatmapstocurrentpath;

	int maxpoolingtoavgpoolinginbackwardpass;

};

class heatmaprunner {
public:

	heatmaprunner();

	void init(const std::string & configfile);
	void process_heatmap(const std::string & imgfile, const int classindstype2);

protected:
	void init_caffe();

	configstuff configs;

	int baseimghei, baseimgwid;

	std::vector<std::vector<float> > imgmean;

//static shared_ptr<Net<float> > net_;
	shared_ptr<Net<float> > net_;

	std::vector<std::vector<double> > img;
	std::vector<std::vector<double> > img3;
	std::vector<std::vector<double> > img2;

	std::vector<std::string> classnames;

	int inhei;
	int inwid;

	int baseimgsize;

	int imgmeanhei;
	int imgmeanwid;

	relpropopts ro;

	int classindstype;
	int numclasses;

	std::vector<std::vector<double> > img_asinputted;

	void painfully_processimage(const std::string & imgfile);

};

heatmaprunner::heatmaprunner() :
		inhei(-1), inwid(-1), baseimgsize(-1), imgmeanhei(-1), imgmeanwid(-1), classindstype(
				-1) {

}

void heatmaprunner::init(const std::string & configfile) {
	configs.readconfig2(configfile);

	ro.codeexectype = 0;
	ro.biastreatmenttype = configs.biastreatmenttype;

	ro.lrn_forward_type = 0;
	ro.lrn_backward_type = 1;

	ro.lastlayerindex = configs.lastlayerindex;
	ro.firstlayerindex = configs.firstlayerindex;

	baseimgsize = configs.baseimgsize;

	ro.alphabeta_beta = configs.alphabeta_beta;
	ro.epsstab = configs.epsstab;
	ro.relpropformulatype = configs.relpropformulatype;

	//classindstype=configs.classindstype;
	numclasses = configs.numclasses;
	ro.numclasses = numclasses;

	ro.maxpoolingtoavgpoolinginbackwardpass=configs.maxpoolingtoavgpoolinginbackwardpass;

	init_caffe();

}

void heatmaprunner::init_caffe()
{
	//inhei=configs.netinhei;
	//inwid=configs.netinwid;

	//read imgmean
	switch (configs.use_mean_file_asbinaryprotoblob) {
	case 0: {
		std::ifstream f;
		int tmpint = -1;
		f.open((configs.mean_file).c_str());
		f >> tmpint;
		imgmean.resize(tmpint);
		int tmphei = -1, tmpwid = -1;
		f >> tmphei;
		f >> tmpwid;

		for (size_t ch = 0; ch < imgmean.size(); ++ch) {
			imgmean[ch].resize(tmphei * tmpwid);

			for (int w = 0; w < tmpwid; ++w) {
				for (int h = 0; h < tmphei; ++h) {
					f >> imgmean[ch][h + w * tmphei];
				}
			}

		}
		imgmeanhei = tmphei;
		imgmeanwid = tmpwid;

		f.close();
	}
		break;

	case 1: {
		//bgr
		readablob3d(imgmean, imgmeanhei, imgmeanwid, configs.mean_file);
	}
		break;
	case 2: {
		//rgb
		std::vector<std::vector<float> > tmpchannels;
		readablob3d(tmpchannels, imgmeanhei, imgmeanwid, configs.mean_file);
		imgmean.resize(3);
		for (int ch = 0; ch < 3; ++ch) {
			imgmean[ch] = tmpchannels[2 - ch];
		}

	}
		break;
	default: {
		std::cerr
				<< "unrecognized value for configs.use_mean_file_asbinaryprotoblob "
				<< configs.use_mean_file_asbinaryprotoblob << std::endl;
		exit(1);
	}
	}

	std::cout << "imagemeanfile canvas size (hei,wid) " << imgmeanhei << " "
			<< imgmeanwid << std::endl;

	if ((imgmeanhei <= 0) || (imgmeanwid <= 0)) {
		std::cerr
				<< "imagemean file not read properly!! (imgmeanhei<=0)||(imgmeanwid<=0) "
				<< std::endl;
		exit(1);
	}

	if ((imgmeanhei < inhei) || (imgmeanwid < inwid)) {
		std::cerr
				<< "imagemean canvas size smaller than  nnet receptive field size (imgmeanhei< inhei )||(imgmeanwid < inwid) "
				<< " imgmeanhei vs  inhei " << imgmeanhei << " " << inhei
				<< " imgmeanwid vs  inwid " << imgmeanwid << " " << inwid
				<< std::endl;
		exit(1);
	}

	classnames.resize(numclasses);
	{
		std::ifstream f;
		f.open(configs.synsetfile.c_str());
		for (int i = 0; i < numclasses; ++i) {
			std::string tmp;
			getline(f, tmp);
			classnames[i] = tmp;

		}

		f.close();
	}


	std::cout << "loading net" << std::endl;
	{
		net_.reset(new Net<float>(configs.param_file, caffe::TEST));

		const std::vector<Blob<float>*>& input_blobs = net_->input_blobs();

		if (input_blobs.size() <= 0) {
			LOG(FATAL) << "input_blobs.size()<=0";
		}

		if (input_blobs.size() > 1) {
			LOG(FATAL)
					<< "input_blobs.size()>1 , dont know which input blob to fill!";
		}

		if (input_blobs[0]->num_axes() != 4) {
			LOG(FATAL)
					<< " input_blobs[0]->num_axes() != 4 . dont know how to get the receptive field size from that ";
		}
		if (input_blobs[0]->shape(2) != input_blobs[0]->shape(3)) {
			LOG(FATAL)
					<< "input_blobs[0]->shape(2) != input_blobs[0]->shape(3). code not tested for non-quadratic receptive fields. exiting here. remove this error message and check the code if to see if it works or what has to be changed.";
		}

		if (input_blobs[0]->shape(1) != 3) {
			LOG(FATAL)
					<< "input_blobs[0]->shape(1) != 3. code not tested for that. exiting here. remove this error message and check the code if to see if it works or what has to be changed.";
		}

		configs.netinhei = input_blobs[0]->shape(2);
		configs.netinwid = input_blobs[0]->shape(3);
		inhei = configs.netinhei;
		inwid = configs.netinwid;

		if (baseimgsize < std::max(configs.netinwid, configs.netinhei)) {
			LOG(FATAL)
					<< "error: baseimgsize < std::max(configs.netinwid,configs.netinhei) . imagesize after resizing cannot be smaller than neural net receptive field. baseimgsize "
					<< baseimgsize << " std::max(netinwid,netinhei) "
					<< std::max(configs.netinwid, configs.netinhei);

		}

		/*
		 if( (input_blobs[0]->count()<3 * inhei * inwid) || ( input_blobs[0]->count() % (3 * inhei * inwid )!=0))
		 {
		 LOG(ERROR)<< "(input_blobs[0]->count()<3 * inhei * inwid) || ( input_blobs[0]->count() % (3 * inhei * inwid )!=0)" ;

		 LOG(ERROR)<< 0 <<" ! inputblob size incombatible to: 3 * inhei * inwid=" << 3 * inhei * inwid << " inhei " << inhei << " inwid " << inwid << " input_blobs[i]->count() " << input_blobs[0]->count() <<std::endl;
		 LOG(ERROR) << " numcolor ch, inhei inwid: " << 3 << " "<< inhei <<" " << inwid <<std::endl;
		 LOG(FATAL) << "input_blobs[0]->shape_string() " << input_blobs[0]->shape_string() << std::endl;

		 }
		 */

	}
	std::cout << "loading net finished" << std::endl;
	net_->CopyTrainedLayersFrom(configs.model_file);
}

void heatmaprunner::painfully_processimage(const std::string & imgfile) {

	std::string imagefile = imgfile;

	int hei = -1;
	int wid = -1;

	loadimagealex(img, hei, wid, imagefile, baseimgsize);

	baseimghei = hei;
	baseimgwid = wid;

	if (img.size() == 1) {
		img.resize(3);
		img[1].resize(img[0].size());
		img[2].resize(img[0].size());
		std::copy(img[0].begin(), img[0].end(), img[1].begin());
		std::copy(img[0].begin(), img[0].end(), img[2].begin());
	}

	for (int ch = 0; ch < (int) imgmean.size(); ++ch) {

		for (int h = 0; h < hei; ++h) {
			for (int w = 0; w < wid; ++w) {
				img[ch][h + w * hei] *= 255.0;
//img[ch][h+w*hei]-=imgmean[2-ch][h+w*hei];
//img[ch][h+w*hei]/=255.0;
			}
		}
	}

	img3.resize(3);
	img2.resize(3);

	for (int ch = 0; ch < (int) imgmean.size(); ++ch) {
		img3[ch].resize(inhei * inwid);
	}

//crop img! t

	int bordersize = (baseimgsize - std::max(inhei, inwid)) / 2;
	int bordersize_imgmean = std::min((imgmeanhei - inhei) / 2,
			(imgmeanwid - inwid) / 2);
	std::cout << "bordersize" << bordersize << std::endl;
	std::cout << "bordersize_imgmean" << bordersize_imgmean << std::endl;
	std::cout << "inhei inwid" << inhei << " " << inwid << std::endl;

	for (int ch = 0; ch < (int) imgmean.size(); ++ch) {
		img2[ch].resize(inhei * inwid);
		std::fill(img2[ch].begin(), img2[ch].end(), 0.0);

		if (hei > wid) {
			for (int h = 0; h < inhei; ++h) {
				if (wid >= inwid) {
					int offset = (wid - inwid) / 2;
					int offset_mean = (imgmeanwid - inwid) / 2;
					for (int w = 0; w < inwid; ++w) {
						img3[ch][h + w * inhei] = img[ch][(bordersize + h)
								+ (offset + w) * hei];
						img2[ch][h + w * inhei] = img3[ch][h + w * inhei]
								- imgmean[2 - ch][(bordersize_imgmean + h)
										+ (offset_mean + w) * imgmeanhei];

					}
				} else //if(wid< inwid)
				{
					int offset = (inwid - wid) / 2;
					//int offset_mean=(imgmeanwid-inwid)/2;

					for (int w = 0; w < wid; ++w) {
						img3[ch][h + (offset + w) * inhei] = img[ch][(bordersize
								+ h) + (0 + w) * hei];
						img2[ch][h + (offset + w) * inhei] = img3[ch][h
								+ (offset + w) * inhei]
								- imgmean[2 - ch][(bordersize_imgmean + h)
										+ (0 + w) * imgmeanhei];

					}
					for (int w = 0; w < offset; ++w) {
						img3[ch][h + (w) * inhei] = img[ch][(bordersize + h)
								+ (0 + 0) * hei];
						img2[ch][h + (w) * inhei] = img3[ch][h + (w) * inhei]
								- imgmean[2 - ch][(bordersize_imgmean + h)
										+ (0 + 0) * imgmeanhei];

					}
					for (int w = offset + wid; w < inwid; ++w) {
						img3[ch][h + (w) * inhei] = img[ch][(bordersize + h)
								+ (0 + wid - 1) * hei];
						img2[ch][h + (w) * inhei] = img3[ch][h + (w) * inhei]
								- imgmean[2 - ch][(bordersize_imgmean + h)
										+ (0 + wid - 1) * imgmeanhei];

					}
				} //else of if(wid>=inwid)
			} // for(int h=0;h<inhei;++h)
		} else {
			for (int w = 0; w < inwid; ++w) {
				if (hei >= inhei) {
					int offset = (hei - inhei) / 2;
					int offset_mean = (imgmeanhei - inhei) / 2;

					for (int h = 0; h < inhei; ++h) {
						img3[ch][h + w * inhei] = img[ch][(offset + h)
								+ (bordersize + w) * hei];
						img2[ch][h + w * inhei] =
								img3[ch][h + w * inhei]
										- imgmean[2 - ch][(offset_mean + h)
												+ (bordersize_imgmean + w)
														* imgmeanhei];

					}
				} else // if(hei>=inhei)
				{
					int offset = (inhei - hei) / 2;
					for (int h = 0; h < hei; ++h) {
						img3[ch][(offset + h) + w * inhei] = img[ch][(0 + h)
								+ (bordersize + w) * hei];
						img2[ch][(offset + h) + w * inhei] =
								img3[ch][(offset + h) + w * inhei]
										- imgmean[2 - ch][(0 + h)
												+ (bordersize_imgmean + w)
														* imgmeanhei];

					}
					for (int h = 0; h < offset; ++h) {
						img3[ch][(h) + w * inhei] = img[ch][(0 + 0)
								+ (bordersize + w) * hei];
						img2[ch][(h) + w * inhei] =
								img3[ch][(h) + w * inhei]
										- imgmean[2 - ch][(0 + 0)
												+ (bordersize_imgmean + w)
														* imgmeanhei];

					}
					for (int h = hei + offset; h < inhei; ++h) {
						img3[ch][(h) + w * inhei] = img[ch][(0 + hei - 1)
								+ (bordersize + w) * hei];
						img2[ch][(h) + w * inhei] =
								img3[ch][(h) + w * inhei]
										- imgmean[2 - ch][(0 + hei - 1)
												+ (bordersize_imgmean + w)
														* imgmeanhei];

					}
				} // else of  if(hei>=inhei)
			}
		} // else of if(hei>wid)

	} //for(int ch=0;ch<imgmean.size();++ch)

	img_asinputted.resize(3);
	for (int ch = 0; ch < 3; ++ch) {
		img_asinputted[ch].resize(img3[ch].size());
		for (int h = 0; h < inhei; ++h) {
			for (int w = 0; w < inwid; ++w) {
				img_asinputted[ch][(h) + w * inhei] = img3[ch][(h) + w * inhei]
						/ 255;
			}
		}
	}

}

void getoutputpath_createdir(std::string & curoutpath,
		const std::string & prependpath, const std::string & rootstub,
		const std::string & imagefile) {
	assert(!rootstub.empty());

	size_t pos = imagefile.find(rootstub);

	if (pos == std::string::npos) {
		std::cerr << "did not find rootstub " << rootstub << " in imagefile "
				<< imagefile << std::endl;
		exit(1);
	}

	curoutpath = prependpath + "/" + imagefile.substr(pos);

	::boost::filesystem::path pt = curoutpath;
	pt = pt.branch_path();

	if (!::boost::filesystem::exists(pt)) {
		::boost::filesystem::create_directories(pt);
	}

}

void heatmaprunner::process_heatmap(const std::string & imgfile, const int classindstype2) {
	classindstype = classindstype2;

	::boost::filesystem::path pt(imgfile);
	if (!::boost::filesystem::is_regular_file(pt)) {
		std::cerr << "imagefile is no regular file " << imgfile << std::endl;
		exit(1);
	}

	std::string outputname;
	getoutputpath_createdir(outputname, configs.standalone_outpath,
			configs.standalone_rootpath, imgfile);

	painfully_processimage(imgfile);

	const std::vector<Blob<float>*>& input_blobs = net_->input_blobs();

	if (input_blobs[0]->count() != 3 * inhei * inwid) {
		LOG(WARNING)
				<< "input_blobs[0]->count()!=3 * inhei * inwid. You might be using an oversized network";
	}

	float* paddedimg = new float[3 * inhei * inwid];

	std::fill(paddedimg, paddedimg + 3 * inhei * inwid, 0.0);

	for (int ch = 0; ch < (int) imgmean.size(); ++ch) {
		for (int h = 0; h < inhei; ++h) {
			for (int w = 0; w < inwid; ++w) {
				paddedimg[ch * inhei * inwid + h * inwid + w] = img2[2 - ch][h
						+ w * inhei];
			}
		}
	}

	for (unsigned int i = 0; i < input_blobs.size(); ++i) {
		memcpy(input_blobs[i]->mutable_cpu_data(), paddedimg,
				sizeof(float) * 3 * inhei * inwid);
	}

	delete[] paddedimg;

//net_->set_normlayertreatment( lrpforlrnindex);
	std::cout << "preforward" << std::endl;
	const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
	std::cout << "postforward" << std::endl;

	std::vector<double> values(numclasses, -10);

	std::cout << "output_blobs.size() " << output_blobs.size() << std::endl;

	std::cout << "<output_blobs[0]->count() " << output_blobs[0]->count()
			<< std::endl;

	for (int i = 0; i < ro.numclasses; ++i) {
		values[i] = *(output_blobs[0]->cpu_data() + i);
		std::cout << ro.numclasses << " " << i << " values[i] " << values[i]
				<< std::endl;

	}

	std::vector<int> permutation;
	sortingPermutation(values, permutation);

	std::cout << "permutation.size() " << permutation.size() << " "
			<< classnames.size() << std::endl;

	{
		for (int i = 0; i < std::min(10, numclasses); ++i) {
			std::cout << permutation[numclasses - 1 - i] << " | "
					<< classnames[permutation[numclasses - 1 - i]] << " "
					<< values[permutation[numclasses - 1 - i]] << std::endl;

		}

//force_redraw();

	}
	std::cout << "prehm " << std::endl;

	std::vector<std::vector<double> > rawhm;

	{
		std::cout << "heatmapping for " << permutation[numclasses - 1]
				<< std::endl;
		std::vector<int> classinds2(1, permutation[numclasses - 1]);

		if (classindstype == -2) {
			classinds2.resize(std::min(5, numclasses));
			for (int i = 0; i < (int) classinds2.size(); ++i) {

				classinds2[i] = permutation[numclasses - 1 - i];
				std::cout << "heatmapping for " << classinds2[i] << std::endl;
			}
		} else if (classindstype >= 0) {
			classinds2[0] = classindstype;
			std::cout << "heatmapping for " << classinds2[0] << std::endl;
		}

		if(ro.relpropformulatype == 99){
			ro.relpropformulatype = 11;
		}
        if (ro.relpropformulatype == 11)
        {
            net_->Backward_Gradient(classinds2, rawhm, ro);
            //compute gradient l2 norm
            for(int p = 0; p<rawhm[0].size(); ++p)
            {
                double norm = sqrt(rawhm[0][p]*rawhm[0][p] + rawhm[1][p]*rawhm[1][p] + rawhm[2][p]*rawhm[2][p]);
                /*if (norm < 0)
                {
                    std::cout << "IMPOSSSIBLE! NORM IS " << norm << std::endl ;
                }else
                {
                    std::cout << norm << std::endl;
                }*/
                rawhm[0][p] = norm;
                rawhm[1][p] = norm;
                rawhm[2][p] = norm;
            }

        }
        else
        {
		    net_->Backward_Relevance(classinds2, rawhm, ro);
        }
	}

	std::cout << "posthm " << std::endl;

	{

		std::vector<double> vs(rawhm[0].size(), 0);
		double maxabs = 0;
		for (int i = 0; i < (int) vs.size(); ++i) {
			vs[i] = (rawhm[0][i] + rawhm[1][i] + rawhm[2][i]) / 3.0;
			//maxabs=std::max(maxabs,fabs(hues[i]));
			maxabs = std::max(maxabs, vs[i]);
		}

		std::vector<std::vector<double> > img5(3);
		for (int ch = 0; ch < 3; ++ch) {
			img5[ch].resize(rawhm[ch].size());
			std::fill(img5[ch].begin(), img5[ch].end(), 0);
		}

		if (maxabs > 0) {
			for (int h = 0; h < inhei; ++h) {
				for (int w = 0; w < inwid; ++w) {
					if (vs[(h) + w * inhei] < 0) {
						img5[2][(h) + w * inhei] = -vs[(h) + w * inhei]
								/ maxabs;
					} else {
						img5[0][(h) + w * inhei] = vs[(h) + w * inhei] / maxabs;
						img5[1][(h) + w * inhei] = vs[(h) + w * inhei] / maxabs;
					}

					//img5[ch][(h) + w * inhei] = jethm[ch][(h) + w * inhei];
				}
			}
		}

		std::cout << "postimg5create " << std::endl;
		//std::string outf1 = outputname + "_heatmap.jpg";
		//saveimgasjpg(outf1, inhei, inwid, img5);
		std::string outf = outputname + "_heatmap.png";
		saveimgaspng(outf, inhei, inwid, img5);

	}

	{
		std::string outf = outputname + "_top10scores.txt";
		std::ofstream f;
		f.open(outf.c_str());
		//output classinfo for min(numclasses,10) classes
		for (int i = 0; i < std::min(10, numclasses); ++i) {
			f << permutation[numclasses - 1 - i] << " "
					<< values[permutation[numclasses - 1 - i]] << std::endl;
		}
		f.close();
	}

	{

		std::string outfile = outputname + "_rawhm.txt";
		std::ofstream f;
		f.open(outfile.c_str());

		f << 3 << std::endl;
		f << inhei << " " << inwid << std::endl;

		for (int ch = 0; ch < 3; ++ch) {
			for (int h = 0; h < inhei; ++h) {
				for (int w = 0; w < inwid; ++w) {
					f << rawhm[ch][(h) + w * inhei] << " ";
				}
				f << std::endl;
			}
		}

		f.close();

		//std::string outfile2 = outputname + "_as_inputted_into_the_dnn.jpg";
		std::string outfile2 = outputname + "_as_inputted_into_the_dnn.png";
		if (img_asinputted.size() == 3) {
			if (((int) img_asinputted[0].size() == inhei * inwid)
					&& ((int) img_asinputted[1].size() == inhei * inwid)
					&& ((int) img_asinputted[2].size() == inhei * inwid)) {
				//saveimgasjpg(outfile2, inhei, inwid, img_asinputted);
				saveimgaspng(outfile2, inhei, inwid, img_asinputted);
			}
		}

	} //if(configs.outputheatmapstocurrentpath>0)

	//outputname+"_heatmap.jpg"
}

void readablob3d(std::vector<std::vector<float> > & channels, int & imgmeanhei,
		int & imgmeanwid, const std::string & mean_file) {
	int num_channels = 3;
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels)
			<< "Number of channels of mean file doesn't match input layer.";

	channels.resize(num_channels);
	/* The format of the mean file is planar 32-bit float BGR or grayscale. */

	float* data = mean_blob.mutable_cpu_data();

	imgmeanhei = mean_blob.height();
	imgmeanwid = mean_blob.width();
	for (int ch = 0; ch < num_channels; ++ch) {
		channels[ch].resize(mean_blob.height() * mean_blob.width());
		std::copy(data, data + mean_blob.height() * mean_blob.width(),
				channels[ch].begin());
		/* Extract an individual channel. */
		//std::vector<float> chn(mean_blob.height() * mean_blob.width());
		//std::copy(data,data+mean_blob.height() * mean_blob.width(),chn.begin());
		//cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		//channels.push_back(chn);
		data += mean_blob.height() * mean_blob.width();
	}
}

void sortingPermutation(const std::vector<double>& values,
		std::vector<int>& permutation) {
	std::vector<std::pair<double, int> > pairs;
	for (int i = 0; i < (int) values.size(); i++)
		pairs.push_back(std::pair<double, int>(values[i], i));

	std::sort(pairs.begin(), pairs.end(), CmpPair());

	typedef std::vector<std::pair<double, int> >::const_iterator I;
	for (I p = pairs.begin(); p != pairs.end(); ++p)
		permutation.push_back(p->second);
}

void loadimagealex(std::vector<std::vector<double> > & img, int & hei,
		int & wid, const std::string & imgfile, const int maxdim) {

	Magick::Image im;
	im.read(imgfile);
	double resizer = 1;
	if (im.columns() > im.rows()) {
		resizer = maxdim / (double) im.columns();
	} else {
		resizer = maxdim / (double) im.rows();
	}

	Magick::Geometry geo((int) round(im.columns() * resizer),
			(int) round(im.rows() * resizer));
	geo.aspect(false);
	im.resize(geo);

	std::ostringstream error;

	if ((im.type() != Magick::TrueColorType)
			&& (im.type() != Magick::GrayscaleType)) {
		std::cout << "imgtype " << im.type() << std::endl;
		try {
			im.type(Magick::TrueColorType);
		} catch (Magick::Exception & e) {
			throw e;
		} catch (...) {
			std::cerr << "umknown image type" << std::endl;
			exit(1);
		}
	}

	if (im.type() == Magick::TrueColorType) {
		// color image
		img.resize(3);
		for (int i = 0; i < (int) img.size(); ++i) {
			hei = im.rows();
			wid = im.columns();
			img[i].resize(hei * wid);

		}

		double largestvalue = pow(2.0, sizeof(Magick::Quantum) * 8) - 1;

		// here later a better solution can be found: just allocate a part of the pic if pics are very large
		Magick::PixelPacket *pixel, *pixel_cache = im.getPixels(0, 0,
				im.columns(), im.rows());

		for (unsigned int w = 0; w < im.columns(); ++w) {
			for (unsigned int h = 0; h < im.rows(); ++h) {
				pixel = &pixel_cache[w + im.columns() * h];

				img[0][h + hei * w] = (pixel->red / largestvalue);
				img[1][h + hei * w] = (pixel->green / largestvalue);
				img[2][h + hei * w] = (pixel->blue / largestvalue);
			}

		}
	} else if (im.type() == Magick::GrayscaleType) {
		std::cout << "grey image!!!" << std::endl;
		img.resize(1);
		for (int i = 0; i < (int) img.size(); ++i) {
			hei = im.rows();
			wid = im.columns();
			img[i].resize(hei * wid);

		}
		double largestvalue = pow(2.0, sizeof(Magick::Quantum) * 8) - 1;

		// here later a better solution can be found: just allocate a part of the pic if pics are very large
		Magick::PixelPacket *pixel, *pixel_cache = im.getPixels(0, 0,
				im.columns(), im.rows());

		for (unsigned int w = 0; w < im.columns(); ++w) {
			for (unsigned int h = 0; h < im.rows(); ++h) {
				pixel = &pixel_cache[w + im.columns() * h];

				img[0][h + hei * w] = (pixel->red / largestvalue);

			}

		}

		//create a 3-d image
		img.push_back(img[0]);
		img.push_back(img[0]);

	} else {
		std::cerr << "unknown image type" << std::endl;
		exit(1);
	}

}

void hsv2rgb(std::vector<std::vector<double> > & rgbpix,
		const std::vector<std::vector<double> > & hsvpix) {

	assert(hsvpix.size() == 3);
	rgbpix.resize(3);
	for (size_t i = 0; i < rgbpix.size(); ++i) {
		rgbpix[i].resize(hsvpix[0].size());
	}

	std::vector<double> h(hsvpix[0]), k(hsvpix[0].size()), p(hsvpix[0].size(),
			0), t(hsvpix[0].size(), 0), n(hsvpix[0].size(), 0);

	std::vector<int> k0(hsvpix[0].size(), 0), k1(hsvpix[0].size(), 0), k2(
			hsvpix[0].size(), 0), k3(hsvpix[0].size(), 0), k4(hsvpix[0].size(),
			0), k5(hsvpix[0].size(), 0);

	double maxval = 0;
	for (size_t i = 0; i < hsvpix[0].size(); ++i) {
		h[i] = 6 * h[i];
		k[i] = floor(h[i]);
		if ((k[i] == 0) || (k[i] == 6)) {
			k0[i] = 1;
		}
		if (k[i] == 1) {
			k1[i] = 1;
		}
		if (k[i] == 2) {
			k2[i] = 1;
		}
		if (k[i] == 3) {
			k3[i] = 1;
		}
		if (k[i] == 4) {
			k4[i] = 1;
		}
		if (k[i] == 5) {
			k5[i] = 1;
		}
		p[i] = h[i] - k[i];
		t[i] = 1 - hsvpix[1][i];
		n[i] = 1 - hsvpix[1][i] * p[i];
		p[i] = 1 - (hsvpix[1][i] * (1 - p[i]));

		/*
		 if(i%10==0)
		 {
		 std::cout << hsvpix[0][i] <<" " << hsvpix[1][i] <<" "<< hsvpix[2][i] <<" " <<std::endl;
		 std::cout << p[i] <<" "<< t[i] <<" "<< n[i] <<" " <<std::endl;
		 }
		 */
		rgbpix[0][i] = k0[i] + k1[i] * n[i] + k2[i] * t[i] + k3[i] * t[i]
				+ k4[i] * p[i] + k5[i];
		rgbpix[1][i] = k0[i] * p[i] + k1[i] + k2[i] + k3[i] * n[i]
				+ k4[i] * t[i] + k5[i] * t[i];
		rgbpix[2][i] = k0[i] * t[i] + k1[i] * t[i] + k2[i] * p[i] + k3[i]
				+ k4[i] + k5[i] * n[i];

		maxval = std::max(maxval, rgbpix[0][i]);
		maxval = std::max(maxval, rgbpix[1][i]);
		maxval = std::max(maxval, rgbpix[2][i]);

	} //for(size_t i=0;i< hsvpix[0].size();++i)

	for (size_t i = 0; i < hsvpix[0].size(); ++i) {
		double tmp = 0;
		if (maxval > 0) {
			tmp = hsvpix[2][i] / maxval;
		}
		rgbpix[0][i] *= tmp;
		rgbpix[1][i] *= tmp;
		rgbpix[2][i] *= tmp;

//std::cout << "r g b " << rgbpix[0][i] << " " << rgbpix[1][i] << " " << rgbpix[2][i] <<std::endl;
	}

} //function

// **********************************************

bool configstuff::filexists_check(const std::string & file) {
	::boost::filesystem::path pt = file;
	if (!::boost::filesystem::exists(pt)
			|| (::boost::filesystem::is_directory(pt))) {
		std::cerr << "error: file does not exist " << pt.native() << std::endl;
		exit(1);
		//return false;
	}
	return true;
}

void configstuff::deblankbeginandend(::std::string & str) {

	if (str.length() == 0) {
		return;
	}
	::std::string lerrz = " \t";

	size_t start2, start;

	start2 = str.find_first_not_of(lerrz);
	start = start2;
	size_t oldstart;
	do {

		str = str.substr(start);

		oldstart = start;
		start2 = str.find_first_not_of(lerrz);
		start = start2;

	} while (start != oldstart);

	start2 = str.find_last_not_of(lerrz);
	start = start2;
	if ((start != ::std::string::npos) && (start + 1 < str.length())) {
		str.erase(start + 1);
	}

}

template<typename vartype>
bool configstuff::readattributefromstring(vartype & variable,
		const ::std::string & contents, const ::std::string & attributename) {
	::std::istringstream filestr;
	filestr.str(contents);
	int totallength = contents.length();
	int curlength = 0;

	//::std::cout <<" file contents " << std::endl;
	//::std::cout << contents << std::endl<< std::endl<< std::endl;

	while ((filestr.good()) && (!filestr.fail()) && (!filestr.bad())
			&& (!filestr.eof()) && (curlength < totallength)) {
		::std::string line, nextline;

		int buflen = 3000;
		char buf[buflen + 1];
		filestr.getline(buf, buflen);
		buf[buflen] = '\0';
		line = buf;

		//getline(filestr, line);

		//::std::ostringstream temp;
		//temp<<filestr.gcount();
		//std::cout << "last read "<< temp.str() <<std::endl;
		curlength += (int) filestr.gcount();
		//curlength+=line.length()+endlinelength; // filestr.gcount() does not work whyever

		//::std::cout <<curlength <<"---"<< totallength <<std::endl;

		deblankbeginandend(line);

		//std::cout << attributename << " | vs " <<line <<std::endl;

		if (0 == line.compare(0, attributename.length(), attributename)) {
			getline(filestr, nextline);
			deblankbeginandend(nextline);
			std::istringstream holder;
			holder.str(nextline);
			holder >> variable;

			//filestr.close();
			return (true);
		}
	}

	return (false);

}

void configstuff::readconfig2(const std::string & configfile) {
	//haveconfigs=-1;
	filexists_check(configfile);

	{
		std::ifstream file;
		file.open(configfile.c_str());

		// get length of file:
		file.seekg(0, ::std::ios::end);
		int length = file.tellg();
		file.seekg(0, ::std::ios::beg);

		// allocate memory:
		char * buffer(NULL);
		buffer = new char[length + 1];

		// read data as a block:
		file.read(buffer, length);

		file.close();

		buffer[length] = '\0';

		::std::string str(buffer);
		delete[] buffer;
		buffer = NULL;

		std::ostringstream error;
		std::string attribute;

		//int tmpint = -1;
		std::string tmpstr;

		attribute = "param_file";
		if (false == readattributefromstring(param_file, str, attribute)) {
			error
					<< "generalparams::loadoptionsfromfile: failed to load attribute: "
					<< attribute << std::endl;
			std::cerr << error.str();
			exit(1);
		}

		if (!::boost::filesystem::is_regular_file(
				::boost::filesystem::path(param_file))) {
			std::cerr << "is not a regular file: " << param_file << std::endl;
			exit(1);
		}

		attribute = "model_file";
		if (false == readattributefromstring(model_file, str, attribute)) {
			error
					<< "generalparams::loadoptionsfromfile: failed to load attribute: "
					<< attribute << std::endl;
			std::cerr << error.str();
			exit(1);
		}

		if (!::boost::filesystem::is_regular_file(
				::boost::filesystem::path(model_file))) {
			std::cerr << "is not a regular file: " << model_file << std::endl;
			exit(1);
		}

		attribute = "mean_file";
		if (false == readattributefromstring(mean_file, str, attribute)) {
			error
					<< "generalparams::loadoptionsfromfile: failed to load attribute: "
					<< attribute << std::endl;
			std::cerr << error.str();
			exit(1);
		}

		if (!::boost::filesystem::is_regular_file(
				::boost::filesystem::path(mean_file))) {
			std::cerr << "is not a regular file: " << mean_file << std::endl;
			exit(1);
		}

		attribute = "use_mean_file_asbinaryprotoblob";
		if (false
				== readattributefromstring(use_mean_file_asbinaryprotoblob, str,
						attribute)) {
			error
					<< "generalparams::loadoptionsfromfile: failed to load attribute: "
					<< attribute << std::endl;
			std::cerr << error.str();
			exit(1);
		}
		//	int lastlayerindex;
		//int firstlayerindex;

		attribute = "lastlayerindex";
		if (false == readattributefromstring(lastlayerindex, str, attribute)) {
			error
					<< "generalparams::loadoptionsfromfile: failed to load attribute: "
					<< attribute << std::endl;
			std::cerr << error.str();
			exit(1);
		}

		attribute = "firstlayerindex";
		if (false == readattributefromstring(firstlayerindex, str, attribute)) {
			error
					<< "generalparams::loadoptionsfromfile: failed to load attribute: "
					<< attribute << std::endl;
			std::cerr << error.str();
			exit(1);
		}

		biastreatmenttype = 0;

		attribute = "synsetfile";
		if (false == readattributefromstring(synsetfile, str, attribute)) {
			error
					<< "generalparams::loadoptionsfromfile: failed to load attribute: "
					<< attribute << std::endl;
			std::cerr << error.str();
			exit(1);
		}

		if (!::boost::filesystem::is_regular_file(
				::boost::filesystem::path(synsetfile))) {
			std::cerr << "is not a regular file: " << synsetfile << std::endl;
			exit(1);
		}

		attribute = "baseimgsize";
		if (false == readattributefromstring(baseimgsize, str, attribute)) {
			error
					<< "generalparams::loadoptionsfromfile: failed to load attribute: "
					<< attribute << std::endl;
			std::cerr << error.str();
			exit(1);
		}

		attribute = "standalone_outpath";
		if (false
				== readattributefromstring(standalone_outpath, str,
						attribute)) {
			error
					<< "generalparams::loadoptionsfromfile: failed to load attribute: "
					<< attribute << std::endl;
			std::cerr << error.str();
			exit(1);
		}

		attribute = "standalone_rootpath";
		if (false
				== readattributefromstring(standalone_rootpath, str,
						attribute)) {
			error
					<< "generalparams::loadoptionsfromfile: failed to load attribute: "
					<< attribute << std::endl;
			std::cerr << error.str();
			exit(1);
		}

		attribute = "relpropformulatype";
		if (false
				== readattributefromstring(relpropformulatype, str,
						attribute)) {
			error
					<< "generalparams::loadoptionsfromfile: failed to load attribute: "
					<< attribute << std::endl;
			std::cerr << error.str();
			exit(1);
		}

		attribute = "epsstab";
		if (false == readattributefromstring(epsstab, str, attribute)) {
			error
					<< "generalparams::loadoptionsfromfile: failed to load attribute: "
					<< attribute << std::endl;
			std::cerr << error.str();
			exit(1);
		}

		attribute = "alphabeta_beta";
		if (false == readattributefromstring(alphabeta_beta, str, attribute)) {
			error
					<< "generalparams::loadoptionsfromfile: failed to load attribute: "
					<< attribute << std::endl;
			std::cerr << error.str();
			exit(1);
		}

		attribute = "numclasses";
		if (false == readattributefromstring(numclasses, str, attribute)) {
			error
					<< "generalparams::loadoptionsfromfile: failed to load attribute: "
					<< attribute << std::endl;
			std::cerr << error.str();
			exit(1);
		}

		/*
		 attribute = "maxpoolingtoavgpoolinginbackwardpass";
		 if (false
		 == readattributefromstring(maxpoolingtoavgpoolinginbackwardpass, str,
		 attribute)) {
		 error
		 << "generalparams::loadoptionsfromfile: failed to load attribute: "
		 << attribute << std::endl;
		 std::cerr << error.str();
		 exit(1);
		 }
		 */
		maxpoolingtoavgpoolinginbackwardpass = 0;

	}
}

// ************************************************

void saveimgasjpg(const std::string & file, const int hei, const int wid,
		const std::vector<std::vector<double> > & img) {

	Magick::Geometry fmt(wid, hei);

//Magick::Image *out=new Magick::Image(fmt,Magick::ColorRGB(0,0,0));

	Magick::Image out(fmt, Magick::ColorRGB(0.9, 0.9, 0.9));
//double largestvalue= pow(2.0, sizeof(Magick::Quantum)*8) -1;
	out.modifyImage();
	out.type(Magick::TrueColorType);

	Magick::PixelPacket *pixel, *pixel_cache = out.getPixels(0, 0,
			out.columns(), out.rows());

	for (unsigned int h = 0; h < out.rows(); ++h) {
		for (unsigned int w = 0; w < out.columns(); ++w) {
			pixel = &pixel_cache[w + out.columns() * h];

			*pixel = Magick::ColorRGB(img[0][h + out.rows() * w],
					img[1][h + out.rows() * w], img[2][h + out.rows() * w]);

		}
	}

	out.syncPixels();

//out.rotate(180);

	::boost::filesystem::path pt(
			std::string(file.substr(0, (int) file.length() - 4) + ".jpg"));
	if (!::boost::filesystem::exists(pt.branch_path())) {
		::boost::filesystem::create_directories(pt.branch_path());
	}

	out.compressType(MagickCore::JPEGCompression);
	out.write(pt.native());

}




void saveimgaspng(const std::string & file, const int hei, const int wid,
		const std::vector<std::vector<double> > & img) {

	Magick::Geometry fmt(wid, hei);

//Magick::Image *out=new Magick::Image(fmt,Magick::ColorRGB(0,0,0));

	Magick::Image out(fmt, Magick::ColorRGB(0.9, 0.9, 0.9));
//double largestvalue= pow(2.0, sizeof(Magick::Quantum)*8) -1;
	out.modifyImage();
	out.type(Magick::TrueColorType);

	Magick::PixelPacket *pixel, *pixel_cache = out.getPixels(0, 0,
			out.columns(), out.rows());

	for (unsigned int h = 0; h < out.rows(); ++h) {
		for (unsigned int w = 0; w < out.columns(); ++w) {
			pixel = &pixel_cache[w + out.columns() * h];

			*pixel = Magick::ColorRGB(img[0][h + out.rows() * w],
					img[1][h + out.rows() * w], img[2][h + out.rows() * w]);

		}
	}

	out.syncPixels();

//out.rotate(180);

	::boost::filesystem::path pt(
			std::string(file.substr(0, (int) file.length() - 4) + ".png"));
	if (!::boost::filesystem::exists(pt.branch_path())) {
		::boost::filesystem::create_directories(pt.branch_path());
	}

	out.compressType(MagickCore::NoCompression);
	out.write(pt.native());

}


// *******************************************************

int main(int argc, char ** argv) {

	if (argc - 1 != 3) {
		std::cout << "num arguments: " << argc - 1 << " needed 3" << std::endl;
		std::cout << "usage: ./this configfile imageLIST prependpath"
				<< std::endl;
		exit(1);
	}
	std::string configfile(argv[1]);
	std::string imageLIST(argv[2]);
	std::string prependpath(argv[3]);

	heatmaprunner hru;
	hru.init(configfile);

	// read file list

	::boost::filesystem::path pt(imageLIST);
	if (!::boost::filesystem::is_regular_file(pt)) {
		std::cerr << "file is no regular file " << imageLIST << std::endl;
		exit(1);
	}

	std::vector<std::string> flist;
	std::vector<int> classindstypesvec;

	{
		std::ifstream f;
		f.open(imageLIST.c_str());
		while (!(f.eof() || f.bad() || f.fail())) {
			std::string tmp, tmpfil;
			std::getline(f, tmp);
			if (tmp.length() > 2) {
				int classindstype = -1000;

				std::istringstream hlp;
				hlp.str(tmp);
				hlp >> tmpfil;
				hlp >> classindstype;

				//LOG(INFO)<<"tmp: "<< tmp;
				//LOG(INFO)<<"tmpfil: "<< tmpfil;

				if (classindstype == -1000) {
					LOG(FATAL) << "classindstype==-1000. failed to read it ";
				}
				classindstypesvec.push_back(classindstype);
				flist.push_back(prependpath + "/" + tmpfil);
			}

		}
		f.close();
	}

	for (size_t i = 0; i < flist.size(); ++i) {
		LOG(INFO) << "processing file: " << flist[i] << " classindstype: " << classindstypesvec[i];
		hru.process_heatmap(flist[i], classindstypesvec[i]);
		LOG(INFO) << "FINISHED processing file: " << flist[i];
	}
	std::cout << "finished" << std::endl;
	return 0;
}
