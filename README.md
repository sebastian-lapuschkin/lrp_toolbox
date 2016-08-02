# The LRP Toolbox for Artificial Neural Networks

The Layer-wise Relevance Propagation (LRP) algorithm explains a classifer's prediction
specific  to  a  given  data  point  by  attributing
relevance scores to  important  components
of  the  input  by  using  the  topology  of  the  learned  model  itself.

The LRP Toolbox provides simple and accessible stand-alone implementations of LRP for artificial neural networks supporting Matlab and python. The Toolbox realizes LRP functionality for the Caffe Deep Learning Framework as an extension of Caffe source code published in 10/2015.

The  implementations  for  Matlab  and  python  shall  serve  as  a  playing field to familiarize oneself with the LRP algorithm and are implemented with readability and transparency in mind.  Models and data can be imported and exported using raw text formats, Matlab's .mat files and the .npy format for python/numpy.

<img src="doc/images/1.png" width="280">
<img src="doc/images/2.png" width="280">
<img src="doc/images/7.png" width="280">

<img src="doc/images/volcano2.jpg" width="210"><img src="doc/images/volcano2_hm.jpg" width="210">
<img src="doc/images/scooter10.jpg" width="210"><img src="doc/images/scooter_10_hm.jpg" width="210">



### Installing the Toolbox -- TL;DR version:

For whichever language / purpose you wish to make use of this toolbox download the appropriate sub-package (*python*, *matlab*, *caffe-master-lrp* -- or do a full clone of the project) and then just run the installation script for your implementation of choice, e.g.

    git clone https://github.com/sebastian-lapuschkin/lrp_toolbox/
    cd lrp_toolbox/$yourChoice
    bash install.sh

Make sure to at least skim through the installation scripts! For more details and instructions please refer to [the manual](https://github.com/sebastian-lapuschkin/lrp_toolbox/blob/master/manual.pdf).

### The LRP Toolbox Paper

When using (any part) of this toolbox, please cite [our (recently accepted) paper](BROKEN LINK)(link TBA)

    @article{lapuschkin2016LRP,
      title={The LRP Toolbox for Artificial Neural Networks},
      author={Lapuschkin, Sebastian and Binder, Alexander and Montavon, Gr\'{e}goire and M\"{u}ller, Klaus-Robert and Samek, Wojciech },
      journal={The Journal of Machine Learning Research},
      year={2016},
      volume={?17?},
      number={????}
      pages={????},
      publisher={JMLR. org}
    }
    
### Misc

For further research and projects involving LRP, visit [heatmapping.org (WIP. the ugly will vanish soon.)](http://heatmapping.org)

