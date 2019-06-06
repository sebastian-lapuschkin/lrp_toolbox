# The LRP Toolbox for Artificial Neural Networks (1.3.0rc2)

The Layer-wise Relevance Propagation (LRP) algorithm explains a classifer's prediction
specific  to  a  given  data  point  by  attributing
relevance scores to  important  components
of  the  input  by  using  the  topology  of  the  learned  model  itself.

The LRP Toolbox provides simple and accessible stand-alone implementations of LRP for artificial neural networks supporting Matlab and python. The Toolbox realizes LRP functionality for the Caffe Deep Learning Framework as an extension of Caffe source code published in 10/2015.

The  implementations  for  Matlab  and  python  are intended as a sandbox or playground to familiarize the user to the LRP algorithm and  thus are implemented with readability and transparency in mind.  Models and data can be imported and exported using raw text formats, Matlab's .mat files and the .npy format for python/numpy.

<img src="doc/images/1.png" width="280"><img src="doc/images/2.png" width="280"><img src="doc/images/7.png" width="280">

<img src="doc/images/volcano2.jpg" width="210"><img src="doc/images/volcano2_hm.jpg" width="210">
<img src="doc/images/scooter10.jpg" width="210"><img src="doc/images/scooter_10_hm.jpg" width="210">


### See the LRP Toolbox in Action
To try out either the python-based MNIST demo, or the Caffe based ImageNet demo in your browser, click on the respective panels:

[<img src="http://heatmapping.org/images/mnist.png" width=210>](http://heatmapping.org/mnist.html)
[<img src="http://heatmapping.org/images/caffe.png" width=210>](http://heatmapping.org/caffe.html)


### New in 1.3.0rc2:
#### standalone python implementation:
* update to python 3
* update treatment of softmax and target class
* lrp_aware option for efficient calculation of multiple backward passes
* custom colormaps in render.py
* __gpu support__ when installing [cupy](https://github.com/cupy/cupy). this is an optional feature. without the cupy package, the code will execute using numpy.

### caffe implementation
* update the installation config
* new formula types 100, 102, 104
* new python wrapper to use lrp in pycaffe
* pycaffe demo file
* bugfixes


### New in version 1.2.0
#### The standalone implementations for python and Matlab:
* Convnets with Sum- and Maxpooling are now supported, including demo code.
* LRP-parameters can now be set for each layer individually
* w² and flat weight decomposition implemented.



#### Caffe:
* Minimal output versions implemented.
* Matthew Zeiler et al.'s  Deconvolution, Karen Simonyan et al.'s Sensitivity Maps, and aspects of Grégoire Montavon et al.'s Deep Taylor Decomposition are implemented, alongside the flat weight decomposition for uniformly projecting relevance scores to a neuron's receptive field have been implemented.

#### Also:
* Various optimizations, refactoring, bits and pieces here and there.



### Obtaining the LRP Toolbox:
You can directly download the latest full release / current verson from github. However, if you prefer to only download what is necessary for your project/language/purpose, make use of the pre-packaged downloads available at [heatmapping.org](http://www.heatmapping.org/)


### Installing the Toolbox:

After having obtained the toolbox code, data and models of choice, simply move into the subpackage folder of you choice -- matlab, python or caffe-master-lrp -- and execute the installation script (written for Ubuntu 14.04 or newer). 

    <obtain the toolbox>
    cd lrp_toolbox/$yourChoice
    bash install.sh

Make sure to at least skim through the installation scripts! For more details and instructions please refer to [the manual](https://github.com/sebastian-lapuschkin/lrp_toolbox/blob/master/doc/manual/manual.pdf).

### The LRP Toolbox Paper

When using (any part) of this toolbox, please cite [our paper](http://jmlr.org/papers/volume17/15-618/15-618.pdf)

    @article{JMLR:v17:15-618,
        author  = {Sebastian Lapuschkin and Alexander Binder and Gr{{\'e}}goire Montavon and Klaus-Robert M{{{\"u}}}ller and Wojciech Samek},
        title   = {The LRP Toolbox for Artificial Neural Networks},
        journal = {Journal of Machine Learning Research},
        year    = {2016},
        volume  = {17},
        number  = {114},
        pages   = {1-5},
        url     = {http://jmlr.org/papers/v17/15-618.html}
    }


    
### Misc & Related

For further research and projects involving LRP, visit [heatmapping.org](http://heatmapping.org)

Also, consider paying https://github.com/albermax/innvestigate a visit! Next to LRP, iNNvestigate efficiently implements a hand full of additional DNN analysis methods and can boast with a >500-fold increase in computation speed when compared with our CPU-bound Caffe implementation! 

