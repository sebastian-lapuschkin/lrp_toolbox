# The LRP Toolbox for Keras

This is forked repository from [sebastian-lapuschkin/lrp_toolbox](https://github.com/sebastian-lapuschkin/lrp_toolbox).

## Train network models

### With CPUs

```sh
docker-compose run cpu python python/train_mnist.py
```

### With GPUs

```sh
docker-compose -f docker-compose-gpu.yml run gpu python python/train_mnist.py
```

### Visualize the model by the LRP

```sh
docker-compose up
# then, access to localhost:8888 through the browser
```

# LICENSE

This project includes python modules from [sebastian-lapuschkin/lrp_toolbox](https://github.com/sebastian-lapuschkin/lrp_toolbox).

It's required to follow both licenses for [this project](https://github.com/yakigac/lrp_toolbox_keras/blob/master/LICENSE) and [the license](https://github.com/sebastian-lapuschkin/lrp_toolbox/blob/master/python/LICENSE) of the original project.

### The LRP Toolbox Paper

When using any part of this toolbox, please cite [the original paper](http://jmlr.org/papers/volume17/15-618/15-618.pdf) to follow the original license.

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

### Misc

For further research and projects involving LRP, visit [heatmapping.org](http://heatmapping.org).
