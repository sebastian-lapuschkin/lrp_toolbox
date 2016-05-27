#!/bin/bash

#adapt your include and library paths when necessary!

echo building sequential heatmap computation
g++ compute_heatmaps.cpp -o lrp_demo -I ../include/ -I ../build/src/  -I /usr/include/ImageMagick/ -I /usr/include/opencv/  -L /usr/lib/x86_64-linux-gnu/  -Wl,--whole-archive  ../build/lib/libcaffe.a  -Wl,--no-whole-archive -lpthread -lglog -lgflags -lprotobuf -lleveldb -lsnappy -lboost_system -lhdf5_hl -lhdf5 -llmdb -lopencv_core -lopencv_highgui -lopencv_imgproc -latlas -lcblas  -Wall -lMagick++ -lMagickWand -lMagickCore -lboost_filesystem -lboost_system -lboost_thread  -lpthread  -DCPU_ONLY

echo building parallalel heatmap computation
g++ compute_heatmaps_parallel.cpp -o lrp_parallel_demo -I ../include/ -I ../build/src/  -I /usr/include/ImageMagick/ -I /usr/include/opencv/  -L /usr/lib/x86_64-linux-gnu/ -Wl,--whole-archive ../build/lib/libcaffe.a  -Wl,--no-whole-archive -lpthread -lglog -lgflags -lprotobuf -lleveldb -lsnappy -lboost_system -lhdf5_hl -lhdf5 -llmdb -lopencv_core -lopencv_highgui -lopencv_imgproc -latlas -lcblas  -Wall -lMagick++ -lMagickWand -lMagickCore -lboost_filesystem -lboost_system -lboost_thread  -lpthread  -DCPU_ONLY
