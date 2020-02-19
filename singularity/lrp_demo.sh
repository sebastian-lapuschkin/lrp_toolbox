# singularity call to calculate heatmaps according to the config stored in the
# lrp_demo folder

singularity run -B lrp_demo:/opt/lrp_toolbox/caffe-master-lrp/demonstrator/lrp_demo_files/ caffe-lrp-cpu-u16.04.sif -c ./lrp_demo_files/config.txt -f ./lrp_demo_files/testfilelist.txt -p ./lrp_demo_files
