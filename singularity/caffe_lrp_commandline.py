"""
This script catches the parameters passed to the runnable singularity container from the outside and
translates them into command line inputs for the caffe binaries living inside the container.
"""

import sys
import os
import argparse

#list of callable lrp binaries and names
binaries = {'default':           '/opt/lrp_toolbox/caffe-master-lrp/demonstrator/lrp_demo',
            'default_minimal':   '/opt/lrp_toolbox/caffe-master-lrp/demonstrator/lrp_demo_minimal_output',
            'parallel':          '/opt/lrp_toolbox/caffe-master-lrp/demonstrator/lrp_parallel_demo',
            'parallel_minimal':  '/opt/lrp_toolbox/caffe-master-lrp/demonstrator/lrp_parallel_demo_minimal_output'
           }

#prepare command line arguments
parser = argparse.ArgumentParser(description="Process models and data, produce explanatory heatmaps")
parser.add_argument('--binary',   '-b', default='default', help='Which binary to use for computing heatmaps? Determines asynchronicity and amount of meta-information.')
parser.add_argument('--config',   '-c', required=True,     help='Path to the lrp demonstrator configuration file')
parser.add_argument('--filelist', '-f', required=True,     help='Path to the list-of-files-file with pairs of img_path class_to_explain in each line')
parser.add_argument('--prepath',  '-p', default='./',       help='Path segment to preprend to the generated output files. "./" is the default value')
args = parser.parse_args()

assert args.binary in binaries, "Key '{}' is not valid. Valid binary keys are: {}".format(args.binary, ', '.join(["'{}'".format(b) for b in binaries.keys()]))

#now, forward the call.
cmd = '{} {} {} {}'.format( binaries[args.binary],
                                args.config,
                                args.filelist,
                                args.prepath
                           )
print("executing: '{}'".format(cmd))
os.system(cmd)
