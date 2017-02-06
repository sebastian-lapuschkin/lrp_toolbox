#!/bin/bash

# 1) GO TO CAFFE ROOT
# 2) RUN THIS
# 3) BUILD CAFFE


# transformations :
# #  #include "hdf5/serial/hdf5.h" -> #include "hdf5/serial/hdf5.h"
# #  #include "hdf5/serial/hdf5_hl.h" -> #include "hdf5/serial/hdf5_hl.h"    #sadly using sed destroys this script , at the same time makes it safe to use it repeatedly
#  links hdf5
#  modifies makefile.


distro="$( lsb_release -a | grep Release: )"
if [[ "$distro" == *16.04 ]] ||  [[ "$distro" == *15.04 ]]
then	
	if [ -f distro.patched ]
	then
		echo "nothing to patch."
	else


		echo "running patch."
		echo "step 1: manipulate header paths for hdf5 "
		find . -type f -exec sed -i -e 's^"hdf5/serial/hdf5.h"^"hdf5/serial/hdf5.h"^g' -e 's^"hdf5/serial/hdf5_hl.h"^"hdf5/serial/hdf5_hl.h"^g' '{}' \;

		echo "step 2: create symlinks for hdf5 libraries. please enter your super user pw on prompt "
		here=$PWD
		cd /usr/lib/x86_64-linux-gnu
		sudo ln -s libhdf5_serial_hl.so libhdf5_hl.so
		sudo ln -s libhdf5_serial.so libhdf5.so
		cd $here

		echo "step 3: modify makefile to include hdf5 dirs"
		sed -i -e "/INCLUDE_DIRS :=/a INCLUDE_DIRS += \/usr\/include\/hdf5\/serial\/" Makefile.config

		echo "step 4: modify imagemagick include paths and library names for demonstrator"
		cd demonstrator
		sed -i -e "s/-lMagick++/-lMagick++-6.Q16/g" -e "s/-lMagickWand/-lMagickWand-6.Q16/g" -e "s/-lMagickCore/-lMagickCore-6.Q16/g" -e "s|-I /usr/include/ImageMagick/|-I /usr/include/ImageMagick-6 -I /usr/include/x86_64-linux-gnu/ImageMagick-6/|g" build.sh 
		cd ..
		touch distro.patched
	fi
fi




# DO THE FOLLOWING MANUALLY FOR NOW




#modify INCLUDE_DIRS in Makefile.config
#   INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/

# replace in build.sh:
#  -lMagick++ -lMagickWand -lMagickCore  -> -lMagick++-6.Q16 -lMagickWand-6.Q16 -lMagickCore-6.Q16#
#  -I /usr/include/ImageMagick -> -I /usr/include/ImageMagick-6/ -I /usr/include/x86_64-linux-gnu/ImageMagick-6/


# this works. nice.
