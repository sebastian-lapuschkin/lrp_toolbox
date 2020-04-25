#!/bin/bash
#
# Install requirements and run demo code.
# An optional installation of cupy (enables GPU support, requires CUDA) is recommended via (ana/mini)conda
pip install -r requirements.txt &&
python lrp_demo.py

