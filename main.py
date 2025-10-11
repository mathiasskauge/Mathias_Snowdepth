import sys
import os
from pathlib import Path
import tensorflow as tf
import numpy as np

import SnowDepth.data_loader as DL
import SnowDepth.data_splitter as DS
import SnowDepth.architecture as ARCH
import SnowDepth.visualization as VIZ
import SnowDepth.evaluation as EVAL

# Set seed
seed = 18

# Directory for TIF-data and H5 file
data_dir = "data"/"tif_files"
h5_dir = "data"/"h5_dir"
h5_path = h5_dir/"dataframe.h5"

# Create H5 file
if not h5_path.exists():
    DL.create_h5(data_dir, h5_dir, upper_threshold=3)
else:
    print(f"Using existing H5: {h5_path}")