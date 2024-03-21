# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import stat
import sys

# find the import for catkin's python package - either from source space or from an installed underlay
if os.path.exists(os.path.join('C:/Program Files/MATLAB/R2023b/sys/ros1/win64/ros1/share/catkin/cmake', 'catkinConfig.cmake.in')):
    sys.path.insert(0, os.path.join('C:/Program Files/MATLAB/R2023b/sys/ros1/win64/ros1/share/catkin/cmake', '..', 'python'))
try:
    from catkin.environment_cache import generate_environment_script
except ImportError:
    # search for catkin package in all workspaces and prepend to path
    for workspace in 'C:/Program Files/MATLAB/R2023b/sys/ros1/win64/ros1;C:/Program Files/MATLAB/R2023b/toolbox/ros/mlroscpp/custom_messages'.split(';'):
        python_path = os.path.join(workspace, 'lib/site-packages')
        if os.path.isdir(os.path.join(python_path, 'catkin')):
            sys.path.insert(0, python_path)
            break
    from catkin.environment_cache import generate_environment_script

code = generate_environment_script('Z:/matlab_msg_gen_ros1/win64/devel/env.bat')

output_filename = 'Z:/matlab_msg_gen_ros1/win64/build/catkin_generated/setup_cached.bat'
with open(output_filename, 'w', encoding='utf-8') as f:
    # print('Generate script for cached setup "%s"' % output_filename)
    f.write('\n'.join(code))

mode = os.stat(output_filename).st_mode
os.chmod(output_filename, mode | stat.S_IXUSR)
