# CUDA utilized costmap_2d
`costmap_2d` is an ROS package in its navigation stack. In order to boost performance, three of its plugins are modified to use make use of CUDA for parallel processing.

This is also the source code of work mentioned in the paper `CUDA-based parallel optimization of ROS costmap` submitted to Computing Frontiers 2019. We are here to make source code of modified `costmap_2d` and test dataset public.

## How to use this package
* Make sure CUDA Toolkit and other necessary softwares are installed.
* Override the original version of `costmap_2d` by following the instructions from [Overlaying with catkin workspaces](http://wiki.ros.org/catkin/Tutorials/workspace_overlaying).  
If multiple workspaces are in use, consider [chaining catkin workspaces](http://wiki.ros.org/catkin/Tutorials/workspace_overlaying#Chaining_catkin_workspaces).
* Manually compile this workspace. If all environments are corretly configured, it should be working properly.

## Requirements
* A computer running Linux and ROS  
*Only tested in Linux Ubuntu 16.04 LTS and ROS Kinetic Kame.*
* GPU that supports CUDA acceleration  
CUDA Toolkit 8.0 is recommended.

## Test data and base map
*Currently looking for approriate place for public access. Link will be offered later.*
