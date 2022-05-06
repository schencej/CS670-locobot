#!/bin/bash

roslaunch interbotix_xslocobot_gazebo xslocobot_gazebo.launch \
  robot_model:=locobot_wx250s\
  show_lidar:=true\
  use_trajectory_controllers:=true\
  #world_name:=/usr/share/gazebo-11/worlds/stacks.world\
  /locobot/camera/depth/image_raw:=/locobot/rtabmap/rgbd_image
