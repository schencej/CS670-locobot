#! /usr/bin/env python
# move robot with build-in path planning
from interbotix_xs_modules.locobot import InterbotixLocobotXS
import os
import math


def rotation_cal(x, z):
    ro = math.atan2(x-30, z)
    ro = -ro #align with ros method
    return ro


global fw, ya, tu

filename = "coor/coordiantes"
if os.path.exists(filename):
    with open(filename, 'r') as my_info:
        last_line = my_info.readlines()[-1]
else:
    print("Missing coordiantes file!")
    exit()
head, tail = last_line.split("|")
coor = head.split( )
coor = list(map(float, coor))

camera_pos = tail.split( )
camera_pos = list(map(int, camera_pos))

fw = coor[2]/1000 - 0.31
tu = -coor[0]/1000 #align with ros method
ya = rotation_cal(coor[0],coor[2])


def main():
    locobot = InterbotixLocobotXS(robot_model="locobot_wx250s", arm_model="mobile_wx250s", use_move_base_action=True)
    locobot.camera.pan_tilt_go_home()
    locobot.camera.tilt(0.3)

    locobot.base.move_to_pose(fw, tu, 0, True)
    

if __name__=='__main__':
    main()
