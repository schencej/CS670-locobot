#!/usr/bin/env python

import math
from interbotix_xs_modules.locobot import InterbotixLocobotXS
import rospy
from std_msgs.msg import String
import os



#arm_base = [38.7, 19.5, 462.0]


def forward_cal(dep, height, ro):
    #a = math.sqrt(dep**2 - height**2)
    return dep/1000 + abs(ro)/10

def rotation_cal(x, z):
    ro = math.atan2(x-30, z)
    ro = -ro #align with ros method
    return ro

def main():
    #arm_home_pose = [38.7, -40, 450.0]
    #arm_base = [38.7, 453, 0]
    #locobot = InterbotixLocobotXS(robot_model="locobot_wx250s", arm_model="mobile_wx250s")

    locobot.arm.go_to_sleep_pose()


    locobot.arm.set_ee_pose_components(x=0.3, z=0.37)

    locobot.gripper.open()
    #locobot.arm.set_ee_pose_components(x=0.3, z=0.37, y=-(coor[0] - arm_home_pose[0])/1000)
    
    locobot.arm.set_single_joint_position("waist", rotation)
    fc = forward_cal(coor[2], coor[1], rotation)
    locobot.arm.set_ee_cartesian_trajectory(x=fc-0.3, z=0.43-coor[1]/1000 - 0.37) # candi

    #locobot.arm.set_ee_cartesian_trajectory(x=coor[2]/1000-0.3, z=0.43-coor[1]/1000 - 0.37) # candi



    locobot.gripper.close()

    locobot.arm.set_ee_pose_components(x=0.2, z=0.37) #go back
    #locobot.arm.set_ee_pose_components(x=0.3, z=0.2)
    #locobot.arm.set_single_joint_position("waist", math.pi/4.0)
    locobot.arm.set_ee_cartesian_trajectory(x=0.1, z=-0.32)

    locobot.gripper.open()

 
    locobot.arm.go_to_sleep_pose()

if __name__=='__main__':
    global coor, rotation, locobot
    ##"X: -102.7   Y: 18.7   Z: 607.0"
    #z limit 570
    #coor = [-102.7, 18.7, 607.0]
    #coor= [-96.0, 19.0 , 468.0]

    locobot = InterbotixLocobotXS(robot_model="locobot_wx250s", arm_model="mobile_wx250s")

    #read camera position state while doing the detection
    #lcg = locobot.pan_position()
    #print(lcg)

    #read coordiantes from pre-defined file
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
    #print(coor, camera_pos)
    #end of read coordinates



    rotation = rotation_cal(coor[0],coor[2])
    
    print(coor, camera_pos, rotation)




    main()