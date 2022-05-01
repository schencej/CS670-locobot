#!/usr/bin/env python

import rospy
import asyncio
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import BoundingBox2D
from cv_bridge import CvBridge
import pandas
import torch
import numpy as np
import image_geometry
from interbotix_xs_modules.locobot import InterbotixLocobotXS
import time
import os
import statistics
import sys

goal_key = {'chair': (56, "large"), 'pen':(39, "small"), 'bottle':(39, "small"), 'person':(0, "large"), 'cup':(41, "small"), 'bowl':(45, "small"), 
'bench':(13, "large"), 'refri':(72, "large")}



class oject_cand:
    def __init__(self):
        self.x=None
        self.y=None
        self.z=None
        self.save_flag=1
        self.Cam_info = rospy.wait_for_message('/locobot/camera/aligned_depth_to_color/camera_info', CameraInfo)
        self.camera_pos=[0,0]

    def grand_value(self, coor_list, d_img):
        xmin,ymin,xmax,ymax=coor_list[0],coor_list[1],coor_list[2],coor_list[3]

        center_x=round((xmin+xmax)/2)
        center_y=round((ymin+ymax)/2)
        

        depth_group=d_img[round(ymin.item()):round(ymax.item()), round(xmin.item()):round(xmax.item())]

        depth_group_flatten = depth_group.flatten()
        depth_group_flatten = np.array([i for i in depth_group_flatten if i != 0])
        tmp = depth_group_flatten[abs(depth_group_flatten - np.mean(depth_group_flatten)) < 1.5 * np.std(depth_group_flatten)]


        #depth=min(tmp)
        depth = statistics.median(tmp)
        ig_p = image_geometry.PinholeCameraModel()
        ig_p.fromCameraInfo(self.Cam_info)
        pt3d = list(ig_p.projectPixelTo3dRay((center_x,center_y)))
        pt3d[:] = [a/pt3d[2] for a in pt3d]
        final_point = [el * depth for el in pt3d]
        self.x, self.y, self.z = round(final_point[0],1), round(final_point[1],1), round(final_point[2],1)

    def show_coor(self):
        from std_msgs.msg import String
        pub = rospy.Publisher("/coordinates_topic", String, queue_size=10)
        pub.publish("X: " + str(self.x)+"   Y: " + str(self.y) + "   Z: "+ str(self.z) + " | Camera position(Pan, Tilt): " + str(self.camera_pos[0]) + str(self.camera_pos[1]))

        filename = "coor/coordiantes"
        with open(filename, 'w+') as my_info:
            my_info.write("%s %s %s | %s %s\n" % (self.x, self.y, self.z, self.camera_pos[0], self.camera_pos[1]))

        #await asyncio.sleep(1.0)
        #print(self.x, self.y, self.z)

    """
    def move_it(self):
        locobot = InterbotixLocobotXS(robot_model="locobot_wx250s", arm_model="mobile_wx250s", use_move_base_action=True)
        locobot.base.move_to_pose(self.x, self.y, (self.z-200)/1000, True)
    """


class Img:
    def __init__(self):
        self.rgb_img = None
        self.depth_img = None
        self.coordinates= [] #float format
        self.bridge = CvBridge()
        self.flag_xy = 1
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) #device='cpu'
        self.sub=None
        self.oc = oject_cand()
        #self.Cam_info = CameraInfo()
        #self.cameraModel.fromCameraInfo(self.cameraInfo)

    def rgb_callback(self, img_msg):
        # "Store" message received.
        tmp = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        self.rgb_img = tmp
        self.publish_detections()
        
        

    def depth_callback(self, img_msg):
        # "Store" the message received.
        tmp = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        self.depth_img = tmp
        #print("coor:" , self.coordinates)
        #format: xmin,ymin,xmax,ymax,confidence,class,name
        #generate center point data.
        #oc=oject_cand()
        if self.coordinates:
            #print(tmp.shape)
            self.oc.grand_value(self.coordinates, self.depth_img)
        self.oc.show_coor()
        self.flag_xy=1



    def publish_detections(self):
        #torch.cuda.is_available = lambda : False
        dets = self.model(self.rgb_img)
        result = dets.xyxy[0]
        #print(dets.pandas().xyxy[0])
        if self.flag_xy==1:
            for *xyxy, conf, obcls in result:
                if obcls == ok:
                    #56 is chair, 39 is bottle
                    self.coordinates=[]
                    for xy in xyxy:
                        self.coordinates.append(xy.numpy())
            if self.coordinates:
                self.flag_xy=0


        dets.render()
        viz = dets.imgs[0]
    
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.height = viz.shape[0]
        msg.width = viz.shape[1]
        msg.encoding = 'rgb8'
        msg.is_bigendian = False
        msg.step = 3 * viz.shape[1]
        msg.data = viz.tobytes()
        pub.publish(msg)
        #return xynum
    """
    def run(self):
        rospy.init_node('depth_and_go', anonymous=True)
        self.sub = rospy.Subscriber("/locobot/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
    """






def detector():
    global model, pub

    #rospy.init_node('pixel2depth', anonymous=True)
    
    img=Img()
    

    rospy.Subscriber("/locobot/camera/color/image_raw", Image, img.rgb_callback)
    rospy.Subscriber("/locobot/camera/aligned_depth_to_color/image_raw", Image, img.depth_callback)
    #img.run()
    #rospy.Subscriber("/locobot/camera/aligned_depth_to_color/camera_info", CameraInfo, img.Camera_info_callback)
    pub = rospy.Publisher('/locobot/camera/detections/viz', Image, queue_size=1)
    
    rospy.spin()

def coor_detections(img_msg):
    result=[]
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding='32FC1')
    cv_image_array = np.array(cv_image, dtype = np.float32)
    #cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
    #cv_image_resized = cv2.resize(cv_image_norm, self.desired_shape, interpolation = cv2.INTER_CUBIC)
    #cv2.imshow("Image from my node", cv_image_array)
    #cv2.waitKey(1)
    print(cv_image.shape)
    a=cv_image_array[240,320]/1000
    print(a)

    """
    flag_xy =1
    if flag_xy==1:
        locobot = InterbotixLocobotXS(robot_model="locobot_wx250s", arm_model="mobile_wx250s", use_move_base_action=True)
        locobot.base.move_to_pose(a, 0, 0, True)
        flag_xy=0
    """
if __name__ == '__main__':
    global ok
    filename = "coor/coordiantes"
    if os.path.exists(filename):
        os.remove(filename)
    locobot = InterbotixLocobotXS(robot_model="locobot_wx250s", arm_model="mobile_wx250s")
    locobot.camera.pan_tilt_go_home()


    goal = sys.argv[1].lower()
    ok, o_size = goal_key[goal]


    detector()
