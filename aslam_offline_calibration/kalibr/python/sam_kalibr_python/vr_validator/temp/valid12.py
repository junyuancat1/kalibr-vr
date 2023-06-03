#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from PIL import Image
import cv2
import sm
import aslam_cv as acv
import aslam_cv_backend as acvb
import aslam_cameras_april as acv_april
import kalibr_common as kc

import os
import time
import numpy as np
import pylab as pl
import argparse
import sys
import getopt
import igraph

# make numpy print prettier
np.set_printoptions(suppress=True)

class VRCamera:
    def __init__(self, Size, CameraMatrix: np.ndarray, DistortMatrix: np.ndarray, Extrinsics_R: np.ndarray, Extrinsics_T: np.ndarray):
        self.size = Size
        self.K = CameraMatrix
        self.D = DistortMatrix
        self.E_R = Extrinsics_R
        self.E_T = Extrinsics_T

def distortPoints(undistortedPoints, k, d):
    
    undistorted = np.float32(undistortedPoints[:, np.newaxis, :])

    kInv = np.linalg.inv(k)

    for i in range(len(undistorted)):
        srcv = np.array([undistorted[i][0][0], undistorted[i][0][1], 1])
        dstv = kInv.dot(srcv)
        undistorted[i][0][0] = dstv[0]
        undistorted[i][0][1] = dstv[1]

    distorted = cv2.fisheye.distortPoints(undistorted, k, d)
    return distorted    


class AprilBox:
    # def __init__(self, markerID, top_left, top_right, bottom_left, bottom_right):
    def __init__(self, markerID):
        self.markerId = markerID
        # self.top_left = top_left
        # self.top_right = top_right
        # self.bottom_left = bottom_left
        # self.bottom_right = bottom_right
        self.boardId = int(markerID / 36) + 1

        self.x_line_pos = 0
        self.y_line_pos = 0
        self.z_line_pos = 0
        self.bottom_left3d = [0. ,0., 0.]
        self.bottom_right3d = [0. ,0., 0.]
        self.top_left3d = [0. ,0., 0.]
        self.top_right3d = [0. ,0., 0.]
        if self.boardId == 1:
            self.x_line_pos = markerID % 6
            self.y_line_pos = int(markerID / 6) - 6 * (self.boardId -1)
            self.bottom_left3d = [self.x_line_pos * 0.072 * (1 + 0.2),  self.y_line_pos * 0.072 * (1 + 0.2) , 0.]
            self.bottom_right3d = [self.bottom_left3d[0] + 0.072,  self.y_line_pos * 0.072 * (1 + 0.2) , 0.]
            self.top_left3d = [self.x_line_pos * 0.072 * (1 + 0.2),  self.bottom_left3d[1] + 0.072 , 0.]
            self.top_right3d = [self.bottom_left3d[0] + 0.072,  self.bottom_left3d[1] + 0.072 , 0.]
        elif self.boardId == 2:
            self.y_line_pos = markerID % 6
            self.z_line_pos = int(markerID / 6) - 6 * (self.boardId -1)
            self.bottom_left3d = [0., self.y_line_pos * 0.072 * (1 + 0.2),  self.z_line_pos * 0.072 * (1 + 0.2)]
            self.bottom_right3d = [0., self.bottom_left3d[1] + 0.072,  self.bottom_left3d[2]]
            self.top_left3d = [0., self.bottom_left3d[1],  self.bottom_left3d[2] + 0.072]
            self.top_right3d = [0., self.bottom_left3d[1] + 0.072,  self.bottom_left3d[2] + 0.072]
        elif self.boardId == 3:
            self.z_line_pos = markerID % 6
            self.x_line_pos = int(markerID / 6) - 6 * (self.boardId -1)
            self.bottom_left3d = [self.x_line_pos * 0.072 * (1 + 0.2),  0., self.z_line_pos * 0.072 * (1 + 0.2)]
            self.bottom_right3d = [self.bottom_left3d[0], 0., self.bottom_left3d[2] + 0.072]
            self.top_left3d = [self.bottom_left3d[0] + 0.072, 0., self.bottom_left3d[2]]
            self.top_right3d = [self.bottom_left3d[0] + 0.072, 0., self.bottom_left3d[2] + 0.072]
        else:
            print("WTF? ", markerID, self.boardId)

        # print("image tagid {}, top_left{}, top_right{}, bottom_left{}, bottom_right{}".format(self.markerId, self.top_left, self.top_right, self.bottom_left, self.bottom_right))
        # print("world tagid {}, top_left{}, top_right{}, bottom_left{}, bottom_right{}".format(self.markerId, self.top_left3d, self.top_right3d, self.bottom_left3d, self.bottom_right3d))

class CalibrationTargetDetector(object):
    def __init__(self, camera, targetConfig):
        targetParams = targetConfig.getTargetParams()
        targetType = targetConfig.getTargetType()
        if( targetType == 'aprilgrid' ):
            # print("detection start id = ", targetParams['tagStartId'])
            # print("detection end id = ", targetParams['tagEndId'])
            grid = acv_april.GridCalibrationTargetAprilgrid(targetParams['tagRows'], 
                                                            targetParams['tagCols'], 
                                                            targetParams['tagSize'], 
                                                            targetParams['tagSpacing'],
                                                            targetParams['tagStartId'],
                                                            targetParams['tagEndId'])
        else:
            raise RuntimeError( "Unknown calibration target." )
        
        #setup detector
        options = acv.GridDetectorOptions() 
        options.filterCornerOutliers = True
        self.detector = acv.GridDetector(camera.geometry, grid, options)
  
if __name__ == "__main__": 
    
    vr_camera1_K = np.array([[276.28156, 0, 317.99796], [0, 276.28156, 240.97645], [0, 0, 1]])
    vr_camera1_D = np.array([-0.013057856, 0.026553879, -0.01489986, 0.0031719355])
    vr_camera1_R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vr_camera1_T = np.array([0, 0, 0])
    vr_camera_trackingA = VRCamera((640, 480), vr_camera1_K, vr_camera1_D, vr_camera1_R, vr_camera1_T)

    vr_camera2_K = np.array([[276.44363, 0, 318.01495], [0, 276.44363, 240.67293], [0, 0, 1]])
    vr_camera2_D = np.array([0.0051421098, -0.0088336899, 0.0097894818, -0.0032491733])
    vr_camera2_R = np.array([[0.99949765, 0.029883759, 0.010555321], [-0.025876258, -0.96176534, 0.27264969], [0.018299539, 0.27223959, 0.96205547]])
    vr_camera2_T = np.array([-0.00038707237, 0.10318894, -0.01401102])
    vr_camera_trackingB = VRCamera((640, 480), vr_camera2_K, vr_camera2_D, vr_camera2_R, vr_camera2_T)

    vr_camera3_K = np.array([[273.87003, 0, 317.51742], [0, 273.87003, 238.06963], [0, 0, 1]])
    vr_camera3_D = np.array([0.006437201, -0.015906533, 0.021496724, -0.0065812969])
    vr_camera3_R = np.array([[0.5687224, -0.3726157, 0.73328874], [0.81622218, 0.36585493, -0.44713703], [-0.10166702, 0.85282338, 0.51220709]])
    vr_camera3_T = np.array([-0.0056250621, 0.041670205, -0.013388009])
    vr_camera_control_trackingA = VRCamera((640, 480), vr_camera3_K, vr_camera3_D, vr_camera3_R, vr_camera3_T)

    vr_camera4_K = np.array([[277.06611, 0, 317.67797], [0, 277.06611, 238.94741], [0, 0, 1]])
    vr_camera4_D = np.array([0.026178562, -0.053109878, 0.041701285, -0.010932579])
    vr_camera4_R = np.array([[-0.59714752, -0.35174627, 0.72089486], [0.79234879, -0.11873178, 0.598403], [-0.12489289, 0.92853504, 0.34960612]])
    vr_camera4_T = np.array([-0.079096205, 0.064828795, -0.066688745])
    vr_camera_control_trackingB = VRCamera((640, 480), vr_camera4_K, vr_camera4_D, vr_camera4_R, vr_camera4_T)
    
    vr_camera1_rvec_2_world = np.array([[-1.36089291], [2.74299731], [-0.0365315]])
    vr_camera1_tvec_2_world = np.array([[0.21391437], [0.1228164], [0.30720318]])
    vr_camera1_R_2_world, _ = cv2.Rodrigues(vr_camera1_rvec_2_world)
    vr_camera1_T_2_world = vr_camera1_tvec_2_world
    
    vr_camera2_rvec_2_world = np.array([[-2.68005629], [-1.24526011], [0.04842303]])
    vr_camera2_tvec_2_world = np.array([[-0.20924113], [0.06547744], [0.29285326]])
    vr_camera2_R_2_world, _ = cv2.Rodrigues(vr_camera2_rvec_2_world)
    vr_camera2_T_2_world = vr_camera2_tvec_2_world
    
    img = Image.open("4252229514797.pgm")
    for tag_id in range(0, 108):
        AprilBox(tag_id)