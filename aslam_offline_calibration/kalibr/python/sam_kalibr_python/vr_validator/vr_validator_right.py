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
    
if __name__ == "__main__":        
    parser = argparse.ArgumentParser(description='Validate the intrinsics of a camera.')
    parser.add_argument('--target', dest='targetYaml', help='Calibration target configuration as yaml file', required=True)
    parser.add_argument('--cam', dest='chainYaml', help='Camera configuration as yaml file', required=True)
    parser.add_argument('--image', dest='image', help='Image to validate', required=True)
    parser.add_argument('--camera-index', dest='cameraIndex', help='Index of the camera to validate(1 = vr_camera_trackingA, 2 = vr_camera_trackingB, 3 = vr_camera_control_trackingA, 4 = vr_camera_control_trackingB)', required=True)
    parsed = parser.parse_args()
        
    # sm.setLoggingLevel(sm.LoggingLevel.Debug)
    sm.setLoggingLevel(sm.LoggingLevel.Info)

    targetConfig = kc.ConfigReader.CalibrationTargetParameters(parsed.targetYaml)
    # targetConfig.printDetails()
    camchain = kc.ConfigReader.CameraChainParameters(parsed.chainYaml)
    camConfig = camchain.getCameraParameters(0)
    # camConfig.printDetails(); 
    camera = kc.ConfigReader.AslamCamera.fromParameters(camConfig)
    target = CalibrationTargetDetector(camera, targetConfig)

    # 读取pgm图片
    img = Image.open(parsed.image)
    # 转换为numpy数组
    np_image = np.array(img)
    # 创建OpenCV的Mat对象
    target_image_color = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)
    half = np_image.shape[1] // 2
    right_img = np_image[:, half:]
    timestamp = acv.Time(0, 0)

    vr_camera2_K = np.array([[276.44363, 0, 318.01495], [0, 276.44363, 240.67293], [0, 0, 1]])
    vr_camera2_D = np.array([0.0051421098, -0.0088336899, 0.0097894818, -0.0032491733])
    
    vr_camera4_K = np.array([[277.06611, 0, 317.67797], [0, 277.06611, 238.94741], [0, 0, 1]])
    vr_camera4_D = np.array([0.026178562, -0.053109878, 0.041701285, -0.010932579])
    
    vr_camera_K = np.eye(3)
    vr_camera_D = np.zeros(4) 
    if parsed.cameraIndex == '2':
        vr_camera_K = vr_camera2_K
        vr_camera_D = vr_camera2_D
    elif parsed.cameraIndex == '4':
        vr_camera_K = vr_camera4_K
        vr_camera_D = vr_camera4_D
    
    undistorted_img = cv2.fisheye.undistortImage(right_img, vr_camera_K, vr_camera_D, None, vr_camera2_K, (2000, 2000))

    

    success_r, observation_r = target.detector.findTarget(timestamp, undistorted_img)
    cornersImage_r = observation_r.getCornersImageFrame()
    # print("corners: ", cornersImage_r)
    # print("corners size l: ", len(cornersImage_r))

    for index in range(len(cornersImage_r)):
        corner = cornersImage_r[index]
        cv2.circle(undistorted_img, (int(corner[0]), int(corner[1])), 4, (0, 0, 255), -1)
        cv2.putText(undistorted_img, str(index), (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0, 0, 255), thickness=2)
    cv2.imwrite("./undistorted_right_img.jpg", undistorted_img)

    ori_points = distortPoints(cornersImage_r, vr_camera_K, vr_camera_D)

    if ori_points is None or len(ori_points[0]) <= 0:
        exit(0)
    
    ori_points = np.array(ori_points).reshape(-1, 2)
    # print("ori_points: ", ori_points)