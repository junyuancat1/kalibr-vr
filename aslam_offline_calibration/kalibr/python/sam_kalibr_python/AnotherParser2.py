#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters

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

class AprilBox:
    def __init__(self, boardID, markerID, top_left, top_right, bottom_left, bottom_right):
        self.boardId = boardID
        self.markerId = markerID
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
        self.y_line_pos = markerID % 6
        self.z_line_pos = int(markerID / 6) - 6 * (self.boardId -1)

        # self.bottom_left3d = [self.x_line_pos * 0.072 * (1 + 0.2),  self.y_line_pos * 0.072 * (1 + 0.2) , 0.]
        # self.bottom_right3d = [self.bottom_left3d[0] + 0.072,  self.y_line_pos * 0.072 * (1 + 0.2) , 0.]
        # self.top_left3d = [self.x_line_pos * 0.072 * (1 + 0.2),  self.bottom_left3d[1] + 0.072 , 0.]
        # self.top_right3d = [self.bottom_left3d[0] + 0.072,  self.bottom_left3d[1] + 0.072 , 0.]
        
        self.bottom_left3d = [0., self.y_line_pos * 0.072 * (1 + 0.2),  self.z_line_pos * 0.072 * (1 + 0.2)]
        self.bottom_right3d = [0., self.bottom_left3d[1] + 0.072,  self.bottom_left3d[2]]
        self.top_left3d = [0., self.bottom_left3d[1],  self.bottom_left3d[2] + 0.072]
        self.top_right3d = [0., self.bottom_left3d[1] + 0.072,  self.bottom_left3d[2] + 0.072]

        print("image tagid {}, top_left{}, top_right{}, bottom_left{}, bottom_right{}".format(self.markerId, self.top_left, self.top_right, self.bottom_left, self.bottom_right))
        print("world tagid {}, top_left{}, top_right{}, bottom_left{}, bottom_right{}".format(self.markerId, self.top_left3d, self.top_right3d, self.bottom_left3d, self.bottom_right3d))

class CalibrationTargetDetector(object):
    def __init__(self, camera, targetConfig):
        targetParams = targetConfig.getTargetParams()
        targetType = targetConfig.getTargetType()
        
        #set up target
        if( targetType == 'checkerboard' ):
            grid = acv.GridCalibrationTargetCheckerboard(targetParams['targetRows'], 
                                                            targetParams['targetCols'], 
                                                            targetParams['rowSpacingMeters'], 
                                                            targetParams['colSpacingMeters'])
        
        elif( targetType == 'circlegrid' ):
            options = acv.CirclegridOptions(); 
            options.useAsymmetricCirclegrid = targetParams['asymmetricGrid']
            
            grid = acv.GridCalibrationTargetCirclegrid(targetParams['targetRows'],
                                                          targetParams['targetCols'], 
                                                          targetParams['spacingMeters'], 
                                                          options)
        
        elif( targetType == 'aprilgrid' ):
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
    # 相机标定内参，畸变参数
    camchain_file = "/home/shadowlly/kalibr/devel/lib/kalibr/camchain-precision_camera.yaml"

    # 右侧板
    target_board2_file = "april_6x6_2.yaml"
    
    cmd_parse = "python3 ./ParseApril.py --target {} --cam ~/kalibr/devel/lib/kalibr/camchain-precision_camera.yaml >> {}"
    
    cmd_parse_april_board2 = cmd_parse.format(target_board2_file, "finded_tag_indexs_board2.txt")
    
    os.system(cmd_parse_april_board2)
            

    sm.setLoggingLevel(sm.LoggingLevel.Info)
    targetConfig = kc.ConfigReader.CalibrationTargetParameters(target_board2_file)
    targetConfig.printDetails()
    camchain = kc.ConfigReader.CameraChainParameters(camchain_file)
    camConfig = camchain.getCameraParameters(0)
    camera = kc.ConfigReader.AslamCamera.fromParameters(camConfig)
    target = CalibrationTargetDetector(camera, targetConfig)
  
    target_image = cv2.imread("/home/shadowlly/pics/cam0/IMG_0242.JPG", cv2.IMREAD_GRAYSCALE)
    np_image = np.array(target_image)
    timestamp = acv.Time(0, 0)
    success, observation = target.detector.findTarget(timestamp, np_image)
    cornersImage = observation.getCornersImageFrame()
    print("corners: ", cornersImage)
    print("corners size: ", len(cornersImage))
    
    fname = "finded_tag_indexs_board2.txt"
    numbers = []
    with open(fname, 'r') as f:  #打开文件
        lines = f.readlines() #读取所有行
        last_line = lines[-1] #取最后一行     
        numbers = [int(x) for x in last_line.split()]
        print(numbers)

    if len(cornersImage) is not len(numbers)*4:
        print("ERROR corners is not equals to tag numbers")
        exit()
    
    
#   /// point ordering
#   ///          12-----13  14-----15
#   ///          | TAG 2 |  | TAG 3 |
#   ///          8-------9  10-----11
#   ///          4-------5  6-------7
#   ///    y     | TAG 0 |  | TAG 1 |
#   ///   ^      0-------1  2-------3
#   ///   |-->x

    april_box_list = []
    line_head_corner_index = 0
    for i in range(0,6):
        line_tag_counts = 0
        for j in range(0,6):
            tag_id = i * 6 + j + 36
            if tag_id in numbers:
                print("finded tagid ", tag_id)
                line_tag_counts += 1

        print("line 1 got tag numbers ", line_tag_counts)
        tmp_line_tag_count = 0
        for j in range(0,6):
            tag_id = i * 6 + j + 36
            if tag_id in numbers:
                left_bottom_corner_index = line_head_corner_index + 2 * tmp_line_tag_count
                right_bottom_corner_index = line_head_corner_index + 2 * tmp_line_tag_count + 1
                left_up_corner_index = line_head_corner_index + 2 * line_tag_counts + 2 * tmp_line_tag_count
                right_up_corner_index = line_head_corner_index+ 2 * line_tag_counts + 2 * tmp_line_tag_count + 1
                april_box_list.append(AprilBox(2, tag_id, cornersImage[left_up_corner_index], cornersImage[right_up_corner_index], cornersImage[left_bottom_corner_index], cornersImage[right_bottom_corner_index]))
                print("tag and corners, ", tag_id, left_bottom_corner_index, right_bottom_corner_index, left_up_corner_index, right_up_corner_index)
                tmp_line_tag_count += 1
        line_head_corner_index += line_tag_counts * 4


    CameraMatrix = np.float32([[4246.77883083, 0, 2586.61014888],
                            [0, 4236.89559297, 1739.81219089],
                            [0, 0, 1]]).reshape((3,3))

    CameraDistortion = np.float32([0.12115487 ,0.53058533 ,-1.40555966 ,2.23803307])



    result_image = cv2.imread("result.jpg")

    world_points = np.empty((0,3), dtype=np.float32)
    image_points = np.empty((0,2), dtype=np.float32)

    for tag in april_box_list:
        print("handling tagid: ", tag.markerId)
        world_points = np.append(world_points, [np.array([tag.bottom_left3d[0], tag.bottom_left3d[1], tag.bottom_left3d[2]])], axis=0)
        world_points = np.append(world_points, [np.array([tag.bottom_right3d[0], tag.bottom_right3d[1], tag.bottom_right3d[2]])], axis=0)
        world_points = np.append(world_points, [np.array([tag.top_left3d[0], tag.top_left3d[1], tag.top_left3d[2]])], axis=0)
        world_points = np.append(world_points, [np.array([tag.top_right3d[0], tag.top_right3d[1], tag.top_right3d[2]])], axis=0)
        
        image_points = np.append(image_points, [np.array([tag.bottom_left[0], tag.bottom_left[1]])], axis=0) 
        image_points = np.append(image_points, [np.array([tag.bottom_right[0], tag.bottom_right[1]])], axis=0) 
        image_points = np.append(image_points, [np.array([tag.top_left[0], tag.top_left[1]])], axis=0) 
        image_points = np.append(image_points, [np.array([tag.top_right[0], tag.top_right[1]])], axis=0)    
        
    print("world points: ", world_points)
    print("image points: ", image_points)

    IK = np.eye(3)
    ID = np.zeros((1,5))

    print("reshaped :", image_points.reshape(1, -1, 2))
    undistorted_image_points = cv2.fisheye.undistortPoints(image_points.reshape(1, -1, 2), CameraMatrix, CameraDistortion)
    _, E_rvec, E_tvec = cv2.solvePnP(world_points, undistorted_image_points, IK, ID)
    R, _ = cv2.Rodrigues(E_rvec)
    
        # temp_objpoints = np.asarray(world_points, dtype=np.float64)
    temp_objpoints = np.reshape(world_points, (1, 1, len(world_points), 3))
    temp_imgpoints = np.reshape(image_points, (1, 1, len(image_points), 2))
    calibration_flags = (
                    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                    + cv2.fisheye.CALIB_CHECK_COND
                    + cv2.fisheye.CALIB_FIX_SKEW
                )
    rvecs_temp = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(temp_objpoints))]
    tvecs_temp = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(temp_objpoints))]
    K_temp = np.zeros((3, 3))
    D_temp = np.zeros((4, 1))
    retval, K2, D2, rvecs2, tvecs2 = cv2.fisheye.calibrate(temp_objpoints, temp_imgpoints, target_image.shape[::-1], K_temp, D_temp, rvecs_temp, tvecs_temp,  calibration_flags, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-6))

    print("K ori: ", CameraMatrix)
    print("K now: ", K2)
    
    print("D ori: ", CameraDistortion)
    print("D noe: ", D2)

    print("r2: [{}, {}, {}]".format(rvecs2[0][0], rvecs2[0][1], rvecs2[0][2]))
    print("t2: [{}, {}, {}]".format(tvecs2[0][0], tvecs2[0][1], tvecs2[0][2]))


    # 打印相机的旋转和平移矩阵
    print("Rotation matrix:\n", E_rvec)
    print("Translation matrix:\n", E_tvec)


#  [[-0.04476985]
#  [-0.25774712]
#  [ 1.56199604]]
