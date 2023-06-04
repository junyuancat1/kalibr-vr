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
tag_point_worlds = np.zeros((0, 3))

def E_from_a_2_b(camera_a_Rbc, camera_a_tbc, camera_b_Rbc, camera_b_tbc, camera_a_Rcw, camera_a_tcw):

    camera_a_tbc = camera_a_tbc.reshape(3,1)
    camera_b_tbc = camera_b_tbc.reshape(3,1)
    R_camera_a_to_camera_b = np.dot(camera_b_Rbc, camera_a_Rbc.T)
    T_camera_a_to_camera_b = camera_b_tbc - np.dot(camera_b_Rbc, np.dot(camera_a_Rbc.T, camera_a_tbc))
    # 世界 -> 相机1
    R_world_to_camera_a = camera_a_Rcw
    T_world_to_camera_a = camera_a_tcw.reshape(3,1)
    print("T_world_to_camera_a", T_world_to_camera_a)
    # 世界 -> 相机2
    R_world_to_camera_b = np.dot(R_camera_a_to_camera_b, R_world_to_camera_a)
    T_world_to_camera_b = np.dot(R_camera_a_to_camera_b, T_world_to_camera_a).reshape(3,1) + T_camera_a_to_camera_b.reshape(3,1)
    r_world_to_camera_b, _ = cv2.Rodrigues(R_world_to_camera_b)
    # print("r_world_to_camera_b", r_world_to_camera_b)
    return r_world_to_camera_b, T_world_to_camera_b

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
    def __init__(self, markerID, top_left, top_right, bottom_left, bottom_right):
        self.markerId = markerID
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
        self.center = [(top_left[0][0] + top_right[0][0] + bottom_left[0][0] + bottom_right[0][0]) / 4, (top_left[0][1] + top_right[0][1] + bottom_left[0][1] + bottom_right[0][1]) / 4]
        self.boardId = int(markerID / 36) + 1

        self.x_line_pos = 0
        self.y_line_pos = 0
        self.z_line_pos = 0
        self.bottom_left3d = [0. ,0., 0.]
        self.bottom_right3d = [0. ,0., 0.]
        self.top_left3d = [0. ,0., 0.]
        self.top_right3d = [0. ,0., 0.]
        # if self.boardId == 1:
        #     self.x_line_pos = markerID % 6
        #     self.y_line_pos = int(markerID / 6) - 6 * (self.boardId -1)
        #     self.bottom_left3d = [self.x_line_pos * 0.072 * (1 + 0.2),  self.y_line_pos * 0.072 * (1 + 0.2) , 0.]
        #     self.bottom_right3d = [self.bottom_left3d[0] + 0.072,  self.y_line_pos * 0.072 * (1 + 0.2) , 0.]
        #     self.top_left3d = [self.x_line_pos * 0.072 * (1 + 0.2),  self.bottom_left3d[1] + 0.072 , 0.]
        #     self.top_right3d = [self.bottom_left3d[0] + 0.072,  self.bottom_left3d[1] + 0.072 , 0.]
        # elif self.boardId == 2:
        #     self.y_line_pos = markerID % 6
        #     self.z_line_pos = int(markerID / 6) - 6 * (self.boardId -1)
        #     self.bottom_left3d = [0., self.y_line_pos * 0.072 * (1 + 0.2),  self.z_line_pos * 0.072 * (1 + 0.2)]
        #     self.bottom_right3d = [0., self.bottom_left3d[1] + 0.072,  self.bottom_left3d[2]]
        #     self.top_left3d = [0., self.bottom_left3d[1],  self.bottom_left3d[2] + 0.072]
        #     self.top_right3d = [0., self.bottom_left3d[1] + 0.072,  self.bottom_left3d[2] + 0.072]
        # elif self.boardId == 3:
        #     self.z_line_pos = markerID % 6
        #     self.x_line_pos = int(markerID / 6) - 6 * (self.boardId -1)
        #     self.bottom_left3d = [self.x_line_pos * 0.072 * (1 + 0.2),  0., self.z_line_pos * 0.072 * (1 + 0.2)]
        #     self.bottom_right3d = [self.bottom_left3d[0], 0., self.bottom_left3d[2] + 0.072]
        #     self.top_left3d = [self.bottom_left3d[0] + 0.072, 0., self.bottom_left3d[2]]
        #     self.top_right3d = [self.bottom_left3d[0] + 0.072, 0., self.bottom_left3d[2] + 0.072]
        # else:
        #     print("WTF? ", markerID, self.boardId)
        self.bottom_left3d = tag_point_worlds[self.markerId*4]
        self.bottom_right3d = tag_point_worlds[self.markerId*4 + 1]
        self.top_right3d = tag_point_worlds[self.markerId*4 + 2]
        self.top_left3d = tag_point_worlds[self.markerId*4 + 3]
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
    

    # 初始化numpy数组
    # 解析文档并存储到numpy数组中
    with open("tag_world.txt", 'r') as f:  #打开文件
        lines = f.readlines() #读取所有行
        for line in lines:
            if line.strip() == '':
                continue
            # 获取id和坐标
            parts = line.split()
            tag_id = int(parts[0])
            coordinates = [float(x.strip('[],')) for x in parts[1:]]
            cornersworld = np.array(coordinates).reshape(4, 3)
            tag_point_worlds = np.vstack((tag_point_worlds, cornersworld))
            # print(f'Tag ID: {tag_id}, Coordinates: {coordinates}')
    
    print("length", len(tag_point_worlds))
    
    # 左下
    vr_camera1_K = np.float32([[276.28156, 0, 317.99796], [0, 276.28156, 240.97645], [0, 0, 1]]).reshape(3,3)
    vr_camera1_D = np.float32([-0.013057856, 0.026553879, -0.01489986, 0.0031719355])
    vr_camera1_Rbc1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vr_camera1_tbc1 = np.array([0, 0, 0])
    vr_camera_trackingA = VRCamera((640, 480), vr_camera1_K, vr_camera1_D, vr_camera1_Rbc1, vr_camera1_tbc1)

    # 右下
    vr_camera2_K = np.array([[276.44363, 0, 318.01495], [0, 276.44363, 240.67293], [0, 0, 1]])
    vr_camera2_D = np.array([0.0051421098, -0.0088336899, 0.0097894818, -0.0032491733])
    vr_camera2_Rbc2 = np.array([[-0.99949765, -0.025876258, 0.018299539], [0.029883759, -0.96176534, 0.27223959], [0.010555321, 0.27264969, 0.96205547]])
    vr_camera2_tbc2 = np.array([-0.00038707237, 0.10318894, -0.01401102])
    vr_camera_trackingB = VRCamera((640, 480), vr_camera2_K, vr_camera2_D, vr_camera2_Rbc2, vr_camera2_tbc2)

    # 左上
    # <Rig translation="-0.0056250621 0.041670205 -0.013388009 " rowMajorRotationMat="0.5687224 0.81622218 -0.10166702 -0.3726157 0.36585493 0.85282338 0.73328874 -0.44713703 0.51220709 " />
    vr_camera3_K = np.array([[273.87003, 0, 317.51742], [0, 273.87003, 238.06963], [0, 0, 1]])
    vr_camera3_D = np.array([0.006437201, -0.015906533, 0.021496724, -0.0065812969])
    vr_camera3_Rbc3 = np.array([[0.5687224, 0.81622218, -0.10166702], [-0.3726157, 0.36585493, 0.85282338], [0.73328874, -0.44713703, 0.51220709]])
    vr_camera3_tbc3 = np.array([-0.0056250621, 0.041670205, -0.013388009])
    vr_camera_control_trackingA = VRCamera((640, 480), vr_camera3_K, vr_camera3_D, vr_camera3_Rbc3, vr_camera3_tbc3)

    # 右上
    # <Rig translation="-0.079096205 0.064828795 -0.066688745 " rowMajorRotationMat="-0.59714752 0.79234879 -0.12489289 -0.35174627 -0.11873178 0.92853504 0.72089486 0.598403 0.34960612 " />
    vr_camera4_K = np.array([[277.06611, 0, 317.67797], [0, 277.06611, 238.94741], [0, 0, 1]])
    vr_camera4_D = np.array([0.026178562, -0.053109878, 0.041701285, -0.010932579])
    vr_camera4_Rbc4 = np.array([[-0.59714752, 0.79234879, -0.12489289], [-0.35174627, -0.11873178, 0.92853504], [0.72089486, 0.598403, 0.34960612]])
    vr_camera4_tbc4 = np.array([-0.079096205, 0.064828795, -0.066688745])
    vr_camera_control_trackingB = VRCamera((640, 480), vr_camera4_K, vr_camera4_D, vr_camera4_Rbc4, vr_camera4_tbc4)
    
    # 相机标定内参，畸变参数
    camchain_file = "./camchain-trackingA.yaml"
    #左侧板
    target_board3_file = "april_6x6_all.yaml"
    
    result_output_file_left = "finded_tag_indexs_left.txt"
    result_output_file_right = "finded_tag_indexs_right.txt"
    
    cmd_parse_l = "python3 ./vr_validator_left.py --target {} --cam {}  --camera-index 3 --image {} > {}"
    cmd_parse_r = "python3 ./vr_validator_right.py --target {} --cam {} --camera-index 4 --image {} > {}"

    # image_file = "./4252229514797.pgm"
    image_file = "4250462839129.pgm"
    
    
    # step 1 分别识别左右两侧的AprilTag(结果为图片去畸变后的坐标)
    cmd_parse_april_board_l = cmd_parse_l.format(target_board3_file, camchain_file, image_file, result_output_file_left)
    cmd_parse_april_board_r = cmd_parse_r.format(target_board3_file, camchain_file, image_file, result_output_file_right)
    os.system(cmd_parse_april_board_l)
    os.system(cmd_parse_april_board_r)

    # 空的n*2矩阵，用于存放坐标
    left_tag_ids = []
    left_image_corners_undistorted = np.zeros((0, 2))
    right_tag_ids = []
    right_image_corners_undistorted = np.zeros((0, 2))

    # step 1.1 读取左侧的AprilTag识别结果, 并解回去畸变前的坐标
    with open(result_output_file_left, 'r') as f:  #打开文件
        lines = f.readlines() #读取所有行
        for line in lines:
            if line.strip() == '':
                continue
            # 获取id和坐标
            parts = line.split()
            tag_id = int(parts[0])
            coordinates = [float(x.strip('[],')) for x in parts[1:]]
            cornersImage = np.array(coordinates).reshape((4, 2))
            left_image_corners_undistorted = np.vstack((left_image_corners_undistorted, cornersImage))
            left_tag_ids.append(tag_id)
            # print(f'Left Tag ID: {tag_id}, Coordinates: {coordinates}')
    
    left_image_corners_distorted = distortPoints(left_image_corners_undistorted, vr_camera1_K, vr_camera1_D)

    # step 1.2 读取右侧的AprilTag识别结果, 并解回去畸变前的坐标
    with open(result_output_file_right, 'r') as f:  #打开文件
        lines = f.readlines() #读取所有行
        for line in lines:
            if line.strip() == '':
                continue
            # 获取id和坐标
            parts = line.split()
            tag_id = int(parts[0])
            coordinates = [float(x.strip('[],')) for x in parts[1:]]
            cornersImage = np.array(coordinates).reshape((4, 2))
            right_image_corners_undistorted = np.vstack((right_image_corners_undistorted, cornersImage))
            right_tag_ids.append(tag_id)
            # print(f'Right Tag ID: {tag_id}, Coordinates: {coordinates}')
    
    # print("right_image_corners_undistorted: ", right_image_corners_undistorted)
    right_image_corners_distorted = distortPoints(right_image_corners_undistorted, vr_camera2_K, vr_camera2_D)

    # step 1.v 确认左右两侧的AprilTag识别结果以及畸变无误
    # write points to image
    # img = Image.open(image_file)
    # np_img = np.array(img)
    # for i in range(len(right_image_corners)):
    #     cv2.circle(np_img, (int(right_image_corners[i][0][0] + 640), int(right_image_corners[i][0][1])), 5, (0, 255, 0), -1)
    # cv2.imwrite("right_image_corners.jpg", np_img)
    

    # step 2 将tag组成AprilBox
    # step 2.1 左侧
    april_box_left_list = []
    for i in range(len(left_tag_ids)):
        left_tag_id = left_tag_ids[i]
        bottom_left = left_image_corners_distorted[i * 4]
        bottom_right = left_image_corners_distorted[i * 4 + 1]
        top_right = left_image_corners_distorted[i * 4 + 2]
        top_left = left_image_corners_distorted[i * 4 + 3]

        # bottom_left = left_image_corners_distorted[i * 4 + 1]
        # bottom_right = left_image_corners_distorted[i * 4 + 2]
        # top_right = left_image_corners_distorted[i * 4 + 3]
        # top_left = left_image_corners_distorted[i * 4]
        april_box_left_list.append(AprilBox(left_tag_id, top_left, top_right, bottom_left, bottom_right))
    
    # step 2.2 右侧
    april_box_right_list = []
    for i in range(len(right_tag_ids)):
        right_tag_id = right_tag_ids[i]
        bottom_left = right_image_corners_distorted[i * 4]
        bottom_right = right_image_corners_distorted[i * 4 + 1]
        top_right = right_image_corners_distorted[i * 4 + 2]
        top_left = right_image_corners_distorted[i * 4 + 3]

        # bottom_left = right_image_corners_distorted[i * 4 + 1]
        # bottom_right = right_image_corners_distorted[i * 4 + 2]
        # top_right = right_image_corners_distorted[i * 4 + 3]
        # top_left = right_image_corners_distorted[i * 4]
        april_box_right_list.append(AprilBox(right_tag_id, top_left, top_right, bottom_left, bottom_right))
        
    # step 3 计算相机姿态
    # step 3.1 左侧
    left_board_world_point = np.empty((0,3), dtype=np.float32)
    left_board_image_point = np.empty((0,2), dtype=np.float32)
    for tag in april_box_left_list:
        left_board_world_point = np.append(left_board_world_point, [np.array([tag.bottom_left3d[0], tag.bottom_left3d[1], tag.bottom_left3d[2]])], axis=0)
        left_board_world_point = np.append(left_board_world_point, [np.array([tag.bottom_right3d[0], tag.bottom_right3d[1], tag.bottom_right3d[2]])], axis=0)
        left_board_world_point = np.append(left_board_world_point, [np.array([tag.top_left3d[0], tag.top_left3d[1], tag.top_left3d[2]])], axis=0)
        left_board_world_point = np.append(left_board_world_point, [np.array([tag.top_right3d[0], tag.top_right3d[1], tag.top_right3d[2]])], axis=0)
    
        left_board_image_point = np.append(left_board_image_point, [np.array([tag.bottom_left[0][0], tag.bottom_left[0][1]])], axis=0) 
        left_board_image_point = np.append(left_board_image_point, [np.array([tag.bottom_right[0][0], tag.bottom_right[0][1]])], axis=0) 
        left_board_image_point = np.append(left_board_image_point, [np.array([tag.top_left[0][0], tag.top_left[0][1]])], axis=0) 
        left_board_image_point = np.append(left_board_image_point, [np.array([tag.top_right[0][0], tag.top_right[0][1]])], axis=0)    
    
    img_distort = Image.open(image_file)
    img_distort = np.array(img_distort)
    for i in range(len(left_board_image_point)):
        cv2.circle(img_distort, (int(left_board_image_point[i][0]), int(left_board_image_point[i][1])), 5, (0, 255, 0), -1)
        cv2.putText(img_distort, str(i), (int(left_board_image_point[i][0]), int(left_board_image_point[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite("left_img_distort.jpg", img_distort)
    
    left_undistorted_image_points = cv2.fisheye.undistortPoints(left_board_image_point.reshape(-1, 1, 2), vr_camera3_K, vr_camera3_D)
    # print("left_undistorted_image_points: ", left_undistorted_image_points)
    
    # 无畸变图像上求解相机姿态
    IK = np.eye(3)
    ID = np.zeros((1,5))
    ret, rc1w, tc1w = cv2.solvePnP(left_board_world_point, left_undistorted_image_points, IK, ID)
    # 根据相机姿态求回畸变图像上的点
    # temp_points, _ = cv2.projectPoints(left_board_world_point.reshape(1, -1, 3), rc1w, tc1w, IK, ID)
    temp_points, _ = cv2.fisheye.projectPoints(left_board_world_point.reshape(1, -1, 3), rc1w, tc1w, vr_camera3_K, vr_camera3_D)
    print("camera left rvec: ", rc1w)
    print("camera left tvec: ", tc1w)

    # 含畸变的像素点
    # temp_points, _ = cv2.fisheye.projectPoints(left_board_world_point.reshape(1, -1, 3),  rc1w, tc1w, vr_camera1_K, vr_camera1_D)


    # draw undistorted points
    img_undistort = Image.open(image_file)
    img_undistort = np.array(img_undistort)
    for i in range(len(temp_points[0])):
        cv2.circle(img_undistort, (int(temp_points[0][i][0]), int(temp_points[0][i][1])), 5, (0, 255, 0), -1)
    cv2.imwrite("left_img_undistort.jpg", img_undistort)
    
    # step 3.1.v 左侧内参平均重投影误差验证
    left_average_reprojection_error = 0
    for i in range (0, len(temp_points[0])):
        left_average_reprojection_error += np.sqrt(np.square(left_board_image_point[i][0] - temp_points[0][i][0]) + np.square(left_board_image_point[i][1] - temp_points[0][i][1]))

    left_average_reprojection_error /= len(temp_points[0])
    print("camera left average_reprojection_error: ", left_average_reprojection_error)

    # step 3.2 右侧
    right_board_world_point = np.empty((0,3), dtype=np.float32)
    right_board_image_point = np.empty((0,2), dtype=np.float32)
    for tag in april_box_right_list:
        right_board_world_point = np.append(right_board_world_point, [np.array([tag.bottom_left3d[0], tag.bottom_left3d[1], tag.bottom_left3d[2]])], axis=0)
        right_board_world_point = np.append(right_board_world_point, [np.array([tag.bottom_right3d[0], tag.bottom_right3d[1], tag.bottom_right3d[2]])], axis=0)
        right_board_world_point = np.append(right_board_world_point, [np.array([tag.top_left3d[0], tag.top_left3d[1], tag.top_left3d[2]])], axis=0)
        right_board_world_point = np.append(right_board_world_point, [np.array([tag.top_right3d[0], tag.top_right3d[1], tag.top_right3d[2]])], axis=0)
    
        right_board_image_point = np.append(right_board_image_point, [np.array([tag.bottom_left[0][0], tag.bottom_left[0][1]])], axis=0) 
        right_board_image_point = np.append(right_board_image_point, [np.array([tag.bottom_right[0][0], tag.bottom_right[0][1]])], axis=0) 
        right_board_image_point = np.append(right_board_image_point, [np.array([tag.top_left[0][0], tag.top_left[0][1]])], axis=0) 
        right_board_image_point = np.append(right_board_image_point, [np.array([tag.top_right[0][0], tag.top_right[0][1]])], axis=0)
    
    img_distort = Image.open(image_file)
    img_distort = np.array(img_distort)
    for i in range(len(right_board_image_point)):
        cv2.circle(img_distort, (int(right_board_image_point[i][0] + 640), int(right_board_image_point[i][1])), 5, (0, 255, 0), -1)
        cv2.putText(img_distort, str(i), (int(right_board_image_point[i][0] + 640), int(right_board_image_point[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite("right_img_distort.jpg", img_distort)
    
    IK = np.eye(3)
    ID = np.zeros((1,5))
    right_undistorted_image_points = cv2.fisheye.undistortPoints(right_board_image_point.reshape(-1,1,2), vr_camera4_K, vr_camera4_D)

    ret, rc2w, tc2w = cv2.solvePnP(right_board_world_point, right_undistorted_image_points, IK, ID)
    # temp_points = cv2.projectPoints(right_board_world_point, rc2w, tc2w, IK, ID)
    temp_points = cv2.fisheye.projectPoints(right_board_world_point.reshape(1, -1, 3),  rc2w, tc2w, vr_camera4_K, vr_camera4_D)
    temp_points = temp_points[0]
    print("len(temp_points): ", len(temp_points))
    print("camera right rvec: ", rc2w)
    print("camera right tvec: ", tc2w)
    
    # step 3.2.v 右侧内参平均重投影误差验证
    right_average_reprojection_error = 0
    for i in range (0, len(temp_points[0])):
        # right_average_reprojection_error += np.linalg.norm(temp_points[0][i] - right_undistorted_image_points[i])
        right_average_reprojection_error += np.linalg.norm(temp_points[0][i] - right_board_image_point[i])
    
    right_average_reprojection_error /= len(temp_points[0])
    print("camera right average_reprojection_error: ", right_average_reprojection_error)
    
    
    # step 4. 验证外参相机1-> 相机2
    # 原理 标定板世界坐标系已知，假设相机1是准确的， 那么相机1中的tag点renewing是准确的且能和世界系精确投影
    # P_world -> P_camera1 -> P_camera2 -> P_image2
    # P_world -> P_camera1 中使用step 3.1估计的外参
    # P_camera1 -> P_camera2 中使用数据集device_calibration.xml 中的外参计算得到(即待验证的外参)
    # P_camera2 -> P_image2 中使用数据集中camera2的内参
    
    
    
    # step 4.1 寻找相机1和相机2中共有的tag
    camera1_common_tag_point_list = np.empty((0,2), dtype=np.int32)
    camera2_common_tag_point_list = np.empty((0,2), dtype=np.int32)
    point_world_list = np.empty((0,3), dtype=np.float32)
    
    for tag1 in april_box_left_list:
        for tag2 in april_box_right_list:
            if tag1.markerId == tag2.markerId:
                camera1_common_tag_point_list = np.append(camera1_common_tag_point_list, [np.array([tag1.bottom_left[0][0], tag1.bottom_left[0][1]])], axis=0)
                camera1_common_tag_point_list = np.append(camera1_common_tag_point_list, [np.array([tag1.bottom_right[0][0], tag1.bottom_right[0][1]])], axis=0)
                camera1_common_tag_point_list = np.append(camera1_common_tag_point_list, [np.array([tag1.top_left[0][0], tag1.top_left[0][1]])], axis=0)
                camera1_common_tag_point_list = np.append(camera1_common_tag_point_list, [np.array([tag1.top_right[0][0], tag1.top_right[0][1]])], axis=0)                                       
                
                camera2_common_tag_point_list = np.append(camera2_common_tag_point_list, [np.array([tag2.bottom_left[0][0], tag2.bottom_left[0][1]])], axis=0)
                camera2_common_tag_point_list = np.append(camera2_common_tag_point_list, [np.array([tag2.bottom_right[0][0], tag2.bottom_right[0][1]])], axis=0)
                camera2_common_tag_point_list = np.append(camera2_common_tag_point_list, [np.array([tag2.top_left[0][0], tag2.top_left[0][1]])], axis=0)
                camera2_common_tag_point_list = np.append(camera2_common_tag_point_list, [np.array([tag2.top_right[0][0], tag2.top_right[0][1]])], axis=0)
                
                point_world_list = np.append(point_world_list, [np.array([tag1.bottom_left3d[0], tag1.bottom_left3d[1], tag1.bottom_left3d[2]])], axis=0)
                point_world_list = np.append(point_world_list, [np.array([tag1.bottom_right3d[0], tag1.bottom_right3d[1], tag1.bottom_right3d[2]])], axis=0)
                point_world_list = np.append(point_world_list, [np.array([tag1.top_left3d[0], tag1.top_left3d[1], tag1.top_left3d[2]])], axis=0)
                point_world_list = np.append(point_world_list, [np.array([tag1.top_right3d[0], tag1.top_right3d[1], tag1.top_right3d[2]])], axis=0)
    
    # 当没有共同的tag时，退出
    if len(camera1_common_tag_point_list) == 0:
        print("no common tag")
        exit()
                
    # 绘制验证
    img = Image.open(image_file)
    target_image_color = np.array(img)
    target_image_color = cv2.cvtColor(target_image_color, cv2.COLOR_BGR2RGB)
    
    for i in range(0, len(camera1_common_tag_point_list)):
        cv2.circle(target_image_color, (int(camera1_common_tag_point_list[i][0]), int(camera1_common_tag_point_list[i][1])), 3, (255,0,0), -1)
    
    for i in range(0, len(camera2_common_tag_point_list)):
        cv2.circle(target_image_color, (int(camera2_common_tag_point_list[i][0] + 640), int(camera2_common_tag_point_list[i][1])), 3, (0,255,0), -1)
        cv2.line(target_image_color, (int(camera1_common_tag_point_list[i][0]), int(camera1_common_tag_point_list[i][1])), (int(camera2_common_tag_point_list[i][0] + 640), int(camera2_common_tag_point_list[i][1])), (255/len(camera1_common_tag_point_list)*i, 255/len(camera1_common_tag_point_list)*i, 0), 3)
    
    cv2.imwrite("common_tag.jpg", target_image_color)
    
    # print center
    for tag1 in april_box_left_list:
        cv2.putText(target_image_color, str(tag1.markerId), (int(tag1.center[0]), int(tag1.center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    
    for tag2 in april_box_right_list:
        cv2.putText(target_image_color, str(tag2.markerId), (int(tag2.center[0] + 640), int(tag2.center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
    cv2.imwrite("common_tag_center.jpg", target_image_color)
 
    # step 4.2 P_world -> P_camera1 (Eworld_to_camera1 · P_world = P_camera1) -> P_camera1 = Rworld_to_camera1 · P_world + tvec1
   
   
    Rc1w, _ = cv2.Rodrigues(rc1w)
    Eworld_to_camera1 = np.hstack((Rc1w, tc1w))
    Eworld_to_camera1 = np.vstack((Eworld_to_camera1, np.array([0,0,0,1])))
    
    Rc2w, _ = cv2.Rodrigues(rc2w)
    E2_to_world = np.hstack((Rc2w, tc2w))
    E2_to_world = np.vstack((E2_to_world, np.array([0,0,0,1])))
    

    camera1_points = np.empty((0,3), dtype=np.float32)
    for world_point in point_world_list:
        camera1_point = np.dot(Rc1w, world_point.reshape(3,1)) + tc1w
        point = np.array([camera1_point[0][0], camera1_point[1][0], camera1_point[2][0]])
        camera1_points = np.append(camera1_points, [point], axis=0)

    # step 4.3 P_camera1 -> P_camera2 -> P_image2 (fisheye projectPoints)
    # (Ecamera1_to_camera1 · P_camera1 = Ecamera2_to_camera2 · P_camera2)
    # (Rcamera1_to_camera1 · P_camera1 + tcamera1_to_camera1 = Rcamera2_to_camera1 · P_camera2 + tcamera2_to_camera1)
    # (P_camera2 = inv(Ecamera2_to_camera1) · Ecamera1_to_camera1 · P_camera1)  
    # P_camera2 = inv(Rcamera2_to_camera1) · (Rcamera1_to_camera1 · P_camera1 + tcamera1_to_camera1 - tcamera2_to_camera1)
    # NewR = inv(Rcamera2_to_camera1) · Rcamera1_to_camera1, NewT = inv(Rcamera2_to_camera1) · (tcamera1_to_camera1 - tcamera2_to_camera1)
    # (P_image2 = projectPoints(P_camera2))

    Rc2c1 = np.dot(vr_camera2_Rbc2.T, vr_camera1_Rbc1)
    rc2c1, _ = cv2.Rodrigues(Rc2c1)
    tc2c1 = np.dot(vr_camera2_Rbc2.T, vr_camera1_tbc1 - vr_camera2_tbc2)
    
    Rc2w_2 = np.dot(Rc2c1,Rc1w)
    print("tc1w: ", tc1w)
    tc2w_2 = np.dot(Rc2c1,tc1w).reshape(1,3) + tc2c1
    rc2w_2, _ = cv2.Rodrigues(Rc2w_2)
    
    image2_points, _ = cv2.fisheye.projectPoints(point_world_list.reshape(1, -1, 3), rc2w_2, tc2w_2, vr_camera2_K, vr_camera2_D)
   
    img = Image.open(image_file)
    np_img = np.array(img)
    rgb_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    for index in range(0, len(image2_points[0])):
        image_point = image2_points
        if image_point[0][index][0] < 0 or image_point[0][index][0] > 640 or image_point[0][index][1] < 0 or image_point[0][index][1] > 480:
            continue
        distance = np.sqrt(np.power(image_point[0][index][0] - camera2_common_tag_point_list[index][0], 2) + np.power(image_point[0][index][1] - camera2_common_tag_point_list[index][1], 2))
        color = (0, 0, 0)
        if distance < 3: # green point
            color = (0, 255, 0)
        elif distance < 4: # yellow point
            color = (0, 255, 255)
        else: # red point
            color = (0, 0, 255)
        cv2.circle(rgb_img, (int(image_point[0][index][0] + 640), int(image_point[0][index][1])), 5, color, -1)  
        cv2.line(rgb_img, (int(image_point[0][index][0] + 640), int(image_point[0][index][1])), (int(camera1_common_tag_point_list[index][0]), int(camera1_common_tag_point_list[index][1])), color, 2)
                  
    # 左侧平均误差
    left_average_reprojection_error = round(left_average_reprojection_error, 5)
    cv2.putText(rgb_img, "avg reproj error: " + str(left_average_reprojection_error), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # 右侧平均误差
    right_average_reprojection_error = round(right_average_reprojection_error, 5)
    cv2.putText(rgb_img, "avg reproj error: " + str(right_average_reprojection_error), (640 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite("result!.jpg", rgb_img)
    
# --------------------------------------------------------------------
#     camera left rvec:  [[-1.31864372]
#  [ 2.66745841]
#  [-0.45909558]]
# camera left tvec:  [[0.21389172]
#  [0.11939645]
#  [0.27458963]]
# len(temp_points[0][0]):  1
# temp_points[0][i]:  [0.77895048 0.43481775]
# camera left average_reprojection_error:  0.0016436500304428674
# camera right rvec:  [[-2.78232977]
#  [-1.2651627 ]
#  [ 0.40838129]]
# camera right tvec:  [[-0.21116153]
#  [ 0.07293855]
#  [ 0.28693892]]



# camera left rvec:  [[0.01867105]
#  [ 2.37005514]
#  [-1.92390398]]
# camera left tvec:  [[0.19028125]
#  [0.23620335]
#  [0.22807506]]
# len(temp_points[0][0]):  1
# temp_points[0][i]:  [0.70136941 0.70133784]
# camera left average_reprojection_error:  0.0036166238047459303
# camera right rvec:  [[ 1.40496441]
#  [ 1.60617537]
#  [-1.39236939]]
# camera right tvec:  [[-0.14519866]
#  [ 0.22861433]
#  [ 0.25446183]]


# camera left rvec:  [[ 0.01503205]
#  [ 2.36084438]
#  [-1.9380543 ]]
# camera left tvec:  [[0.19063579]
#  [0.23884748]
#  [0.22340515]]
# camera left average_reprojection_error:  0.4025393872051297
# len(temp_points):  1
# camera right rvec:  [[ 1.40175713]
#  [ 1.60409248]
#  [-1.39756435]]
# camera right tvec:  [[-0.14467937]
#  [ 0.23015832]
#  [ 0.25502762]]

    fresh_rc1w = np.float32([-1.31864372, 2.66745841, -0.45909558]).reshape(1,3)
    fresh_tc1w = np.float32([0.21389172, 0.11939645, 0.27458963]).reshape(3,1)
    fresh_Rc1w, _ = cv2.Rodrigues(fresh_rc1w)
    
    fresh_rc2w = np.float32([-2.78232977, -1.2651627, 0.40838129]).reshape(1,3)
    fresh_tc2w = np.float32([-0.21116153, 0.07293855, 0.28693892]).reshape(3,1)
    fresh_Rc2w, _ = cv2.Rodrigues(fresh_rc2w)

    fresh_rc3w = np.float32([0.01503205, 2.36084438, -1.9380543]).reshape(1,3)
    fresh_tc3w = np.float32([0.19063579, 0.23884748, 0.22340515]).reshape(3,1)
    fresh_Rc3w, _ = cv2.Rodrigues(fresh_rc3w)
    
    fresh_rc4w = np.float32([1.40175713, 1.60409248, -1.39756435]).reshape(1,3)
    fresh_tc4w = np.float32([-0.14467937, 0.23015832, 0.25502762]).reshape(3,1)
    fresh_Rc4w, _ = cv2.Rodrigues(fresh_rc4w)
    
    image_file_t = "4250462839129.pgm" # 3,4
    image_file_b = "4252229514797.pgm" # 1,2
    
    image_top = np.array(Image.open(image_file_t))
    image_bottom = np.array(Image.open(image_file_b))
    half_top = image_top.shape[1] // 2
    half_bottom = image_bottom.shape[1] // 2
    
    image_camera_1 = image_bottom[:, :half_bottom]
    image_camera_2 = image_bottom[:, half_bottom:]
    image_camera_3 = image_top[:, :half_top]
    image_camera_4 = image_top[:, half_top:]
    
    # temp_points = cv2.projectPoints(right_board_world_point, rc2w, tc2w, IK, ID)
    whole_world_apriltag_points = [] # 108 * 4
    for index in range(0, 108):
        tag = AprilBox(index,[[0,0]],[[0,0]],[[0,0]],[[0,0]])
        whole_world_apriltag_points.append(tag.bottom_left3d)
        whole_world_apriltag_points.append(tag.bottom_right3d)
        whole_world_apriltag_points.append(tag.top_right3d)
        whole_world_apriltag_points.append(tag.top_left3d)
        
    # projection all points to camera1 image
    image1_points, _ = cv2.fisheye.projectPoints(np.array(whole_world_apriltag_points).reshape(1, -1, 3), fresh_rc1w, fresh_tc1w, vr_camera1_K, vr_camera1_D)
    image2_points, _ = cv2.fisheye.projectPoints(np.array(whole_world_apriltag_points).reshape(1, -1, 3), fresh_rc2w, fresh_tc2w, vr_camera2_K, vr_camera2_D)
    
    image3_points, _ = cv2.fisheye.projectPoints(np.array(whole_world_apriltag_points).reshape(1, -1, 3), fresh_rc3w, fresh_tc3w, vr_camera3_K, vr_camera3_D)
    
    image4_points, _ = cv2.fisheye.projectPoints(np.array(whole_world_apriltag_points).reshape(1, -1, 3), fresh_rc4w, fresh_tc4w, vr_camera4_K, vr_camera4_D)

    image1_points_filter,_ = cv2.projectPoints(np.array(whole_world_apriltag_points).reshape(1, -1, 3), fresh_rc1w, fresh_tc1w, IK, ID)
    image2_points_filter,_ = cv2.projectPoints(np.array(whole_world_apriltag_points).reshape(1, -1, 3), fresh_rc2w, fresh_tc2w, IK, ID)
    image3_points_filter,_ = cv2.projectPoints(np.array(whole_world_apriltag_points).reshape(1, -1, 3), fresh_rc3w, fresh_tc3w, IK, ID)
    image4_points_filter,_ = cv2.projectPoints(np.array(whole_world_apriltag_points).reshape(1, -1, 3), fresh_rc4w, fresh_tc4w, IK, ID)
    
    rad_image1_points_filter = []
    rad_image2_points_filter = []
    rad_image3_points_filter = []
    rad_image4_points_filter = []
    
    ratio_filter = 0.01
    
    for point in image1_points_filter:
        distance = np.sqrt((point[0][0] - vr_camera1_K[0][2]) ** 2 + (point[0][1] - vr_camera1_K[1][2]) ** 2)
        rad_image1_points_filter.append(distance)  
    rad_image1_points_filter_np = np.array(rad_image1_points_filter)
    rad_image1_points_filter_np.sort()
    max_distance1 = rad_image1_points_filter_np[-int(len(rad_image1_points_filter_np) *ratio_filter )]
    print(max_distance1)

    for point in image2_points_filter:
        distance = np.sqrt((point[0][0] - vr_camera2_K[0][2]) ** 2 + (point[0][1] - vr_camera2_K[1][2]) ** 2)
        rad_image2_points_filter.append(distance)
    rad_image2_points_filter = np.array(rad_image2_points_filter)
    rad_image2_points_filter = rad_image2_points_filter.argsort()
    max_distance2 = rad_image2_points_filter[-int(len(rad_image2_points_filter) * ratio_filter)] # 0.1
     
    for point in image3_points_filter:
        distance = np.sqrt((point[0][0] - vr_camera3_K[0][2]) ** 2 + (point[0][1] - vr_camera3_K[1][2]) ** 2)
        rad_image3_points_filter.append(distance)
    rad_image3_points_filter = np.array(rad_image3_points_filter)
    rad_image3_points_filter = rad_image3_points_filter.argsort()
    max_distance3 = rad_image3_points_filter[-int(len(rad_image3_points_filter) * ratio_filter)]
        
    for point in image4_points_filter:
        distance = np.sqrt((point[0][0] - vr_camera4_K[0][2]) ** 2 + (point[0][1] - vr_camera4_K[1][2]) ** 2)
        rad_image4_points_filter.append(distance)
    rad_image4_points_filter = np.array(rad_image4_points_filter)
    rad_image4_points_filter = rad_image4_points_filter.argsort()
    max_distance4 = rad_image4_points_filter[-int(len(rad_image4_points_filter) * ratio_filter)]
    
    print("max_distance1: ", max_distance1, "max_distance2: ", max_distance2, "max_distance3: ", max_distance3, "max_distance4: ", max_distance4)
    
    # projection all points to camera1 world
    # camera1_point = np.dot(Rc1w, world_point.reshape(3,1)) + tc1w
    camera1_temp_points = []
    camera2_temp_points = []
    camera3_temp_points = []
    camera4_temp_points = []
    for wold_point in whole_world_apriltag_points:
        camera1_temp_points.append(np.dot(fresh_Rc1w, wold_point.reshape(3,1)) + fresh_tc1w)
        camera2_temp_points.append(np.dot(fresh_Rc2w, wold_point.reshape(3,1)) + fresh_tc2w)
        camera3_temp_points.append(np.dot(fresh_Rc3w, wold_point.reshape(3,1)) + fresh_tc3w)
        camera4_temp_points.append(np.dot(fresh_Rc4w, wold_point.reshape(3,1)) + fresh_tc4w)

    camera1_temp_points = np.array(camera1_temp_points).reshape(1, -1, 3)
    camera2_temp_points = np.array(camera2_temp_points).reshape(1, -1, 3)
    camera3_temp_points = np.array(camera3_temp_points).reshape(1, -1, 3)
    camera4_temp_points = np.array(camera4_temp_points).reshape(1, -1, 3)
     
    # 相机1到相机2的姿态变换
    # Rc2c1 = np.dot(vr_camera2_Rbc2.T, vr_camera1_Rbc1)
    # tc2c1 = np.dot(vr_camera2_Rbc2.T, vr_camera1_tbc1 - vr_camera2_tbc2)
    # Rc2w_2 = np.dot(Rc2c1,fresh_Rc1w)
    # tc2w_2 = np.dot(Rc2c1,fresh_tc1w).reshape(1,3) + tc2c1
    # rc2w_2, _ = cv2.Rodrigues(Rc2w_2)

    # Ec1c11Pc1 = Ec2c1Pc2, Ewc1Pw = Pc1  Ewc1Pw = Ec1c1Pc1.inv · Ec1c2 · Pc2,    Ec1c2.inv · Ec1c1 · Ewc1 · Pw = Pc2
    rc2w_200, tc2w_200 = E_from_a_2_b(vr_camera1_Rbc1, vr_camera1_tbc1, vr_camera2_Rbc2, vr_camera2_tbc2, fresh_Rc1w, fresh_tc1w)
    image2_points_from_1, _ = cv2.fisheye.projectPoints(np.array(whole_world_apriltag_points).reshape(1, -1, 3), rc2w_200, tc2w_200, vr_camera2_K, vr_camera2_D)
    print("rc2w_2: ", rc2w_200.T, "------>", "fresh_rc2w", fresh_rc2w)
    print("tc2w_2: ", tc2w_200.T, "------>", "fresh_tc2w", fresh_tc2w.T)

    # 相机1到相机3的姿态变换
    rc3w_1, tc3w_1 = E_from_a_2_b(vr_camera1_Rbc1, vr_camera1_tbc1, vr_camera3_Rbc3, vr_camera3_tbc3, fresh_Rc1w, fresh_tc1w)
    image3_points_from_1, _ = cv2.fisheye.projectPoints(np.array(whole_world_apriltag_points).reshape(1, -1, 3), rc3w_1, tc3w_1, vr_camera3_K, vr_camera3_D)
    print("rc3w_1: ", rc3w_1.T, "------>", "fresh_rc3w", fresh_rc3w)
    print("tc3w_1: ", tc3w_1.T, "------>", "fresh_tc3w", fresh_tc3w.T)

    # 相机1到相机4的姿态变换
    rc4w_1, tc4w_1  = E_from_a_2_b(vr_camera1_Rbc1, vr_camera1_tbc1, vr_camera4_Rbc4, vr_camera4_tbc4, fresh_Rc1w, fresh_tc1w)
    image4_points_from_1, _ = cv2.fisheye.projectPoints(np.array(whole_world_apriltag_points).reshape(1, -1, 3), rc4w_1, tc4w_1, vr_camera4_K, vr_camera4_D)
    print("rc4w_1: ", rc4w_1.T, "------>", "fresh_rc4w", fresh_rc4w)
    print("tc4w_1: ", tc4w_1.T, "------>", "fresh_tc4w", fresh_tc4w.T)
   
   
    # 相机2到相机3的姿态变换    
    rc3w_2, tc3w_2 = E_from_a_2_b(vr_camera2_Rbc2, vr_camera2_tbc2, vr_camera3_Rbc3, vr_camera3_tbc3, fresh_Rc2w, fresh_tc2w)
    image3_points_from_2, _ = cv2.fisheye.projectPoints(np.array(whole_world_apriltag_points).reshape(1, -1, 3), rc3w_2, tc3w_2, vr_camera3_K, vr_camera3_D)

    # 相机2到相机4的姿态变换
    rc4w_2, tc4w_2 = E_from_a_2_b(vr_camera2_Rbc2, vr_camera2_tbc2, vr_camera4_Rbc4, vr_camera4_tbc4, fresh_Rc2w, fresh_tc2w)
    image4_points_from_2, _ = cv2.fisheye.projectPoints(np.array(whole_world_apriltag_points).reshape(1, -1, 3), rc4w_2, tc4w_2, vr_camera4_K, vr_camera4_D)
    
    # 相机3到相机4的姿态变换
    rc4w_3, tc4w_3 = E_from_a_2_b(vr_camera3_Rbc3, vr_camera3_tbc3, vr_camera4_Rbc4, vr_camera4_tbc4, fresh_Rc3w, fresh_tc3w)
    image4_points_from_3, _ = cv2.fisheye.projectPoints(np.array(whole_world_apriltag_points).reshape(1, -1, 3), rc4w_3, tc4w_3, vr_camera4_K, vr_camera4_D)
    
    point_in_camera1_indexs = []
    point_in_camera2_indexs = []
    point_in_camera3_indexs = []
    point_in_camera4_indexs = []
    
    print("length: ", len(image1_points[0]))
    # draw image1 points
    for index in range(0, len(image1_points[0])):
        # 0-640, 0-480
        if image1_points[0][index][0] > 30 and image1_points[0][index][0] < 610 and image1_points[0][index][1] > 30 and image1_points[0][index][1] < 450:    
            if camera1_temp_points[0][index][2] > 0:
                point_in_camera1_indexs.append(index)
                cv2.circle(image_camera_1, (int(image1_points[0][index][0]), int(image1_points[0][index][1])), 2, (0, 0, 255), 2)
    cv2.imwrite("image1_points.jpg", image_camera_1)
    
    # draw image2 points
    for index in range(0, len(image2_points[0])):
        # 0-640, 0-480
        if image2_points[0][index][0] > 30 and image2_points[0][index][0] < 610 and image2_points[0][index][1] > 30 and image2_points[0][index][1] < 450:
            if camera2_temp_points[0][index][2] > 0:
                cv2.circle(image_camera_2, (int(image2_points[0][index][0]), int(image2_points[0][index][1])), 2, (0, 0, 255), 2)
                point_in_camera2_indexs.append(index)
    cv2.imwrite("image2_points.jpg", image_camera_2)

    # draw image3 points
    for index in range(0, len(image3_points[0])):
        # 0-640, 0-480
        if image3_points[0][index][0] > 30 and image3_points[0][index][0] < 610 and image3_points[0][index][1] > 30 and image3_points[0][index][1] < 450:
            if camera3_temp_points[0][index][2] > 0:
                cv2.circle(image_camera_3, (int(image3_points[0][index][0]), int(image3_points[0][index][1])) , 2, (0, 0, 255), 2)
                point_in_camera3_indexs.append(index)
    cv2.imwrite("image3_points.jpg", image_camera_3)
    
    # draw image4 points
    for index in range(0, len(image4_points[0])):
        # 0-640, 0-480
        if image4_points[0][index][0] > 30 and image4_points[0][index][0] < 610 and image4_points[0][index][1] > 30 and image4_points[0][index][1] < 450:
            if camera4_temp_points[0][index][2] > 0:
                cv2.circle(image_camera_4, (int(image4_points[0][index][0]), int(image4_points[0][index][1])) , 2, (0, 0, 255), 2)
                point_in_camera4_indexs.append(index)
    cv2.imwrite("image4_points.jpg", image_camera_4)
    
    
    #combine image1 & image2 into one image
    combine_1_2 = np.hstack((image_camera_1, image_camera_2))
    combine_1_2 = cv2.cvtColor(combine_1_2, cv2.COLOR_GRAY2BGR)
    # draw image1 -> image2 common points and line
    distances_list12 = []
    index_list12 = []
    for index1 in point_in_camera1_indexs:
        if index1 in point_in_camera2_indexs:
            distance = np.sqrt(np.square(image2_points[0][index1][0] - image2_points_from_1[0][index1][0]) + np.square(image2_points[0][index1][1] - image2_points_from_1[0][index1][1]))
            distances_list12.append(distance)
            index_list12.append(index1)
            color = (0, 255, 0)
            if distance < 3: # green
                color = (0, 255, 0)
            elif distance < 4: # yellow
                color = (0, 255, 255)
            elif distance > 5: # red
                color = (0, 0, 255)
            # cv2.circle(combine_1_2, (int(image2_points_from_1[0][index1][0]) + 640, int(image2_points_from_1[0][index1][1])), 2, (255, 0, 0), 2)
            cv2.circle(combine_1_2, (int(image1_points[0][index1][0]), int(image1_points[0][index1][1])), 2, (0, 255, 0), 2)
            cv2.circle(combine_1_2, (int(image2_points[0][index1][0]) + 640, int(image2_points[0][index1][1])), 2, color, 2)

    # 绘制网格线，一个正方形是 60 * 60
    for i in range(640, 1280, 60):
        cv2.line(combine_1_2, (i, 0), (i, 480), (255, 255, 255), 1)
    for i in range(0, 480, 60):
        cv2.line(combine_1_2, (640, i), (1280, i), (255, 255, 255), 1)
    
    #判断每个网格内的点的个数
    for i in range(0, 480, 60):
        for j in range(640, 1280, 60):
            count = 0
            distance = 0
            for index in range(0, len(index_list12)):
                if image2_points[0][index_list12[index]][0] + 640 > j and image2_points[0][index_list12[index]][0] + 640 < j + 60 and image2_points[0][index_list12[index]][1] > i and image2_points[0][index_list12[index]][1] < i + 60:
                    count = count + 1
                    distance = distance + distances_list12[index]
            avg_distance = 0
            if count > 0:
                avg_distance = int(distance / count)
                # print("avg_distance = {} / {} = {}".format(distance, count, avg_distance))
            cv2.putText(combine_1_2, str(avg_distance), (j, i + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)    
    
    cv2.imwrite("combine_1_2.jpg", combine_1_2) 
    
    #combine image1 & image3 into one image
    combine_1_3 = np.hstack((image_camera_1, image_camera_3))
    combine_1_3 = cv2.cvtColor(combine_1_3, cv2.COLOR_GRAY2BGR)
    # draw image1 -> image3 common points and line
    distances_list13 = []
    index_list13 = []
    for index1 in point_in_camera1_indexs:
        if index1 in point_in_camera3_indexs:
            distance = np.sqrt(np.square(image3_points[0][index1][0] - image3_points_from_1[0][index1][0]) + np.square(image3_points[0][index1][1] - image3_points_from_1[0][index1][1]))
            # print("distance = {}".format(distance))
            distances_list13.append(distance)
            index_list13.append(index1)
            color = (0, 255, 0)
            if distance < 3: # green
                color = (0, 255, 0)
            elif distance < 4: # yellow
                color = (0, 255, 255)
            elif distance > 5: # red
                color = (0, 0, 255)
            cv2.circle(combine_1_3, (int(image3_points_from_1[0][index1][0]) + 640, int(image3_points_from_1[0][index1][1])), 2, (255, 0, 0), 2)
            cv2.circle(combine_1_3, (int(image1_points[0][index1][0]), int(image1_points[0][index1][1])), 2, (0, 255, 0), 2)
            cv2.circle(combine_1_3, (int(image3_points[0][index1][0]) + 640, int(image3_points[0][index1][1])), 2, color, 2)
            
    # 绘制网格线，一个正方形是 60 * 60
    for i in range(640, 1280, 60):
        cv2.line(combine_1_3, (i, 0), (i, 480), (255, 255, 255), 1)
    for i in range(0, 480, 60):
        cv2.line(combine_1_3, (640, i), (1280, i), (255, 255, 255), 1)
        
    #判断每个网格内的点的个数
    for i in range(0, 480, 60):
        for j in range(640, 1280, 60):
            count = 0
            distance = 0
            for index in range(0, len(index_list13)):
                if image3_points[0][index_list13[index]][0] + 640 > j and image3_points[0][index_list13[index]][0] + 640 < j + 60 and image3_points[0][index_list13[index]][1] > i and image3_points[0][index_list13[index]][1] < i + 60:
                    count = count + 1
                    distance = distance + distances_list13[index]
            avg_distance = 0
            if count > 0:
                avg_distance = int(distance / count)
            cv2.putText(combine_1_3, str(avg_distance), (j, i + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite("combine_1_3.jpg", combine_1_3)
    
    #combine image1 & image4 into one image
    combine_1_4 = np.hstack((image_camera_1, image_camera_4))
    combine_1_4 = cv2.cvtColor(combine_1_4, cv2.COLOR_GRAY2BGR)
    # draw image1 -> image4 common points and line
    distances_list14 = []
    index_list14 = []
    for index1 in point_in_camera1_indexs:
        if index1 in point_in_camera4_indexs:
            distance = np.sqrt(np.square(image4_points[0][index1][0] - image4_points_from_1[0][index1][0]) + np.square(image4_points[0][index1][1] - image4_points_from_1[0][index1][1]))
            distances_list14.append(distance)
            index_list14.append(index1)
            color = (0, 255, 0)
            if distance < 3: # green
                color = (0, 255, 0)
            elif distance < 4: # yellow
                color = (0, 255, 255)
            elif distance > 5: # red
                color = (0, 0, 255)
            cv2.circle(combine_1_4, (int(image1_points[0][index1][0]), int(image1_points[0][index1][1])), 2, (0, 255, 0), 2)
            cv2.circle(combine_1_4, (int(image4_points[0][index1][0]) + 640, int(image4_points[0][index1][1])), 2, color, 2)
            
    print("distance_list14:", distances_list14)
    # 绘制网格线，一个正方形是 60 * 60
    for i in range(640, 1280, 60):
        cv2.line(combine_1_4, (i, 0), (i, 480), (255, 255, 255), 1)
    for i in range(0, 480, 60):
        cv2.line(combine_1_4, (640, i), (1280, i), (255, 255, 255), 1)
        
    #判断每个网格内的点的个数
    for i in range(0, 480, 60):
        for j in range(640, 1280, 60):
            count = 0
            distance = 0
            for index in range(0, len(index_list14)):
                if image4_points[0][index_list14[index]][0] + 640 > j and image4_points[0][index_list14[index]][0] + 640 < j + 60 and image4_points[0][index_list14[index]][1] > i and image4_points[0][index_list14[index]][1] < i + 60:
                    count = count + 1
                    distance = distance + distances_list14[index]
            avg_distance = 0
            if count > 0:
                avg_distance = int(distance / count)
            cv2.putText(combine_1_4, str(avg_distance), (j, i + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite("combine_1_4.jpg", combine_1_4)
    
    
    
    #combine image2 & image3 into one image
    combine_2_3 = np.hstack((image_camera_2, image_camera_3))
    combine_2_3 = cv2.cvtColor(combine_2_3, cv2.COLOR_GRAY2BGR)
    # draw image2 -> image3 common points and line
    distances_list23 = []
    index_list23 = []
    for index2 in point_in_camera2_indexs:
        if index2 in point_in_camera3_indexs:
            distance = np.sqrt(np.square(image3_points[0][index2][0] - image3_points_from_2[0][index2][0]) + np.square(image3_points[0][index2][1] - image3_points_from_2[0][index2][1]))
            distances_list23.append(distance)  
            index_list23.append(index2)
            color = (0, 255, 0)
            if distance < 3:
                color = (0, 255, 0)
            elif distance < 4:
                color = (0, 255, 255)
            elif distance > 5:
                color = (0, 0, 255)
            cv2.circle(combine_2_3, (int(image2_points[0][index2][0]), int(image2_points[0][index2][1])), 2, (0, 255, 0), 2)
            cv2.circle(combine_2_3, (int(image3_points[0][index2][0]) + 640, int(image3_points[0][index2][1])), 2, color, 2)
    
    # 绘制网格线，一个正方形是 60 * 60
    for i in range(640, 1280, 60):
        cv2.line(combine_2_3, (i, 0), (i, 480), (255, 255, 255), 1)
    for i in range(0, 480, 60):
        cv2.line(combine_2_3, (640, i), (1280, i), (255, 255, 255), 1)
        
    #判断每个网格内的点的个数
    for i in range(0, 480, 60):
        for j in range(640, 1280, 60):
            count = 0
            distance = 0
            for index in range(0, len(index_list23)):
                if image3_points[0][index_list23[index]][0] + 640 > j and image3_points[0][index_list23[index]][0] + 640 < j + 60 and image3_points[0][index_list23[index]][1] > i and image3_points[0][index_list23[index]][1] < i + 60:
                    count = count + 1
                    distance = distance + distances_list23[index]
            avg_distance = 0
            if count > 0:
                avg_distance = int(distance / count)
            cv2.putText(combine_2_3, str(avg_distance), (j, i + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite("combine_2_3.jpg", combine_2_3)
    
    #combine image2 & image4 into one image
    combine_2_4 = np.hstack((image_camera_2, image_camera_4))
    combine_2_4 = cv2.cvtColor(combine_2_4, cv2.COLOR_GRAY2BGR)
    distances_list24 = []
    index_list24 = []
    # draw image2 -> image4 common points and line
    for index2 in point_in_camera2_indexs:
        if index2 in point_in_camera4_indexs:
            distance = np.sqrt(np.square(image4_points[0][index2][0] - image4_points_from_2[0][index2][0]) + np.square(image4_points[0][index2][1] - image4_points_from_2[0][index2][1]))
            distances_list24.append(distance)
            index_list24.append(index2)
            color = (0, 255, 0)
            if distance < 3:
                color = (0, 255, 0)
            elif distance < 4:
                color = (0, 255, 255)
            elif distance > 5:
                color = (0, 0, 255)
            cv2.circle(combine_2_4, (int(image2_points[0][index2][0]), int(image2_points[0][index2][1])), 2, (0, 255, 0), 2)
            cv2.circle(combine_2_4, (int(image4_points[0][index2][0]) + 640, int(image4_points[0][index2][1])), 2, color, 2)
            
    # 绘制网格线，一个正方形是 60 * 60
    for i in range(640, 1280, 60):
        cv2.line(combine_2_4, (i, 0), (i, 480), (255, 255, 255), 1)
    for i in range(0, 480, 60):
        cv2.line(combine_2_4, (640, i), (1280, i), (255, 255, 255), 1)
        
    #判断每个网格内的点的个数
    for i in range(0, 480, 60):
        for j in range(640, 1280, 60):
            count = 0
            distance = 0
            for index in range(0, len(index_list24)):
                if image4_points[0][index_list24[index]][0] + 640 > j and image4_points[0][index_list24[index]][0] + 640 < j + 60 and image4_points[0][index_list24[index]][1] > i and image4_points[0][index_list24[index]][1] < i + 60:
                    count = count + 1
                    distance = distance + distances_list24[index]
            avg_distance = 0
            if count > 0:
                avg_distance = int(distance / count)
            cv2.putText(combine_2_4, str(avg_distance), (j, i + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
    cv2.imwrite("combine_2_4.jpg", combine_2_4)
    
    #combine image3 & image4 into one image
    combine_3_4 = np.hstack((image_camera_3, image_camera_4))
    combine_3_4 = cv2.cvtColor(combine_3_4, cv2.COLOR_GRAY2BGR)
    
    # draw image3 -> image4 common points and line
    distances_list34 = []
    index_list34 = []
    for index3 in point_in_camera3_indexs:
        if index3 in point_in_camera4_indexs:
            distance = np.sqrt(np.square(image4_points[0][index3][0] - image4_points_from_3[0][index3][0]) + np.square(image4_points[0][index3][1] - image4_points_from_3[0][index3][1]))
            distances_list34.append(distance)
            index_list34.append(index3)
            color = (0, 255, 0)
            if distance < 3:
                color = (0, 255, 0)
            elif distance < 4:
                color = (0, 255, 255)
            elif distance > 5:
                color = (0, 0, 255)
            cv2.circle(combine_3_4, (int(image3_points[0][index3][0]), int(image3_points[0][index3][1])), 2, (0, 255, 0), 2)
            cv2.circle(combine_3_4, (int(image4_points[0][index3][0]) + 640, int(image4_points[0][index3][1])), 2, color, 2)
            
    # 绘制网格线，一个正方形是 60 * 60
    for i in range(640, 1280, 60):
        cv2.line(combine_3_4, (i, 0), (i, 480), (255, 255, 255), 1)
    for i in range(0, 480, 60):
        cv2.line(combine_3_4, (640, i), (1280, i), (255, 255, 255), 1)
        
    #判断每个网格内的点的个数
    for i in range(0, 480, 60):
        for j in range(640, 1280, 60):
            count = 0
            distance = 0
            for index in range(0, len(index_list34)):
                if image4_points[0][index_list34[index]][0] + 640 > j and image4_points[0][index_list34[index]][0] + 640 < j + 60 and image4_points[0][index_list34[index]][1] > i and image4_points[0][index_list34[index]][1] < i + 60:
                    count = count + 1
                    distance = distance + distances_list34[index]
            avg_distance = 0
            if count > 0:
                avg_distance = int(distance / count)
            cv2.putText(combine_3_4, str(avg_distance), (j, i + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite("combine_3_4.jpg", combine_3_4)
    

    