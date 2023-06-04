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

        self.bottom_left3d = tag_point_worlds[self.markerId*4]
        self.bottom_right3d = tag_point_worlds[self.markerId*4 + 1]
        self.top_right3d = tag_point_worlds[self.markerId*4 + 2]
        self.top_left3d = tag_point_worlds[self.markerId*4 + 3]

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

    # 右下
    vr_camera2_K = np.array([[276.44363, 0, 318.01495], [0, 276.44363, 240.67293], [0, 0, 1]])
    vr_camera2_D = np.array([0.0051421098, -0.0088336899, 0.0097894818, -0.0032491733])
    vr_camera2_Rbc2 = np.array([[-0.99949765, -0.025876258, 0.018299539], [0.029883759, -0.96176534, 0.27223959], [0.010555321, 0.27264969, 0.96205547]])
    vr_camera2_tbc2 = np.array([-0.00038707237, 0.10318894, -0.01401102])
 
    # 左上
    # <Rig translation="-0.0056250621 0.041670205 -0.013388009 " rowMajorRotationMat="0.5687224 0.81622218 -0.10166702 -0.3726157 0.36585493 0.85282338 0.73328874 -0.44713703 0.51220709 " />
    vr_camera3_K = np.array([[273.87003, 0, 317.51742], [0, 273.87003, 238.06963], [0, 0, 1]])
    vr_camera3_D = np.array([0.006437201, -0.015906533, 0.021496724, -0.0065812969])
    vr_camera3_Rbc3 = np.array([[0.5687224, 0.81622218, -0.10166702], [-0.3726157, 0.36585493, 0.85282338], [0.73328874, -0.44713703, 0.51220709]])
    vr_camera3_tbc3 = np.array([-0.0056250621, 0.041670205, -0.013388009])
 
    # 右上
    # <Rig translation="-0.079096205 0.064828795 -0.066688745 " rowMajorRotationMat="-0.59714752 0.79234879 -0.12489289 -0.35174627 -0.11873178 0.92853504 0.72089486 0.598403 0.34960612 " />
    vr_camera4_K = np.array([[277.06611, 0, 317.67797], [0, 277.06611, 238.94741], [0, 0, 1]])
    vr_camera4_D = np.array([0.026178562, -0.053109878, 0.041701285, -0.010932579])
    vr_camera4_Rbc4 = np.array([[-0.59714752, 0.79234879, -0.12489289], [-0.35174627, -0.11873178, 0.92853504], [0.72089486, 0.598403, 0.34960612]])
    vr_camera4_tbc4 = np.array([-0.079096205, 0.064828795, -0.066688745])
    
    # 相机标定内参，畸变参数
    camchain_file = "./camchain-trackingA.yaml"
    #左侧板
    target_board3_file = "april_6x6_all.yaml"
    
    result_output_file_left = "finded_tag_indexs_left.txt"
    result_output_file_right = "finded_tag_indexs_right.txt"
    
    cmd_parse_l = "python3 ./vr_validator_left.py --target {} --cam {}  --camera-index 3 --image {} > {}"
    cmd_parse_r = "python3 ./vr_validator_right.py --target {} --cam {} --camera-index 4 --image {} > {}"

    image_file = "./4252229514797.pgm"
    # image_file = "4250462839129.pgm"
    
    
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
        april_box_left_list.append(AprilBox(left_tag_id, top_left, top_right, bottom_left, bottom_right))
    
    # step 2.2 右侧
    april_box_right_list = []
    for i in range(len(right_tag_ids)):
        right_tag_id = right_tag_ids[i]
        bottom_left = right_image_corners_distorted[i * 4]
        bottom_right = right_image_corners_distorted[i * 4 + 1]
        top_right = right_image_corners_distorted[i * 4 + 2]
        top_left = right_image_corners_distorted[i * 4 + 3]
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
    
    left_undistorted_image_points = cv2.fisheye.undistortPoints(left_board_image_point.reshape(-1, 1, 2), vr_camera1_K, vr_camera1_D)
    # left_undistorted_image_points = left_image_corners_undistorted
    # 已知
    
    # 无畸变图像上求解相机姿态
    IK = np.eye(3)
    ID = np.zeros((1,5))
    ret, rc1w, tc1w = cv2.solvePnP(left_board_world_point, left_undistorted_image_points, IK, ID)
    # 根据相机姿态求回畸变图像上的点
    temp_points, _ = cv2.projectPoints(left_board_world_point.reshape(1, -1, 3), rc1w, tc1w, IK, ID)
    # 含畸变的像素点
    # temp_points, _ = cv2.fisheye.projectPoints(left_board_world_point.reshape(1, -1, 3),  rc1w, tc1w, vr_camera1_K, vr_camera1_D)
    print("camera left rvec: ", rc1w.T)
    print("camera left tvec: ", tc1w.T)
    temp_points = temp_points.reshape(-1, 2)

    # draw undistorted points
    img_undistort = Image.open(image_file)
    img_undistort = np.array(img_undistort)
    for i in range(len(temp_points[0])):
        cv2.circle(img_undistort, (int(temp_points[i][0]), int(temp_points[i][1])), 5, (0, 255, 0), -1)
    cv2.imwrite("left_img_undistort.jpg", img_undistort)
    
    # step 3.1.v 左侧内参平均重投影误差验证
    left_average_reprojection_error = 0
    print("len(temp_points): ", len(temp_points))
    for i in range (0, len(temp_points)):
        left_average_reprojection_error += np.linalg.norm(temp_points[i] - left_undistorted_image_points[i])

    print("left_average_reprojection_error = {}, length = {}".format(left_average_reprojection_error, len(temp_points)))
    left_average_reprojection_error /= len(temp_points)

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
    right_undistorted_image_points = cv2.fisheye.undistortPoints(right_board_image_point.reshape(-1,1,2), vr_camera2_K, vr_camera2_D)
    # right_undistorted_image_points = right_image_corners_undistorted
    
    ret, rc2w, tc2w = cv2.solvePnP(right_board_world_point, right_undistorted_image_points, IK, ID)
    temp_points = cv2.projectPoints(right_board_world_point, rc2w, tc2w, IK, ID)
    print("camera right rvec: ", rc2w.T)
    print("camera right tvec: ", tc2w.T)
    
    
    
    # image_file_b = "4252229514797.pgm" # 1,2
    # image_bottom = np.array(Image.open(image_file_b))
    # half_bottom = image_bottom.shape[1] // 2
    # image_camera_1 = image_bottom[:, :half_bottom]
    # whole_world_apriltag_points = [] # 108 * 4
    # for index in range(0, 108):
    #     tag = AprilBox(index,[[0,0]],[[0,0]],[[0,0]],[[0,0]])
    #     whole_world_apriltag_points.append(tag.bottom_left3d)
    #     whole_world_apriltag_points.append(tag.bottom_right3d)
    #     whole_world_apriltag_points.append(tag.top_right3d)
    #     whole_world_apriltag_points.append(tag.top_left3d)

    # image1_points, _ = cv2.fisheye.projectPoints(np.array(whole_world_apriltag_points).reshape(1, -1, 3), rc1w, tc1w, vr_camera1_K, vr_camera1_D)
    # for index in range(0, len(image1_points[0])):
    #     # 0-640, 0-480
    #     if image1_points[0][index][0] > 30 and image1_points[0][index][0] < 610 and image1_points[0][index][1] > 30 and image1_points[0][index][1] < 450:    
    #         # if camera1_temp_points[0][index][2] > 0:
    #         #     point_in_camera1_indexs.append(index)
    #         print("index = {}, point = {}".format(index, image1_points[0][index]))
    #         cv2.circle(image_camera_1, (int(image1_points[0][index][0]), int(image1_points[0][index][1])), 2, (0, 0, 255), 2)
    # cv2.imwrite("image1_points.jpg", image_camera_1)