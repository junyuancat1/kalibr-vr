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
    def __init__(self, markerID, top_left, top_right, bottom_left, bottom_right):
        self.markerId = markerID
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
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

        print("image tagid {}, top_left{}, top_right{}, bottom_left{}, bottom_right{}".format(self.markerId, self.top_left, self.top_right, self.bottom_left, self.bottom_right))
        print("world tagid {}, top_left{}, top_right{}, bottom_left{}, bottom_right{}".format(self.markerId, self.top_left3d, self.top_right3d, self.bottom_left3d, self.bottom_right3d))

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
    vr_camera2_R = np.array([[-0.99949765, -0.025876258, 0.018299539], [0.029883759, -0.96176534, 0.27223959], [0.010555321, 0.27264969, 0.96205547]])
    vr_camera2_T = np.array([-0.00038707237, 0.10318894, -0.01401102])
    
    vr_camera_trackingB = VRCamera((640, 480), vr_camera2_K, vr_camera2_D, vr_camera2_R, vr_camera2_T)

    # <Rig translation="-0.0056250621 0.041670205 -0.013388009 " rowMajorRotationMat="0.5687224 0.81622218 -0.10166702 -0.3726157 0.36585493 0.85282338 0.73328874 -0.44713703 0.51220709 " />
    vr_camera3_K = np.array([[273.87003, 0, 317.51742], [0, 273.87003, 238.06963], [0, 0, 1]])
    vr_camera3_D = np.array([0.006437201, -0.015906533, 0.021496724, -0.0065812969])
    vr_camera3_R = np.array([[0.5687224, 0.81622218, -0.10166702], [-0.3726157, 0.36585493, 0.85282338], [0.73328874, -0.44713703, 0.51220709]])
    vr_camera3_T = np.array([-0.0056250621, 0.041670205, -0.013388009])
    vr_camera_control_trackingA = VRCamera((640, 480), vr_camera3_K, vr_camera3_D, vr_camera3_R, vr_camera3_T)

    # <Rig translation="-0.079096205 0.064828795 -0.066688745 " rowMajorRotationMat="-0.59714752 0.79234879 -0.12489289 -0.35174627 -0.11873178 0.92853504 0.72089486 0.598403 0.34960612 " />
    vr_camera4_K = np.array([[277.06611, 0, 317.67797], [0, 277.06611, 238.94741], [0, 0, 1]])
    vr_camera4_D = np.array([0.026178562, -0.053109878, 0.041701285, -0.010932579])
    vr_camera4_R = np.array([[-0.59714752, 0.79234879, -0.12489289], [-0.35174627, -0.11873178, 0.92853504], [0.72089486, 0.598403, 0.34960612]])
    vr_camera4_T = np.array([-0.079096205, 0.064828795, -0.066688745])
    vr_camera_control_trackingB = VRCamera((640, 480), vr_camera4_K, vr_camera4_D, vr_camera4_R, vr_camera4_T)
    
    # 相机标定内参，畸变参数
    camchain_file = "./camchain-trackingA.yaml"
    #左侧板
    target_board3_file = "april_6x6_all.yaml"
    
    cmd_parse_l = "python3 ./vr_validator_left.py --target {} --cam {} --image {} > {}"
    cmd_parse_r = "python3 ./vr_validator_right.py --target {} --cam {} --image {} > {}"
    
    # image_file = "./4252229514797.pgm"
    image_file = "4250462839129.pgm"
    
    cmd_parse_april_board_l = cmd_parse_l.format(target_board3_file, camchain_file, image_file, "finded_tag_indexs_left.txt")
    cmd_parse_april_board_r = cmd_parse_r.format(target_board3_file, camchain_file, image_file, "finded_tag_indexs_right.txt")
    

    os.system(cmd_parse_april_board_l)
    os.system(cmd_parse_april_board_r)
    

    sm.setLoggingLevel(sm.LoggingLevel.Info)
    targetConfig = kc.ConfigReader.CalibrationTargetParameters(target_board3_file)
    camchain = kc.ConfigReader.CameraChainParameters(camchain_file)
    camConfig = camchain.getCameraParameters(0)
    camera = kc.ConfigReader.AslamCamera.fromParameters(camConfig)
    target = CalibrationTargetDetector(camera, targetConfig)

    img = Image.open(image_file)
    np_image = np.array(img)
    target_image_color = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)
    half = np_image.shape[1] // 2
    left_img = np_image[:, :half]
    undistorted_img = cv2.fisheye.undistortImage(left_img, vr_camera1_K, vr_camera1_D, None, vr_camera1_K, (2000, 2000))
    timestamp = acv.Time(0, 0)
    success, observation = target.detector.findTarget(timestamp, undistorted_img)
    cornersImage = observation.getCornersImageFrame()
    cornersImage = distortPoints(cornersImage, vr_camera1_K, vr_camera1_D)
    
    for index in range(len(cornersImage)):
        corner = cornersImage[index]
        # print("corner: ", corner)
        cv2.circle(target_image_color, (int(corner[0][0]), int(corner[0][1])), 5, (0, 0, 255), -1)
        # cv2.putText(target_image_color, str(index), (int(corner[0][0]), int(corner[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite("validator0_target_image_color.jpg", target_image_color)

    fname = "finded_tag_indexs_left.txt"
    numbers = []
    with open(fname, 'r') as f:  #打开文件
        lines = f.readlines() #读取所有行
        for line in lines:
            if line.strip() == '':
                continue
            # 获取id和坐标
            parts = line.split()
            tag_id = int(parts[0])
            coordinates = [float(x.strip('[],')) for x in parts[1:]]
            print(f'Tag ID: {tag_id}, Coordinates: {coordinates}')

    april_box_list1 = []
    line_head_corner_index = 0
    for i in range(0,6):
        line_tag_counts = 0
        for j in range(0,6):
            tag_id = i * 6 + j
            if tag_id in numbers:
                line_tag_counts += 1

        # print("line 1 got tag numbers ", line_tag_counts)
        tmp_line_tag_count = 0
        for j in range(0,6):
            tag_id = i * 6 + j
            if tag_id in numbers:
                left_bottom_corner_index = line_head_corner_index + 2 * tmp_line_tag_count
                right_bottom_corner_index = line_head_corner_index + 2 * tmp_line_tag_count + 1
                left_up_corner_index = line_head_corner_index + 2 * line_tag_counts + 2 * tmp_line_tag_count
                right_up_corner_index = line_head_corner_index+ 2 * line_tag_counts + 2 * tmp_line_tag_count + 1
                april_box_list1.append(AprilBox(tag_id, cornersImage[left_up_corner_index], cornersImage[right_up_corner_index], cornersImage[left_bottom_corner_index], cornersImage[right_bottom_corner_index]))
                print("tag and corners left, ", tag_id, left_bottom_corner_index, right_bottom_corner_index, left_up_corner_index, right_up_corner_index)
                tmp_line_tag_count += 1
        line_head_corner_index += line_tag_counts * 4

    world_points = np.empty((0,3), dtype=np.float32)
    image_points = np.empty((0,2), dtype=np.float32)

    for tag in april_box_list1:
        # print("handling tagid: ", tag.markerId)
        world_points = np.append(world_points, [np.array([tag.bottom_left3d[0], tag.bottom_left3d[1], tag.bottom_left3d[2]])], axis=0)
        world_points = np.append(world_points, [np.array([tag.bottom_right3d[0], tag.bottom_right3d[1], tag.bottom_right3d[2]])], axis=0)
        world_points = np.append(world_points, [np.array([tag.top_left3d[0], tag.top_left3d[1], tag.top_left3d[2]])], axis=0)
        world_points = np.append(world_points, [np.array([tag.top_right3d[0], tag.top_right3d[1], tag.top_right3d[2]])], axis=0)
        
        image_points = np.append(image_points, [np.array([tag.bottom_left[0][0], tag.bottom_left[0][1]])], axis=0) 
        image_points = np.append(image_points, [np.array([tag.bottom_right[0][0], tag.bottom_right[0][1]])], axis=0) 
        image_points = np.append(image_points, [np.array([tag.top_left[0][0], tag.top_left[0][1]])], axis=0) 
        image_points = np.append(image_points, [np.array([tag.top_right[0][0], tag.top_right[0][1]])], axis=0)    
        
    IK = np.eye(3)
    ID = np.zeros((1,5))
    undistorted_image_points = cv2.fisheye.undistortPoints(image_points.reshape(-1,1,2), vr_camera1_K, vr_camera1_D)
    ret, rvec1, tvec1 = cv2.solvePnP(world_points, undistorted_image_points, IK, ID)
    
    temp_points = cv2.projectPoints(world_points, rvec1, tvec1, IK, ID)
    
    for i in range (0, len(temp_points[0])):
        print("distance between points: ", np.linalg.norm(temp_points[0][i] - undistorted_image_points[i]))
    
    print("=======================================================")
    distort_points = cv2.fisheye.distortPoints(undistorted_image_points, vr_camera1_K, vr_camera1_D)
    for i in range (0, len(distort_points)):
        print("distortion distance between points: ", np.linalg.norm(distort_points[i] - image_points[i]))
    
    print("rvec: ", rvec1)
    print("tvec: ", tvec1)
# ===========================================================================================================

    img = Image.open(image_file)
    # 转换为numpy数组
    np_image = np.array(img)
    # 创建OpenCV的Mat对象
    target_image_color = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)
    half = np_image.shape[1] // 2
    right_img = np_image[:, half:]
    undistorted_img = cv2.fisheye.undistortImage(right_img, vr_camera2_K, vr_camera2_D, None, vr_camera2_K, (2000, 2000))
    timestamp = acv.Time(0, 0)
    success, observation = target.detector.findTarget(timestamp, undistorted_img)
    cornersImage = observation.getCornersImageFrame()
    cornersImage = distortPoints(cornersImage, vr_camera2_K, vr_camera2_D)
    # print("corners: ", cornersImage)
    # print("corners size: ", len(cornersImage))
    
    for index in range(len(cornersImage)):
        corner = cornersImage[index]
        # print("corner: ", corner)
        cv2.circle(target_image_color, (int(corner[0][0] + 640), int(corner[0][1])), 5, (0, 0, 255), -1)
        # cv2.putText(target_image_color, str(index), (int(corner[0][0] + 640), int(corner[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite("validator_target_image_color.jpg", target_image_color)
    
    fname = "finded_tag_indexs_right.txt"
    numbers = []
    with open(fname, 'r') as f:  #打开文件
        lines = f.readlines() #读取所有行
        for line in lines:
            if line.strip() == '':
                continue
            # 获取id和坐标
            parts = line.split()
            tag_id = int(parts[0])
            coordinates = [float(x.strip('[],')) for x in parts[1:]]
            print(f'Tag ID: {tag_id}, Coordinates: {coordinates}')

    april_box_list2 = []
    line_head_corner_index = 0
    for i in range(0,6):
        line_tag_counts = 0
        for j in range(0,6):
            tag_id = i * 6 + j
            if tag_id in numbers:
                line_tag_counts += 1

        # print("line 1 got tag numbers ", line_tag_counts)
        tmp_line_tag_count = 0
        for j in range(0,6):
            tag_id = i * 6 + j
            if tag_id in numbers:
                left_bottom_corner_index = line_head_corner_index + 2 * tmp_line_tag_count
                right_bottom_corner_index = line_head_corner_index + 2 * tmp_line_tag_count + 1
                left_up_corner_index = line_head_corner_index + 2 * line_tag_counts + 2 * tmp_line_tag_count
                right_up_corner_index = line_head_corner_index+ 2 * line_tag_counts + 2 * tmp_line_tag_count + 1
                april_box_list2.append(AprilBox(tag_id, cornersImage[left_up_corner_index], cornersImage[right_up_corner_index], cornersImage[left_bottom_corner_index], cornersImage[right_bottom_corner_index]))
                print("tag and corners right, ", tag_id, left_bottom_corner_index, right_bottom_corner_index, left_up_corner_index, right_up_corner_index)
                tmp_line_tag_count += 1
        line_head_corner_index += line_tag_counts * 4

    world_points = np.empty((0,3), dtype=np.float32)
    image_points = np.empty((0,2), dtype=np.float32)

    for tag in april_box_list2:
        # print("handling tagid: ", tag.markerId)
        world_points = np.append(world_points, [np.array([tag.bottom_left3d[0], tag.bottom_left3d[1], tag.bottom_left3d[2]])], axis=0)
        world_points = np.append(world_points, [np.array([tag.bottom_right3d[0], tag.bottom_right3d[1], tag.bottom_right3d[2]])], axis=0)
        world_points = np.append(world_points, [np.array([tag.top_left3d[0], tag.top_left3d[1], tag.top_left3d[2]])], axis=0)
        world_points = np.append(world_points, [np.array([tag.top_right3d[0], tag.top_right3d[1], tag.top_right3d[2]])], axis=0)
        
        image_points = np.append(image_points, [np.array([tag.bottom_left[0][0], tag.bottom_left[0][1]])], axis=0) 
        image_points = np.append(image_points, [np.array([tag.bottom_right[0][0], tag.bottom_right[0][1]])], axis=0) 
        image_points = np.append(image_points, [np.array([tag.top_left[0][0], tag.top_left[0][1]])], axis=0) 
        image_points = np.append(image_points, [np.array([tag.top_right[0][0], tag.top_right[0][1]])], axis=0)    
        
    IK = np.eye(3)
    ID = np.zeros((1,5))
    undistorted_image_points = cv2.fisheye.undistortPoints(image_points.reshape(-1,1,2), vr_camera2_K, vr_camera2_D)
    ret, rvec2, tvec2 = cv2.solvePnP(world_points, undistorted_image_points, IK, ID)
    print("rvec: ", rvec2)
    print("tvec: ", tvec2)

    # 已知一个世界点Pw, 相信坐标系4是准的，则Pw -> Pc4 -> Pc3 -> Pi3 求出该点在相机3坐标系下的坐标Pi4    
    # step 1 将世界坐标系下的点Pw转换到相机2坐标系下
    R2_world, _ = cv2.Rodrigues(rvec2)
    E2_world = np.hstack((R2_world, tvec2))
    E2_world = np.vstack((E2_world, np.array([0,0,0,1])))
    # step 2 将相机2坐标系下的点Pc2转换到相机1坐标系下
    E2 = np.hstack((vr_camera2_R, vr_camera2_T.reshape(3,1)))
    E2 = np.vstack((E2, np.array([0,0,0,1])))
    E2 = np.linalg.inv(E2)
    
    R2 = E2[0:3, 0:3]
    print("R2: ", R2)
    print("vr_camera2_R: ", vr_camera2_R)
    rvec2_r1, _ = cv2.Rodrigues(vr_camera2_R)
    tvec2_t1 = vr_camera2_T
    print("rvec2_r1: ", rvec2_r1)
    print("tvec2_t1: ", tvec2_t1)


    # print("trans R2_R1: ", R2_R1)
    # E2_E1 = np.hstack((R2_R1, T2_T1))
    # E2_E1 = np.vstack((E2_E1, np.array([0,0,0,1])))
    
    # 将world_points中的点从世界坐标系转换到相机2坐标系下
    world_points2 = np.empty((0,3), dtype=np.float32)
    for world_point in world_points:
        world_point_temp = np.append(world_point, [1])
        world_point_temp = world_point_temp.reshape(4,1)
        point2 = np.dot(E2_world, world_point_temp)
        point2 = point2.reshape(1,4)
        print("world_point: ", world_point, "to camera2: ", point2)
        point2 = np.array([point2[0][0], point2[0][1], point2[0][2]])
        world_points2 = np.append(world_points2, np.array([point2]), axis=0)

    image_point2, _ = cv2.fisheye.projectPoints(world_points2.reshape(1, -1, 3), np.zeros((3,1)), np.zeros((3,1)), vr_camera2_K, vr_camera2_D)
    
    for image_point in image_point2[0]:
        print("image_point: ", image_point)
        cv2.circle(target_image_color, (int(image_point[0] + 640), int(image_point[1])), 5, (0,255,0), -1)
    cv2.imwrite("validator2_target_image_color.jpg", target_image_color)

    # 将world_points2中的点从相机2坐标系转换到相机1坐标系下
    world_points1 = np.empty((0,3), dtype=np.float32)
    for world_point in world_points2:
        world_point_temp = np.append(world_point, [1])
        world_point_temp = world_point_temp.reshape(4,1)
        point1 = np.dot(E2, world_point_temp)
        point1 = point1.reshape(1,4)
        point1 = np.array([point1[0][0], point1[0][1], point1[0][2]])
        print("world_point: ", world_point, "to camera1: ", point1)
        world_points1 = np.append(world_points1, np.array([point1]), axis=0)

    # image_point1, _ = cv2.fisheye.projectPoints(world_points1.reshape(1, -1, 3), np.zeros((3,1)), np.zeros((3,1)), vr_camera1_K, vr_camera1_D)
    image_point1, _ = cv2.fisheye.projectPoints(world_points2.reshape(1, -1, 3), rvec2_r1, tvec2_t1, vr_camera1_K, vr_camera1_D)
    
    
    print("image_point1: ", image_point1)
    for image_point in image_point1[0]:
        print("image_point: ", image_point)
        if image_point[0] < 0 or image_point[0] > 640 or image_point[1] < 0 or image_point[1] > 480:
            print("out of range, becasue{}".format(image_point))
            continue
        cv2.circle(target_image_color, (int(image_point[0]), int(image_point[1])), 5, (255,0,0), -1)
    cv2.imwrite("validator1_target_image_color.jpg", target_image_color)
    
    # check if tagid both in april_box_list1 and april_box_list2
    point1_list = np.empty((0,2), dtype=np.float32)
    point2_list = np.empty((0,2), dtype=np.float32)
    point2_world_list = np.empty((0,3), dtype=np.float32)
    
    for april_board2_index in range(len(april_box_list2)):
        april_board2 = april_box_list2[april_board2_index]
        for april_board1_index in range(len(april_box_list1)):
            april_board1 = april_box_list1[april_board1_index]
            if april_board2.markerId == april_board1.markerId:
                
                print("april_board2.markerId: ", april_board2.markerId)

                point1_list = np.append(point1_list, [np.array([april_board1.bottom_left[0][0], april_board1.bottom_left[0][1]])], axis=0)
                point1_list = np.append(point1_list, [np.array([april_board1.bottom_right[0][0], april_board1.bottom_right[0][1]])], axis=0)
                point1_list = np.append(point1_list, [np.array([april_board1.top_left[0][0], april_board1.top_left[0][1]])], axis=0)
                point1_list = np.append(point1_list, [np.array([april_board1.top_right[0][0], april_board1.top_right[0][1]])], axis=0)
                
                point2_list = np.append(point2_list, [np.array([april_board2.bottom_left[0][0], april_board2.bottom_left[0][1]])], axis=0)
                point2_list = np.append(point2_list, [np.array([april_board2.bottom_right[0][0], april_board2.bottom_right[0][1]])], axis=0)
                point2_list = np.append(point2_list, [np.array([april_board2.top_left[0][0], april_board2.top_left[0][1]])], axis=0)
                point2_list = np.append(point2_list, [np.array([april_board2.top_right[0][0], april_board2.top_right[0][1]])], axis=0)
                
                point2_world_list = np.append(point2_world_list, [np.array([april_board2.bottom_left3d[0], april_board2.bottom_left3d[1], april_board2.bottom_left3d[2]])], axis=0)
                point2_world_list = np.append(point2_world_list, [np.array([april_board2.bottom_right3d[0], april_board2.bottom_right3d[1], april_board2.bottom_right3d[2]])], axis=0)
                point2_world_list = np.append(point2_world_list, [np.array([april_board2.top_left3d[0], april_board2.top_left3d[1], april_board2.top_left3d[2]])], axis=0)
                point2_world_list = np.append(point2_world_list, [np.array([april_board2.top_right3d[0], april_board2.top_right3d[1], april_board2.top_right3d[2]])], axis=0)
                
      
    print("point1_list: ", point1_list)
     # 将world_points中的点从世界坐标系转换到相机2坐标系下
    camera_points2 = np.empty((0,3), dtype=np.float32)
    for world_point in point2_world_list:
        world_point_temp = np.append(world_point, [1])
        world_point_temp = world_point_temp.reshape(4,1)
        point2 = np.dot(E2_world, world_point_temp)
        point2 = point2.reshape(1,4)
        print("world_point: ", world_point, "to camera2: ", point2)
        point2 = np.array([point2[0][0], point2[0][1], point2[0][2]])
        camera_points2 = np.append(camera_points2, np.array([point2]), axis=0)

    image_point1, _ = cv2.fisheye.projectPoints(camera_points2.reshape(1, -1, 3), rvec2_r1, tvec2_t1, vr_camera1_K, vr_camera1_D)
    
    print("image_point1: ", image_point1)
    for index in range(len(image_point1[0])):
        image_point = image_point1[0][index]
        if image_point[0] < 0 or image_point[0] > 640 or image_point[1] < 0 or image_point[1] > 480:
            print("out of range, becasue{}".format(image_point))
            continue
        cv2.circle(target_image_color, (int(image_point[0]), int(image_point[1])), 5, (255,0,0), -1)
        distance = np.sqrt((image_point[0] - point1_list[index][0])**2 + (image_point[1] - point1_list[index][1])**2)
        print("image_point: ", image_point, "point1_list[index]: ", point1_list[index])
        print("distance: ", distance)
        cv2.putText(target_image_color, str(int(distance)), (int(image_point[0]), int(image_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
        # cv2.line(target_image_color, (int(point1_list[index][0]), int(point1_list[index][1])), (int(point2_list[index][0] + 640), int(point2_list[index][1])), (0,0,255), 2)
        
    cv2.imwrite("validator3_target_image_color.jpg", target_image_color)
