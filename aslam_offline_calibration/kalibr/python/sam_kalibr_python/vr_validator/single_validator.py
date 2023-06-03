import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
from PIL import Image

class VRCamera:
    def __init__(self, Size, CameraMatrix: np.ndarray, DistortMatrix: np.ndarray, Extrinsics_R: np.ndarray, Extrinsics_T: np.ndarray):
        self.size = Size
        self.K = CameraMatrix
        self.D = DistortMatrix
        self.E_R = Extrinsics_R
        self.E_T = Extrinsics_T

    def count_image_point_ref(self, VRCamera2, image_point):
        # 将此相机看到的点投影到相机2的图像上
        # 假设该点真实世界坐标系下的位置为Pw
        # P = R * Pw + T
        # Pi = K * P
        # Zpi = K * E * Pw, Pw = inv(K * E) * Zpi
        # Pw = inv(K1 * E1) * Z1pi1, Pw = inv(K2 * E2) * Z2pi2
        # Pi2 = K2 * E2 * inv(K1 * E1) * pi1 * (Z1 / Z2)
        pi1 = np.array([image_point[0], image_point[1], 1])
        Pw = np.dot(np.linalg.inv(np.dot(self.K, self.E_R)), (pi1 - np.dot(self.K, self.E_T)))
        print(Pw)
        pi2 = np.dot(np.dot(VRCamera2.K, VRCamera2.E_R), Pw) + np.dot(VRCamera2.K, VRCamera2.E_T)
        print(np.dot(self.K, self.E_T))
        return pi2

    # def count_world_point_ref(self, VRCamera2, world_point):
        

# <Rig translation="0 0 0 " rowMajorRotationMat="1 0 0 0 1 0 0 0 1 " />
# <Rig translation="-0.00038707237 0.10318894 -0.01401102 " rowMajorRotationMat="-0.99949765 -0.025876258 0.018299539 0.029883759 -0.96176534 0.27223959 0.010555321 0.27264969 0.96205547 " />
# <Rig translation="-0.0056250621 0.041670205 -0.013388009 " rowMajorRotationMat="0.5687224 0.81622218 -0.10166702 -0.3726157 0.36585493 0.85282338 0.73328874 -0.44713703 0.51220709 " />
# <Rig translation="-0.079096205 0.064828795 -0.066688745 " rowMajorRotationMat="-0.59714752 0.79234879 -0.12489289 -0.35174627 -0.11873178 0.92853504 0.72089486 0.598403 0.34960612 " />

vr_camera1_K = np.array([[276.28156, 0, 317.99796], [0, 276.28156, 240.97645], [0, 0, 1]])
vr_camera1_D = np.array([-0.013057856, 0.026553879, -0.01489986, 0.0031719355])
vr_camera1_R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
vr_camera1_T = np.array([[0], [0], [0]])
vr_camera_trackingA = VRCamera((640, 480), vr_camera1_K, vr_camera1_D, vr_camera1_R, vr_camera1_T)

vr_camera2_K = np.array([[276.44363, 0, 318.01495], [0, 276.44363, 240.67293], [0, 0, 1]])
vr_camera2_D = np.array([0.0051421098, -0.0088336899, 0.0097894818, -0.0032491733])
vr_camera2_R = np.array([[0.99949765, 0.029883759, 0.010555321], [-0.025876258, -0.96176534, 0.27264969], [0.018299539, 0.27223959, 0.96205547]])
vr_camera2_T = np.array([[-0.00038707237], [0.10318894], [-0.01401102]])
vr_camera_trackingB = VRCamera((640, 480), vr_camera2_K, vr_camera2_D, vr_camera2_R, vr_camera2_T)

vr_camera3_K = np.array([[273.87003, 0, 317.51742], [0, 273.87003, 238.06963], [0, 0, 1]])
vr_camera3_D = np.array([0.006437201, -0.015906533, 0.021496724, -0.0065812969])
vr_camera3_R = np.array([[0.5687224, -0.3726157, 0.73328874], [0.81622218, 0.36585493, -0.44713703], [-0.10166702, 0.85282338, 0.51220709]])
vr_camera3_T = np.array([[-0.0056250621], [0.041670205], [-0.013388009]])
vr_camera_control_trackingA = VRCamera((640, 480), vr_camera3_K, vr_camera3_D, vr_camera3_R, vr_camera3_T)

vr_camera4_K = np.array([[277.06611, 0, 317.67797], [0, 277.06611, 238.94741], [0, 0, 1]])
vr_camera4_D = np.array([0.026178562, -0.053109878, 0.041701285, -0.010932579])
vr_camera4_R = np.array([[-0.59714752, -0.35174627, 0.72089486], [0.79234879, -0.11873178, 0.598403], [-0.12489289, 0.92853504, 0.34960612]])
vr_camera4_T = np.array([[-0.079096205], [0.064828795], [-0.066688745]])
vr_camera_control_trackingB = VRCamera((640, 480), vr_camera4_K, vr_camera4_D, vr_camera4_R, vr_camera4_T)


point_control_trackingA = np.array([468.08209229, 295.57080078])
point_control_trackingB = vr_camera_control_trackingA.count_image_point_ref(vr_camera_control_trackingB, point_control_trackingA)
print(point_control_trackingB)

# world point tag1 = (0, 0, 0) to image point
# true_world_point = np.array([0, 0, 0])
# 世界点1 到相机3的投影
# r2: [[-1.31614046], [2.66412007], [-0.45082514]]
# t2: [[0.21337044], [0.11876129], [0.27692042]]

r_world_to_camera3 = np.array([[-1.31614046], [2.66412007], [-0.45082514]])
R_world_to_camera3, _ = cv2.Rodrigues(r_world_to_camera3)
t_world_to_camera3 = np.array([[0.21337044], [0.11876129], [0.27692042]])

# world point tag1 = (0, 0, 0) to camera3 point
# R_world = np.dot(np.linalg.inv(R1), R2)
# T_world = t2 - t1

# 已知相机3坐标系到世界坐标系的外参 R_world_to_camera3, t_world_to_camera3
# 已知相机3和相机1的外参，以及相机3到世界坐标系下的外参，将世界坐标系下的点投影到相机1坐标系下
R_camera3_to_camera1 = np.dot(np.linalg.inv(vr_camera1_R), vr_camera3_R)
T_camera3_to_camera1 = vr_camera3_T - vr_camera1_T
R_world_to_camera1 = np.dot(R_world_to_camera3, R_camera3_to_camera1)
T_world_to_camera1 = T_camera3_to_camera1 + t_world_to_camera3
objectPoints = np.dot(R_world_to_camera1, np.array([0, 0, 0]).reshape(3, 1)) + T_world_to_camera1
print("objectPoints: ", objectPoints)

camera4_r, _ = cv2.Rodrigues(vr_camera4_R)
imagePoints, jacobian = cv2.fisheye.projectPoints(objectPoints.reshape(1, -1, 3), camera4_r, vr_camera4_T, vr_camera4_K, vr_camera4_D)
print("imagePoints: ", imagePoints)

img = Image.open("4252229514797.pgm")
# 转换为numpy数组
np_image = np.array(img)
# 创建OpenCV的Mat对象
half = np_image.shape[1] // 2
left_img = np_image[:, :half]
right_img = np_image[:, half:]

target_right_image = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
cv2.circle(target_right_image, (int(276.13261332), int(546.71144042)), 4, (0, 0, 255), -1)
cv2.imwrite("target_right_image.jpg", target_right_image)



# 相机3系到相机4系的变换
# true_world_poitn1 = np.array