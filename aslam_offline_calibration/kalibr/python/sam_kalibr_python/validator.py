import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

class AprilBox:
    def __init__(self, markerID):
        self.boardId = int(markerID / 36) + 1
        self.markerId = markerID
        self.x_line_pos = 0
        self.y_line_pos = 0
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

        # print("world pos tagid {}, top_left{}, top_right{}, bottom_left{}, bottom_right{}".format(self.markerId, self.top_left3d, self.top_right3d, self.bottom_left3d, self.bottom_right3d))



# 外参矩阵和平移向量
#board 1
# Rotation matrix:
#  [[ 1.64350904]
#  [ 2.23213605]
#  [-0.76643097]]
# Translation matrix:
#  [[-0.06370602]
#  [-0.20364015]
#  [ 1.58159193]]


t1 = np.array([[-0.06370602], [-0.20364015], [1.58159193]])
r1 = np.array([[1.64350904], [2.23213605], [-0.76643097]])
R1, _ = cv2.Rodrigues(r1)
R = R1
E1 = np.concatenate((R1, t1), axis=1)
E1 = np.concatenate((E1, np.array([[0, 0, 0, 1]])), axis=0)
# print("E1: ", E1)

sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
singular = sy < 1e-6
# print("singular: ",singular)

if  not singular :
    x = math.atan2(R[2,1] , R[2,2])
    y = math.atan2(-R[2,0], sy)
    z = math.atan2(R[1,0], R[0,0])
    # print(np.array([x * 180/math.pi, y* 180/math.pi, z* 180/math.pi]))
else :
    x = math.atan2(-R[1,2], R[1,1])
    y = math.atan2(-R[2,0], sy)
    z = 0
    # print(np.array([x * 180/math.pi, y* 180/math.pi, z* 180/math.pi]))



# Rotation matrix:
#  [[ 1.64081142]
#  [ 2.22375189]
#  [-0.77718015]]
# Translation matrix:
#  [[-0.04476985]
#  [-0.25774712]
#  [ 1.56199604]]

# #board2
t2 = np.array([[-0.04476985], [-0.25774712], [1.56199604]])
r2 = np.array([[1.64081142], [2.22375189], [-0.77718015]])
R2, _ = cv2.Rodrigues(r2)
E2 = np.concatenate((R2, t2), axis=1)
E2 = np.concatenate((E2, np.array([[0, 0, 0, 1]])), axis=0)
# print("E2: ", E2)

R = R2
sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
singular = sy < 1e-6
if  not singular :
    x = math.atan2(R[2,1] , R[2,2])
    y = math.atan2(-R[2,0], sy)
    z = math.atan2(R[1,0], R[0,0])
    # print(np.array([x * 180/math.pi, y* 180/math.pi, z* 180/math.pi]))
else :
    x = math.atan2(-R[1,2], R[1,1])
    y = math.atan2(-R[2,0], sy)
    z = 0
    # print(np.array([x * 180/math.pi, y* 180/math.pi, z* 180/math.pi]))

# Rotation matrix:
#  [[ 1.63879646]
#  [ 2.21011371]
#  [-0.73963377]]
# Translation matrix:
#  [[-0.11122841]
#  [-0.23227544]
#  [ 1.55176322]]
# #board3
t3 = np.array([[-0.11122841], [-0.23227544], [1.55176322]])
r3 = np.array([[1.63879646], [2.21011371], [-0.73963377]])
R3, _ = cv2.Rodrigues(r3)
E3 = np.hstack((R3, t3))
E3 = np.vstack((E3, np.array([0,0,0,1])))

# print(np.dot(E1, E2))
# world2 -> world1
R_world = np.dot(np.linalg.inv(R1), R2)
T_world = np.dot(np.linalg.inv(R1),t2 - t1)
R = R_world

sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
singular = sy < 1e-6
if  not singular :
    x = math.atan2(R[2,1] , R[2,2])
    y = math.atan2(-R[2,0], sy)
    z = math.atan2(R[1,0], R[0,0])
    # print(np.array([x * 180/math.pi, y* 180/math.pi, z* 180/math.pi]))
else :
    x = math.atan2(-R[1,2], R[1,1])
    y = math.atan2(-R[2,0], sy)
    z = 0
    # print(np.array([x * 180/math.pi, y* 180/math.pi, z* 180/math.pi]))
# print(R_world)
# print(T_world)


# pos_36 = [0.0, 0.0, 0.0, 1]
# Pc = np.dot(np.linalg.inv(E2), pos_36)
# Pb = np.dot(E1, Pc)
# print("PB!!:", Pb)
apriltag_world_point_list = []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for tag_id in range(0, 36):
    tag = AprilBox(tag_id)
    world_pos_board1 = [[tag.bottom_left3d[0]], [tag.bottom_left3d[1]], [tag.bottom_left3d[2]], [1]]
    apriltag_world_point_list.append([[tag.bottom_left3d[0]], [tag.bottom_left3d[1]], [tag.bottom_left3d[2]]])
    apriltag_world_point_list.append([[tag.bottom_right3d[0]], [tag.bottom_right3d[1]], [tag.bottom_right3d[2]]])
    apriltag_world_point_list.append([[tag.top_right3d[0]], [tag.top_right3d[1]], [tag.top_right3d[2]]])
    apriltag_world_point_list.append([[tag.top_left3d[0]], [tag.top_left3d[1]], [tag.top_left3d[2]]])
    ax.scatter(world_pos_board1[0], world_pos_board1[1], world_pos_board1[2], c='g')
    # print("tagid {} pos: {}".format(tag_id, world_pos_board1))
    
for tag_id in range(36, 72):
    tag = AprilBox(tag_id)
    # 点在板子2坐标系下的坐标
    world_pos_board2_bottom_left = [[tag.bottom_left3d[0]], [tag.bottom_left3d[1]], [tag.bottom_left3d[2]], [1]]
    world_pos_board2_bottom_right = [[tag.bottom_right3d[0]], [tag.bottom_right3d[1]], [tag.bottom_right3d[2]], [1]]
    world_pos_board2_top_right = [[tag.top_right3d[0]], [tag.top_right3d[1]], [tag.top_right3d[2]], [1]]
    world_pos_board2_top_left = [[tag.top_left3d[0]], [tag.top_left3d[1]], [tag.top_left3d[2]], [1]]
    R_world = np.dot(np.linalg.inv(R1), R2)
    T_world = np.dot(np.linalg.inv(R1),t2 - t1)
    E_ = np.concatenate((R_world, T_world), axis=1)
    E_ = np.concatenate((E_, np.array([[0, 0, 0, 1]])), axis=0)
    # 点在板子1坐标系下的坐标
    world_pos_board1_bottom_left = np.dot(E_, world_pos_board2_bottom_left)
    world_pos_board1_bottom_right = np.dot(E_, world_pos_board2_bottom_right)
    world_pos_board1_top_right = np.dot(E_, world_pos_board2_top_right)
    world_pos_board1_top_left = np.dot(E_, world_pos_board2_top_left)
    
    ax.scatter(world_pos_board1_bottom_left[0], world_pos_board1_bottom_left[1], world_pos_board1_bottom_left[2], c='r')
    
    apriltag_world_point_list.append(world_pos_board1_bottom_left)
    apriltag_world_point_list.append(world_pos_board1_bottom_right)
    apriltag_world_point_list.append(world_pos_board1_top_right)
    apriltag_world_point_list.append(world_pos_board1_top_left)
    
for tag_id in range(72, 108):
    tag = AprilBox(tag_id)
    # 点在板子3坐标系下的坐标
    world_pos_board3_bottom_left = [[tag.bottom_left3d[0]], [tag.bottom_left3d[1]], [tag.bottom_left3d[2]], [1]]
    world_pos_board3_bottom_right = [[tag.bottom_right3d[0]], [tag.bottom_right3d[1]], [tag.bottom_right3d[2]], [1]]
    world_pos_board3_top_right = [[tag.top_right3d[0]], [tag.top_right3d[1]], [tag.top_right3d[2]], [1]]
    world_pos_board3_top_left = [[tag.top_left3d[0]], [tag.top_left3d[1]], [tag.top_left3d[2]], [1]]
    R_world = np.dot(np.linalg.inv(R1), R3)
    T_world = np.dot(np.linalg.inv(R1),t3 - t1)
    E_ = np.concatenate((R_world, T_world), axis=1)
    E_ = np.concatenate((E_, np.array([[0, 0, 0, 1]])), axis=0)
    
    world_pos_board1_bottom_left = np.dot(E_, world_pos_board3_bottom_left)
    world_pos_board1_bottom_right = np.dot(E_, world_pos_board3_bottom_right)
    world_pos_board1_top_right = np.dot(E_, world_pos_board3_top_right)
    world_pos_board1_top_left = np.dot(E_, world_pos_board3_top_left)

    ax.scatter(world_pos_board1_bottom_left[0], world_pos_board1_bottom_left[1], world_pos_board1_bottom_left[2], c='b')
    
    apriltag_world_point_list.append(world_pos_board1_bottom_left)
    apriltag_world_point_list.append(world_pos_board1_bottom_right)
    apriltag_world_point_list.append(world_pos_board1_top_right)
    apriltag_world_point_list.append(world_pos_board1_top_left)
    
for tag_id in range(0, 108):
    # print("tagid ", tag_id)
    # print("point1 ", apriltag_world_point_list[tag_id * 4])
    # print("point2 ", apriltag_world_point_list[tag_id * 4 + 1])
    # print("point3 ", apriltag_world_point_list[tag_id * 4 + 2])
    # print("point4 ", apriltag_world_point_list[tag_id * 4 + 3])
    print("{} [{}, {}, {}], [{}, {}, {}], [{}, {}, {}], [{}, {}, {}]".format(tag_id, apriltag_world_point_list[tag_id * 4][0][0], apriltag_world_point_list[tag_id * 4][1][0], apriltag_world_point_list[tag_id * 4][2][0], apriltag_world_point_list[tag_id * 4 + 1][0][0], apriltag_world_point_list[tag_id * 4 + 1][1][0], apriltag_world_point_list[tag_id * 4 + 1][2][0], apriltag_world_point_list[tag_id * 4 + 2][0][0], apriltag_world_point_list[tag_id * 4 + 2][1][0], apriltag_world_point_list[tag_id * 4 + 2][2][0], apriltag_world_point_list[tag_id * 4 + 3][0][0], apriltag_world_point_list[tag_id * 4 + 3][1][0], apriltag_world_point_list[tag_id * 4 + 3][2][0]))
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


# ============================================================== Handle Images ========================================================

class VRCamera:
    def __init__(self, Size, CameraMatrix, DistortMatrix):
        self.size = Size
        self.K = CameraMatrix
        self.D = DistortMatrix


# <Calibration size="640 480 " principal_point="317.99796 240.97645 " focal_length="276.28156 276.28156 " model="FISHEYE_4_PARAMETERS" radial_distortion="-0.013057856 0.026553879 -0.01489986 0.0031719355 0 0 " distortion_limit="4.040000" undistortion_limit="1.339465" />
# <Calibration size="640 480 " principal_point="318.01495 240.67293 " focal_length="276.44363 276.44363 " model="FISHEYE_4_PARAMETERS" radial_distortion="0.0051421098 -0.0088336899 0.0097894818 -0.0032491733 0 0 " distortion_limit="4.080000" undistortion_limit="1.335519" />
# <Calibration size="640 480 " principal_point="317.67797 238.94741 " focal_length="277.06611 277.06611 " model="FISHEYE_4_PARAMETERS" radial_distortion="0.026178562 -0.053109878 0.041701285 -0.010932579 0 0 " distortion_limit="4.420000" undistortion_limit="1.352619" />

vr_camera1_K = np.array([[276.28156, 0, 317.99796], [0, 276.28156, 240.97645], [0, 0, 1]])
vr_camera1_D = np.array([-0.013057856, 0.026553879, -0.01489986, 0.0031719355])
vr_camera_trackingA = VRCamera((640, 480), vr_camera1_K, vr_camera1_D)

vr_camera2_K = np.array([[276.44363, 0, 318.01495], [0, 276.44363, 240.67293], [0, 0, 1]])
vr_camera2_D = np.array([0.0051421098, -0.0088336899, 0.0097894818, -0.0032491733])
vr_camera_trackingB = VRCamera((640, 480), vr_camera2_K, vr_camera2_D)

vr_camera3_K = np.array([[273.87003, 0, 317.51742], [0, 273.87003, 238.06963], [0, 0, 1]])
vr_camera3_D = np.array([0.006437201, -0.015906533, 0.021496724, -0.0065812969])
vr_camera_control_trackingA = VRCamera((640, 480), vr_camera3_K, vr_camera3_D)

vr_camera4_K = np.array([[277.06611, 0, 317.67797], [0, 277.06611, 238.94741], [0, 0, 1]])
vr_camera4_D = np.array([0.026178562, -0.053109878, 0.041701285, -0.010932579])
vr_camera_control_trackingB = VRCamera((640, 480), vr_camera4_K, vr_camera4_D)

# 检测板子上的tag
# 鱼眼模型 去畸变 + solvePnp 计算外参
# 分开重叠区域tag和非重叠区域tag
# 已知内参外参，非重叠区域3d点投影回像素系，计算像素误差
# 重叠区域分成4张图，计算设备与设备1->2, 1->3 1->4 2->3 2->4 3->4之间的外参变化，将像素中的A点映射到B中, 计算误差









