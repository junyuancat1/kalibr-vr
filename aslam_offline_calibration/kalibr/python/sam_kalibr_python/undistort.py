import cv2
import numpy as np

# 4220.4207319  4211.72819095 2591.72684113 1756.02132488
# 0.11918242  0.61473932 -1.82388479  2.82637769
# 相机内参和畸变参数
# CameraMatrix = np.float32([[4246.77883083, 0, 4236.89559297],
#                            [0, 2586.61014888, 1739.81219089],
#                            [0, 0, 1]]).reshape((3,3))

# CameraDistortion = np.float32([0.12115487 ,0.53058533 ,-1.40555966 ,2.23803307])



# cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 4220.4207319, 0, 4211.72819095, 0, 2591.72684113, 1756.02132488, 0, 0, 1);
# cv::Mat distCoeffs   = (cv::Mat_<double>(4, 1) << -0.11918242, 0.61473932, -1.82388479, 2.82637769);

# std::cout << "Trying to undistort image " << std::endl;
# cv::Mat undistorted_image;
# cv::fisheye::undistortImage(img, undistorted_image, cameraMatrix, distCoeffs, cameraMatrix, cv::Size(5184, 3456));


CameraMatrix = np.float32([[4220.4207319, 0, 4211.72819095],
                           [0, 2591.72684113, 1756.02132488],
                           [0, 0, 1]]).reshape((3,3))

CameraDistortion = np.float32([0.11918242 ,0.61473932 ,-1.82388479 ,2.82637769])


CameraMatrix = np.float32([[4040.31538579, 0, 2780.82377313],
                           [0, 4147.40542129, 1500.11176288],
                           [0, 0, 1]]).reshape((3,3))

CameraDistortion = np.float32([0.40896237 ,-3.79406948 ,23.96098427 ,-50.45262728])

# 读取图像
img = cv2.imread('result.jpg')
height, width, channels = img.shape

# 去畸变
# undistorted_img = cv2.undistort(img, CameraMatrix, CameraDistortion)
undistorted_img = cv2.fisheye.undistortImage(img, CameraMatrix, CameraDistortion, None, CameraMatrix)
# 显示图像
cv2.imwrite("undistorted_img.jpg", undistorted_img)
