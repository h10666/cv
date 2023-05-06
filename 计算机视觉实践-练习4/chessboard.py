import cv2
import numpy as np

# 定义棋盘格的行数和列数
num_rows = 5
num_cols = 5

# 生成棋盘格世界坐标系下的坐标
world_points = np.zeros((num_rows * num_cols, 3), np.float32)
world_points[:, :2] = np.mgrid[0:num_cols, 0:num_rows].T.reshape(-1, 2)

# 存储棋盘格角点坐标
image_points = []
# 获取标定图像路径列表
# image_paths = glob.glob('calibration_images/*.jpg')
image_paths = ['./photos/7.jpg', './photos/8.jpg', './photos/9.jpg']
# 遍历每一张标定图像
for image_path in image_paths:
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1200, 900))
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 检测棋盘格角点
    ret, corners = cv2.findChessboardCornersSB(gray_image, (num_cols, num_rows), None)

    # 如果检测到了棋盘格角点，则将其存储
    if ret == True:
        image_points.append(corners)
        # 显示棋盘格角点
        cv2.drawChessboardCorners(image, (num_cols, num_rows), corners, ret)
        cv2.imshow('image', image)
        cv2.waitKey(5000)

# 标定相机
ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(
    [world_points] * len(image_points), image_points, gray_image.shape[::-1], None, None)

# 保存标定结果
np.savez('calibration.npz', camera_matrix=camera_matrix, distortion_coefficients=distortion_coefficients)

# 打印标定结果
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", distortion_coefficients.ravel())
