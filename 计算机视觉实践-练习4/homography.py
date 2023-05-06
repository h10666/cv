import cv2
import numpy as np
class GetRoiMouse():

    def __init__(self, img):
        self.lsPointsChoose = []
        self.tpPointsChoose = []
        self.pointsCount = 0  # 顶点计数
        self.pointsMax = 4  # 最大顶点个数
        self.mouseWindowName = 'get four top'
        self.img = img  # 输入的图像

    def mouseclick(self):  # 显示一个窗口完成标点
        cv2.namedWindow(self.mouseWindowName)
        # 设置点击鼠标时进行的处理
        cv2.setMouseCallback(self.mouseWindowName, self.on_mouse)

        cv2.imshow(self.mouseWindowName, self.img)
        cv2.waitKey(0)

	# 检测当前点个数，满足要求时关闭图像显示窗口
    def checkPointsNum(self):
        if len(self.lsPointsChoose) == 4:
            print('I get 4 points!')
            cv2.destroyAllWindows()

    # OpenCV的鼠标响应函数，可以在内部定义鼠标的各种响应
    def on_mouse(self, event, x, y, flags, param):
        # 左键点击
        if event == cv2.EVENT_LBUTTONDOWN:
            print('left-mouse')
            self.pointsCount += 1
            print(self.pointsCount)
            point1 = (x, y)
            # 画出点击的位置
            img1 = self.img.copy()
            cv2.circle(img1, point1, 10, (0, 255, 0), 2)
            self.lsPointsChoose.append([x, y])
            self.tpPointsChoose.append((x, y))
            # 将鼠标选的点用直线连起来
            for i in range(len(self.tpPointsChoose) - 1):
                cv2.line(img1, self.tpPointsChoose[i], self.tpPointsChoose[i + 1], (0, 0, 255), 2)
            cv2.imshow(self.mouseWindowName, img1)
            self.checkPointsNum()


def normalize(points):
    """
    对输入的点进行归一化
    """
    centroid = np.mean(points, axis=0)
    std = np.std(points)
    return (points - centroid) / std

def get_homography(p1, p2):
    """
    通过输入的点计算 Homography 矩阵
    """
    matrixIndex = 0
    A = np.zeros((8, 9))
    for i in range(0, len(p1)):
        x = p1[i][0]
        y = p1[i][1]

        u = p2[i][0]
        v = p2[i][1]

        A[matrixIndex] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
        A[matrixIndex + 1] = [x, y, 1, 0, 0, 0, -u * x, -u * y, -u]

        matrixIndex = matrixIndex + 2

    U, s, V = np.linalg.svd(A, full_matrices=True)
    matrix = V[:, 8].reshape(3, 3)
    return matrix

if __name__ == '__main__':
    # 读取原图像
    img_src = cv2.imread('./photos/3.jpg')
    img_src = cv2.resize(img_src, (1200, 900))
    # 调用上述类，获取原图像的像素点
    mouse1 = GetRoiMouse(img_src)
    mouse1.mouseclick()
    # 为方便下一步运算，需要将像素点的类型转换为浮点型数据
    pts_src = np.float32(mouse1.lsPointsChoose)

    # 读取目标图像
    img_dst = cv2.imread('./photos/4.jpg')
    img_dst = cv2.resize(img_dst, (1200, 900))
    mouse2 = GetRoiMouse(img_dst)
    mouse2.mouseclick()
    # 获取对应点
    pts_dst = np.float32(mouse2.lsPointsChoose)
    # ----------------------------------------------------
    # 目标图像的尺寸
    dw, dh = img_dst.shape[1], img_dst.shape[0]
    # 通过findHomography计算变换矩阵h
    h = get_homography(pts_src, pts_dst)
    # 将变换矩阵h带入仿射变换实现矫正
    img_out = cv2.warpPerspective(img_src, h, (dw, dh))
    cv2.imwrite('result.jpg', img_out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

