import cv2
import numpy as np


class FindKeyPointsAndMatching:
    def __init__(self):
        # 初始化 SIFT 算法用于特征检测和描述符计算
        self.sift = cv2.xfeatures2d.SIFT_create()
        # 初始化 BFMatcher（暴力匹配器）用于特征匹配
        self.brute = cv2.BFMatcher()

    def get_key_points(self, img1, img2):
        # 将图像转换为灰度图，以便特征检测算法处理
        g_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 使用 SIFT 检测关键点和计算描述符
        kp1, kp2 = {}, {}
        print('=============> Detecting key points!')
        kp1['kp'], kp1['des'] = self.sift.detectAndCompute(g_img1, None)
        kp2['kp'], kp2['des'] = self.sift.detectAndCompute(g_img2, None)

        # 返回两个图像的关键点和描述符
        return kp1, kp2

    def match(self, kp1, kp2):
        print('===========> Match key points!')
        # 使用 KNN 匹配描述符（k=2 表示每次找两个最近邻）
        matches = self.brute.knnMatch(kp1['des'], kp2['des'], k=2)
        good_matches = []

        # 筛选优质匹配点：第一最近邻距离比第二最近邻距离小于 0.7 倍
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good_matches.append((m.trainIdx, m.queryIdx))

        # 如果优质匹配点数量大于 4，则计算单应性矩阵
        if len(good_matches) > 4:
            key_points1 = kp1['kp']
            key_points2 = kp2['kp']

            # 提取匹配点的坐标
            matched_kp1 = np.float32(
                [key_points1[i].pt for (_, i) in good_matches]
            )
            matched_kp2 = np.float32(
                [key_points2[i].pt for (i, _) in good_matches]
            )

            print('=============> Random sampling and computing the homography matrix!')
            # 使用 RANSAC 计算单应性矩阵
            homo_matrix, _ = cv2.findHomography(matched_kp1, matched_kp2, cv2.RANSAC, 4)
            return homo_matrix
        else:
            # 如果匹配点不足，返回 None
            return None


class PasteTwoImages:
    def __init__(self):
        pass

    def __call__(self, img1, img2, homo_matrix):
        # 获取两张图像的高度和宽度
        h1, w1 = img1.shape[0], img1.shape[1]
        h2, w2 = img2.shape[0], img2.shape[1]

        # 定义两张图像的边界框
        rect1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(
            (4, 1, 2))
        rect2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(
            (4, 1, 2))

        # 使用单应性矩阵变换第一张图像的边界框
        trans_rect1 = cv2.perspectiveTransform(rect1, homo_matrix)

        # 计算两张图像拼接后覆盖的范围
        total_rect = np.concatenate((rect2, trans_rect1), axis=0)
        min_x, min_y = np.int32(total_rect.min(axis=0).ravel())
        max_x, max_y = np.int32(total_rect.max(axis=0).ravel())

        # 计算将拼接结果平移到正坐标系的变换矩阵
        shift_to_zero_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

        # 对第一张图像进行透视变换和坐标平移
        trans_img1 = cv2.warpPerspective(img1, shift_to_zero_matrix.dot(homo_matrix),
                                         (max_x - min_x, max_y - min_y))

        # 将第二张图像拼接到变换后的第一张图像上
        trans_img1[-min_y:h2 - min_y, -min_x:w2 - min_x] = img2
        return trans_img1
