import cv2
import sys
import utils  # 引入自定义模块 utils，包含关键点检测和图像拼接相关类

# 读取两张待拼接的图像
image1 = cv2.imread("res/1.jpg")
image2 = cv2.imread("res/2.jpg")

# 实例化 FindKeyPointsAndMatching 类，用于关键点检测和匹配
stitch_match = utils.FindKeyPointsAndMatching()

# 检测两张图像的关键点和描述符
kp1, kp2 = stitch_match.get_key_points(img1=image1, img2=image2)

# 根据关键点匹配计算单应性矩阵（Homography Matrix）
homo_matrix = stitch_match.match(kp1, kp2)

# 实例化 PasteTwoImages 类，用于将两张图像拼接成一张
stitch_merge = utils.PasteTwoImages()

# 利用单应性矩阵和拼接方法生成拼接后的图像
merge_image = stitch_merge(image1, image2, homo_matrix)

# 创建一个窗口展示拼接结果
cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.imshow('output', merge_image)

# 按下 ESC 键退出窗口并保存拼接结果
if cv2.waitKey() == 27:
    cv2.destroyAllWindows()
cv2.imwrite('res/output.jpg', merge_image)
print('\n================>Output saved!')
