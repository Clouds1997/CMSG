# 导入必要的库
import numpy as np
# from sklearn.cluster import DBSCAN
import cv2
from PIL import Image

# 计算两个点之间的欧几里得距离
def euclidean_distance(point_a, point_b):
    if(point_a[2] != point_b[2]):
        return 10000
    return np.sqrt(np.sum((point_a[:2] - point_b[:2]) ** 2))

# 获取点p的eps邻域内的点
def get_eps_neighbors(data, cur_index, p, eps):
    neighbors = []
    for i in range(cur_index, data.shape[0]):
        print(i)
        if euclidean_distance(data[i], p) <= eps:
            neighbors.append(i)
    return neighbors

# 聚类
def dbscan(data, eps, min_samples):
    clusters = []
    visited = []
    for i in range(data.shape[0]):
        if i in visited:
            continue
        visited.append(i)
        print(i)
        neighbors = get_eps_neighbors(data, i, data[i], eps)
        if len(neighbors) < min_samples:
            clusters.append(-1)
            continue
        clusters.append(len(clusters))
        for j in neighbors:
            if j in visited:
                continue
            visited.append(j)
            new_neighbors = get_eps_neighbors(data, i, data[j], eps)
            if len(new_neighbors) >= min_samples:
                neighbors.extend(new_neighbors)
    return np.array(clusters)


def get_lable(path):
    img = Image.open(path)
    img_list = np.asarray(img, dtype=np.uint8)
    se_list = np.full((img_list.shape[0],img_list.shape[1], 1), 19)
    # print(img_list.shape)
    rows = img_list.shape[0]
    cols = img_list.shape[1]
    for row in range(rows):
        for col in range(cols):
            # if (item == np.array([107,142,35])).all():
                # print(type(item))
            if((img_list[row][col] == np.array([100,0,142])).all()):
                se_list[row][col] = 0
                continue
            if((img_list[row][col] == np.array([90,100,142])).all()):
                se_list[row][col] = 1
                continue
            if((img_list[row][col] == np.array([0,0,142])).all()):
                se_list[row][col] = 2
                continue
            if((img_list[row][col] == np.array([128,64,128])).all()):
                se_list[row][col] = 3
                continue
            if((img_list[row][col] == np.array([244,35,232])).all()):
                se_list[row][col] = 3
                continue
            if((img_list[row][col] == np.array([70,70,70])).all()):
                se_list[row][col] = 5
                continue
            if((img_list[row][col] == np.array([70,255,70])).all()):
                se_list[row][col] = 6
                continue
            if((img_list[row][col] == np.array([240,142,35])).all()):
                se_list[row][col] = 7
                continue
            if((img_list[row][col] == np.array([107,142,35])).all()):
                se_list[row][col] = 7
                continue
            if((img_list[row][col] == np.array([70,240,70])).all()):
                se_list[row][col] = 10
                continue
            if((img_list[row][col] == np.array([70,70,240])).all()):
                se_list[row][col] = 10
                continue
                # print("233333")
    # print(se_list.shape)
    return se_list

# 设定Eps和MinPts参数
eps = 10
min_samples = 3

# 生成375 * 1240的数值在0-11之间的矩阵
# img = np.random.randint(0, 2, size=(160, 620))
# img = get_lable("/home/liuhy/workspace/DeepI2P-main/dataset/download/raw/00/image_02_se/0000000000.png")
img = cv2.imread("/home/liuhy/workspace/DeepI2P-main/dataset/download/raw/00/image_02_se/0000000000.png")
# print(img) # 376 * 1241 * 3

if([70,255,70] in img):
    print("yes")

# data = img.reshape((-1,3))

# print(data.shape)
# data = np.float32(data)

# #停止条件 (type,max_iter,epsilon)
# criteria = (cv2.TERM_CRITERIA_EPS +
#             cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# #设置标签
# flags = cv2.KMEANS_RANDOM_CENTERS

# #K-Means聚类 聚集成16类
# compactness, labels30, centers30 = cv2.kmeans(data, 30, None, criteria, 10, flags)

# print(compactness)
# print("*" * 80)
# print(labels30.shape)
# print(" * " * 80)
# print(centers30)


# data_new = []

# print(data)

# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         data_new.append([i,j,data[i,j]])

# data_new_np = np.asarray(data_new)

# print(data_new_np)

# clusters = dbscan(data_new_np, eps, min_samples)

# print(data_new_np.shape)

# # 使用DBSCAN算法进行聚类
# dbscan = DBSCAN(eps=eps, min_samples=min_samples)
# clusters = dbscan.fit_predict(data)

# # 输出聚类结果
# print(clusters.shape)
