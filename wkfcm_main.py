import copy
import math
import random
import csv
import os
import cv2
import numpy as np
from dbslib import get_metrics

project_folder = r"E:\Projects\Git Projects\PSO-DBSCAN-SEGMENTATION"

MAX = 10000.0
# 用于结束条件
Epsilon = 0.00000001
sigma = 150  # 高斯核函数的参数


# def initialise_U(data, cluster_number):
#     # 这个函数是隶属度矩阵U的每行加起来都为1. 此处需要一个全局变量MAX.
#     global MAX
#     U = []
#     for i in range(0, len(data)):
#         current = []
#         rand_sum = 0.0
#         for j in range(0, cluster_number):
#             dummy = random.randint(1, int(MAX))
#             # random.randint(a,b)：用于生成一个指定范围内的整数。其中参数a是下限，参数b是上限，生成的随机数n：a<=n<=b
#             current.append(dummy)
#             rand_sum += dummy
#         for j in range(0, cluster_number):
#             current[j] = current[j] / rand_sum
#         U.append(current)
#     return U
#
#
# def distance(point, center):
#     # 该函数计算2点之间的距离（作为列表）。欧几里德距离，闵可夫斯基距离
#     p_p = 0.0
#     p_c = 0.0
#     c_c = 0.0
#     p_c += abs(point - center) ** 2
#     p_p += abs(point - point) ** 2
#     c_c += abs(center - center) ** 2
#     dummy = (2 - 2 * math.exp((-p_c) / (2 * sigma * sigma)))**0.5
#     return dummy
#
#
#
# def distance(point, center):
#     # 该函数计算2点之间的距离（作为列表）。我们指欧几里德距离，闵可夫斯基距离
#     dummy = abs(point - center) ** 2
#     dummy = math.exp((-dummy) / (2 * sigma * sigma))
#     return dummy
#
#
# def end_conditon(U, U_old):
#     # 结束条件。当U矩阵随着连续迭代停止变化时，触发结束
#     global Epsilon
#     for i in range(0, len(U)):
#         for j in range(0, len(U[0])):
#             if abs(U[i][j] - U_old[i][j]) > Epsilon:
#                 return False
#     return True
#
#
# def normalise_U(U):
#     # 在聚类结束时使U模糊化。每个样本的隶属度最大的为1，其余为0
#     for i in range(0, len(U)):
#         maximum = max(U[i])
#         for j in range(0, len(U[0])):
#             if U[i][j] != maximum:
#                 U[i][j] = 0
#             else:
#                 U[i][j] = 1
#     return U
#
#
# def kfuzzy(data, cluster_number, m):
#     # 这是主函数，它将计算所需的聚类中心，并返回最终的归一化隶属矩阵U.
#     # 参数是：簇数(cluster_number)和模糊指数(m)
#     # 迭代次数
#     ilteration_num = 0
#     # 初始化隶属度矩阵U
#     U = initialise_U(data, cluster_number)
#     # print_matrix(U)
#     # 使用fcm初始化聚类中心
#     C = []
#     for j in range(0, cluster_number):
#         current_cluster_center = 0.0
#         dummy_sum_num = 0.0
#         dummy_sum_dum = 0.0
#         for i in range(0, len(data)):
#             # 分母
#             dummy_sum_dum += (U[i][j] ** m)
#         for k in range(0, len(data)):
#             # 分子
#             dummy_sum_num += (U[k][j] ** m) * data[k]
#
#         current_cluster_center += (dummy_sum_num / dummy_sum_dum)
#         # 第j类的所有聚类中心
#         C.append(current_cluster_center)
#     # 循环更新U
#     while True:
#         # 迭代次数
#         ilteration_num += 1
#         # 创建它的副本，以检查结束条件
#         U_old = copy.deepcopy(U)
#         # 距离函数
#         distance_matrix = []
#         for i in range(0, len(data)):
#             current = []
#             for j in range(0, cluster_number):
#                 current.append(distance(data[i], C[j]))
#             distance_matrix.append(current)
#
#         # 更新U
#         for j in range(0, cluster_number):
#             for i in range(0, len(data)):
#                 dummy = 0.0
#                 for k in range(0, cluster_number):
#                     # 分母
#                     dummy += (1 / distance_matrix[i][k]) ** (-1 / (m - 1))
#                 U[i][j] = ((1 / distance(data[i], C[j]) ** 2) ** (-1 / (m - 1))) / dummy
#         # 计算聚类中心
#         C = []
#         for j in range(0, cluster_number):
#             dummy_sum_dum = 0.0
#             dummy_sum_num = 0.0
#             for i in range(0, len(data)):
#                 # 分母
#                 dummy_sum_dum += ((U[i][j]) ** m) * distance_matrix[i][j]
#             for k in range(0, len(data)):
#                 # 分子
#                 dummy_sum_num += (1 / distance_matrix[k][j] ** 2) ** (-1 / (m - 1))
#             current_cluster_center = (dummy_sum_num / dummy_sum_dum)
#             # 第j簇的所有聚类中心
#             C.append(current_cluster_center)
#
#         if end_conditon(U, U_old):
#             # print ("结束聚类")
#             break
#     print("迭代次数：" + str(ilteration_num))
#     # print ("标准化 U")
#     U = normalise_U(U)
#     return U


def initialise_U(data, cluster_number):
    # 这个函数是隶属度矩阵U的每行加起来都为1. 此处需要一个全局变量MAX.
    global MAX
    U = []
    for i in range(0, len(data)):
        current = []
        rand_sum = 0.0
        for j in range(0, cluster_number):
            dummy = random.randint(1, int(MAX))
            # random.randint(a,b)：用于生成一个指定范围内的整数。其中参数a是下限，参数b是上限，生成的随机数n：a<=n<=b
            current.append(dummy)
            rand_sum += dummy
        for j in range(0, cluster_number):
            current[j] = current[j] / rand_sum
        U.append(current)
    return U


def distance(point, center):
    # 该函数计算2点之间的距离（作为列表）。我们指欧几里德距离，闵可夫斯基距离
    if len(point) != len(center):
        return -1
    dummy = 0.0
    for i in range(0, len(point)):
        dummy += abs(point[i] - center[i]) ** 2
    dummy = math.exp((-dummy) / (2 * sigma * sigma))
    return dummy


def end_conditon(U, U_old):
    # 结束条件。当U矩阵随着连续迭代停止变化时，触发结束
    global Epsilon
    for i in range(0, len(U)):
        for j in range(0, len(U[0])):
            if abs(U[i][j] - U_old[i][j]) > Epsilon:
                return False
    return True


def normalise_U(U):
    # 在聚类结束时使U模糊化。每个样本的隶属度最大的为1，其余为0
    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] != maximum:
                U[i][j] = 0
            else:
                U[i][j] = 1
    return U


def calculate_dynamic_weights(U, m):
    """
    计算簇的动态权重。

    参数:
    U -- 隶属度矩阵，shape为(number_of_samples, number_of_clusters)
    m -- 隶属度指数，用于控制隶属度的模糊程度

    返回:
    weights -- 各个簇的动态权重，shape为(number_of_clusters,)
    """
    U = np.array(U)
    # 计算每个簇的隶属度总和
    membership_sums = np.sum(U ** m, axis=0)

    # 计算动态权重
    weights = membership_sums / np.sum(membership_sums)

    return weights


# m的最佳取值范围为[1.5，2.5]
def kfuzzy(data, cluster_number, m):
    # 这是主函数，它将计算所需的聚类中心，并返回最终的归一化隶属矩阵U.
    # 参数是：簇数(cluster_number)和隶属度的因子(m)
    # 迭代次数
    ilteration_num = 0
    # 初始化隶属度矩阵U
    U = initialise_U(data, cluster_number)
    # print_matrix(U)
    # 使用fcm初始化聚类中心
    C = []
    for j in range(0, cluster_number):
        current_cluster_center = []
        for i in range(0, len(data[0])):
            dummy_sum_num = 0.0
            dummy_sum_dum = 0.0
            for k in range(0, len(data)):
                # 分子
                dummy_sum_num += (U[k][j]) * data[k][i]
                # 分母
                dummy_sum_dum += (U[k][j] ** m)
            # 第i列的聚类中心
            current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
        # 第j簇的所有聚类中心
        C.append(current_cluster_center)
    # 循环更新U
    while True:
        # 迭代次数
        ilteration_num += 1
        # 创建它的副本，以检查结束条件
        U_old = copy.deepcopy(U)
        # 距离函数
        distance_matrix = []
        for i in range(0, len(data)):
            current = []
            for j in range(0, cluster_number):
                current.append(distance(data[i], C[j]))
            distance_matrix.append(current)

        # 更新U
        for j in range(0, cluster_number):
            for i in range(0, len(data)):
                dummy = 0.0
                for k in range(0, cluster_number):
                    # 分母
                    dummy += (1 - distance_matrix[i][k]) ** (-1 / (m - 1))
                U[i][j] = ((1 - distance_matrix[i][j]) ** (-1 / (m - 1))) / dummy
        # 计算聚类中心
        C = []
        for j in range(0, cluster_number):
            current_cluster_center = []
            for i in range(0, len(data[0])):
                dummy_sum_num = 0.0
                dummy_sum_dum = 0.0
                for k in range(0, len(data)):
                    # 分子
                    dummy_sum_num += (U[k][j]) * data[k][i] * distance_matrix[k][j]
                    # 分母
                    dummy_sum_dum += (U[k][j] * distance_matrix[k][j])
                # 第i列的聚类中心
                current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
            # 第j簇的所有聚类中心
            C.append(current_cluster_center)

        if end_conditon(U, U_old):
            # print ("结束聚类")
            break
    print("迭代次数：" + str(ilteration_num))
    # print ("标准化 U")
    U = normalise_U(U)
    return U


def wkfuzzy(data, cluster_number, m):
    # 这是主函数，它将计算所需的聚类中心，并返回最终的归一化隶属矩阵U.
    # 参数是：簇数(cluster_number)和隶属度的因子(m)
    # 迭代次数
    ilteration_num = 0
    # 初始化隶属度矩阵U
    U = initialise_U(data, cluster_number)
    # print_matrix(U)
    # 使用fcm初始化聚类中心
    C = []
    for j in range(0, cluster_number):
        current_cluster_center = []
        for i in range(0, len(data[0])):
            dummy_sum_num = 0.0
            dummy_sum_dum = 0.0
            for k in range(0, len(data)):
                # 分子
                dummy_sum_num += (U[k][j]) * data[k][i]
                # 分母
                dummy_sum_dum += (U[k][j] ** m)
            # 第i列的聚类中心
            current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
        # 第j簇的所有聚类中心
        C.append(current_cluster_center)
    # 循环更新U
    while True:
        # 迭代次数
        ilteration_num += 1
        # 创建它的副本，以检查结束条件
        U_old = copy.deepcopy(U)
        # 距离函数
        distance_matrix = []
        for i in range(0, len(data)):
            current = []
            for j in range(0, cluster_number):
                current.append(distance(data[i], C[j]))
            distance_matrix.append(current)
        a = calculate_dynamic_weights(U, m)
        # 更新U
        for j in range(0, cluster_number):
            for i in range(0, len(data)):
                dummy = 0.0
                for k in range(0, cluster_number):
                    # 分母
                    dummy += a[k]*((1 / distance_matrix[i][k]) ** (1 / (m - 1)))
                U[i][j] = a[j] * ((1 / distance_matrix[i][j]) ** (1 / m)) / dummy
        # 计算聚类中心
        C = []
        for j in range(0, cluster_number):
            current_cluster_center = []
            for i in range(0, len(data[0])):
                dummy_sum_num = 0.0
                dummy_sum_dum = 0.0
                for k in range(0, len(data)):
                    # 分子
                    dummy_sum_num += (U[k][j]**m) * data[k][i] * distance_matrix[k][j]
                    # 分母
                    dummy_sum_dum += ((U[k][j]**m) * distance_matrix[k][j])
                # 第i列的聚类中心
                current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
            # 第j簇的所有聚类中心
            C.append(current_cluster_center)

        if end_conditon(U, U_old):
            # print ("结束聚类")
            break
    print("迭代次数：" + str(ilteration_num))
    # print ("标准化 U")
    U = normalise_U(U)
    return U


if __name__ == '__main__':
    # 读取datasets文件夹
    datasets_folder = os.path.join(project_folder, 'Datasets')
    dataset_categories = os.listdir(datasets_folder)
    PSNR = []
    SSIM = []
    NRMSE = []
    csv_path = os.path.join(project_folder, 'kfcm_result.csv')
    csv_file = open(csv_path, 'w', newline='', encoding='gbk')
    # 调用open()函数打开csv文件，传入参数：文件名“demo.csv”、写入模式“w”、newline=''、encoding='gbk'
    writer = csv.writer(csv_file)
    # 用csv.writer()函数创建一个writer对象
    writer.writerow(['Category', 'NRMSE', 'PSNR', 'SSIM'])

    for category in dataset_categories:
        category_folder = os.path.join(datasets_folder, category)
        origin_folder = os.path.join(category_folder, 'Origin')
        initial_folder = os.path.join(category_folder, 'Initial')
        best_folder = os.path.join(category_folder, 'Best')

        # 读取origin文件夹
        origin_contents = os.listdir(origin_folder)
        images_name = [file for file in origin_contents if
                       os.path.splitext(file)[1].lower() == '.jpg']

        for img in images_name:
            image_path = os.path.join(origin_folder, img)
            # 重新初始化并优化
            print(category + ": " + img)
            origin = cv2.imread(image_path, 0)
            rows, cols = origin.shape[:2]
            best_image = origin.copy()
            image = origin.reshape(-1, 1)
            clusters_num = 4
            u = kfuzzy(image, clusters_num, 2)
            for i in range(clusters_num):
                tmp = [row[i] for row in u]
                mask = np.array(tmp).reshape(rows, cols)
                mask = mask == 1
                values = best_image[np.where(mask)]
                values = np.insert(values, 0, 0)
                value = np.mean(values)
                # value = best_position[i]
                best_image[mask] = value
            psnr, ssim, nrmse = get_metrics(origin, best_image)
            cv2.imwrite(category + "_" + img, best_image)
            writer.writerow([str(img), str(nrmse), str(psnr), str(ssim)])
            PSNR.append(psnr)
            SSIM.append(ssim)
            NRMSE.append(nrmse)

        PSNR_mean = np.mean(PSNR)
        SSIM_mean = np.mean(SSIM)
        NRMSE_mean = np.mean(NRMSE)
        PSNR.clear()
        SSIM.clear()
        NRMSE.clear()
        writer.writerow([str(category), str(NRMSE_mean), str(PSNR_mean), str(SSIM_mean)])
        writer.writerow([" ", " ", " ", " "])
