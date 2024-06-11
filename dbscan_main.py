import math
import os
import random
import cv2
import numpy as np
from sklearn.neighbors import sort_graph_by_row_values
from dbslib import read_emphasize_data, precompute_distances, initial_parameters, fitness, generate_image, get_metrics
from sklearn.cluster import DBSCAN
from multiprocessing import Pool
import csv
import time

project_folder = r"E:\Projects\Git Projects\PSO-DBSCAN-SEGMENTATION"


def pso_dbscan(img_path, filename, i_folder, b_folder):
    image = read_emphasize_data(img_path)
    cv2.imwrite('emphasized.jpeg', image)
    nn_set = precompute_distances(image)
    # PSO算法搜索最优解
    # max_iter: 最大迭代次数
    # swarm_size: 粒子群大小
    # max_vel: 粒子最大速度
    # c1, c2: 加速因子
    # 设置参数范围
    Eps, Minpts, csr_matrix = initial_parameters(nn_set, image, 15)
    eps_min = 0.5 * Eps
    eps_max = 1.2 * Eps
    minpts_min = Minpts
    if minpts_min > 6:
        minpts_min = 6
    if minpts_min < 4:
        minpts_min = 4
    minpts_max = 2 * Minpts
    if minpts_max < 12:
        minpts_max = 12
    # PSO算法参数设置
    max_iter = 100
    swarm_size = 10
    max_vel = 0.15
    c1 = 1.0
    c2 = 1.0

    sorted_csr = sort_graph_by_row_values(csr_matrix, warn_when_not_sorted=False)

    # 初始化粒子群位置和速度
    swarm_pos = np.random.uniform(low=0.0, high=1.0, size=(swarm_size, 2))
    swarm_vel = np.zeros((swarm_size, 2))

    # 初始化粒子最优位置和适应度
    swarm_best_pos = np.zeros((swarm_size, 2))
    swarm_best_fitness = np.full((swarm_size,), float('-inf'))  # 这里初始化swarm_best_fitness为负无穷

    # 初始化全局最优解
    global_best_pos = np.zeros(2)
    global_best_fitness = float('-inf')

    # 定义停止迭代的阈值
    stop_threshold = 1e-6
    prev_global_best_fitness = float('-inf')
    flag = 0

    for iteration in range(max_iter):
        for i in range(swarm_size):
            # 更新粒子速度
            swarm_vel[i] = swarm_vel[i] + c1 * random.random() * (swarm_best_pos[i] - swarm_pos[i]) + \
                           c2 * random.random() * (global_best_pos - swarm_pos[i])

            # 控制粒子速度不超过max_vel
            swarm_vel[i] = np.clip(swarm_vel[i], -max_vel, max_vel)

            # 更新粒子位置
            swarm_pos[i] = swarm_pos[i] + swarm_vel[i]

            # 控制粒子位置在合理范围内
            swarm_pos[i] = np.clip(swarm_pos[i], 0.0, 1.0)

            # 计算适应度值
            eps = swarm_pos[i][0] * (eps_max - eps_min) + eps_min
            minpts = int(swarm_pos[i][1] * (minpts_max - minpts_min) + minpts_min)
            dbscan_iter = DBSCAN(eps=eps, min_samples=minpts, metric="precomputed")
            labels_iter = dbscan_iter.fit_predict(sorted_csr.copy())
            masks_iter = labels_iter.reshape(image.shape[0], image.shape[1])
            var = fitness(image, masks_iter)

            # 更新粒子个体最优解
            if var > swarm_best_fitness[i]:
                swarm_best_fitness[i] = var
                swarm_best_pos[i] = swarm_pos[i]

            # 更新全局最优解
            if var > global_best_fitness:
                global_best_fitness = var
                global_best_pos = swarm_pos[i].copy()

            print(iteration, i, eps, minpts, var, labels_iter.max())

        # 检查是否应该停止迭代
        if abs(global_best_fitness - prev_global_best_fitness) < stop_threshold:
            flag += 1
            if flag >= 3:  # 如果连续三次迭代变化小于阈值，停止迭代
                break
        else:
            flag = 0  # 如果本次迭代变化大于阈值，重置flag

        # 更新前一次全局最优适应度值
        prev_global_best_fitness = global_best_fitness

    # 返回全局最优解
    best_eps = global_best_pos[0] * (eps_max - eps_min) + eps_min
    best_minpts = int(global_best_pos[1] * (minpts_max - minpts_min) + minpts_min)
    print(best_eps, best_minpts)

    name, extension = os.path.splitext(filename)
    filename_best = f"{name}_best{extension}"
    filename_initial = f"{name}_initial{extension}"
    generate_image(best_eps, best_minpts, image, sorted_csr, filename_best, b_folder)
    generate_image(Eps, Minpts, image, sorted_csr, filename_initial, i_folder)
    filepath = os.path.join(b_folder, filename_best)
    best = cv2.imread(filepath, 0)
    return best


def evaluate_particle(particle_pos, eps_min, eps_max, minpts_min, minpts_max, sorted_csr, image):
    eps = particle_pos[0] * (eps_max - eps_min) + eps_min
    minpts = int(particle_pos[1] * (minpts_max - minpts_min) + minpts_min)
    dbscan_iter = DBSCAN(eps=eps, min_samples=minpts, metric="precomputed")
    labels_iter = dbscan_iter.fit_predict(sorted_csr.copy())
    masks_iter = labels_iter.reshape(image.shape[0], image.shape[1])
    var = fitness(image, masks_iter)
    print(eps, minpts, var, labels_iter.max())
    return var, particle_pos


def pso_dbscan_parallel(img_path, filename, i_folder, b_folder, max_iter=100, swarm_size=12, max_vel=0.1, c1=1.0,
                        c2=1.0):
    image = read_emphasize_data(img_path)
    cv2.imwrite('emphasized.jpeg', image)
    nn_set = precompute_distances(image)
    Eps, Minpts, csr_matrix = initial_parameters(nn_set, image, 2)
    eps_min = 0.4 * Eps
    eps_max = 1.1 * Eps
    minpts_min = 2
    minpts_max = Minpts
    if minpts_max < 12:
        minpts_max = 12
    sorted_csr = sort_graph_by_row_values(csr_matrix, warn_when_not_sorted=False)

    swarm_pos = np.random.uniform(low=0.0, high=1.0, size=(swarm_size, 2))
    swarm_vel = np.zeros((swarm_size, 2))
    swarm_best_pos = np.zeros((swarm_size, 2))
    swarm_best_fitness = np.full(swarm_size, float('-inf'))
    global_best_pos = np.zeros(2)
    global_best_fitness = float('-inf')

    stop_threshold = 1e-6
    prev_global_best_fitness = float('-inf')
    flag = 0

    with Pool() as pool:
        for iteration in range(max_iter):
            print(iteration)
            fitness_results = [pool.apply_async(evaluate_particle, (
                swarm_pos[i], eps_min, eps_max, minpts_min, minpts_max, sorted_csr, image)) for i in range(swarm_size)]
            results = [p.get() for p in fitness_results]

            for i, (var, pos) in enumerate(results):
                if var > swarm_best_fitness[i]:
                    swarm_best_fitness[i] = var
                    swarm_best_pos[i] = pos
                if var > global_best_fitness:
                    global_best_fitness = var
                    global_best_pos = pos.copy()

            for i in range(swarm_size):
                swarm_vel[i] = swarm_vel[i] + c1 * random.random() * (
                        swarm_best_pos[i] - swarm_pos[i]) + c2 * random.random() * (global_best_pos - swarm_pos[i])
                swarm_vel[i] = np.clip(swarm_vel[i], -max_vel, max_vel)
                swarm_pos[i] = swarm_pos[i] + swarm_vel[i]
                swarm_pos[i] = np.clip(swarm_pos[i], 0.0, 1.0)

            if abs(global_best_fitness - prev_global_best_fitness) < stop_threshold:
                flag += 1
                if flag >= 3:
                    break
            else:
                flag = 0
            prev_global_best_fitness = global_best_fitness

    best_eps = global_best_pos[0] * (eps_max - eps_min) + eps_min
    best_minpts = int(global_best_pos[1] * (minpts_max - minpts_min) + minpts_min)
    print(best_eps, best_minpts)
    name, extension = os.path.splitext(filename)
    filename_best = f"{name}_best{extension}"
    filename_initial = f"{name}_initial{extension}"
    clusters_num = generate_image(best_eps, best_minpts, image, sorted_csr, filename_best, b_folder)
    generate_image(Eps, Minpts, image, sorted_csr, filename_initial, i_folder)
    filepath = os.path.join(b_folder, filename_best)
    best = cv2.imread(filepath, 0)
    return best, clusters_num


def gwo_dbscan(img_path, filename, i_folder, b_folder):
    image = read_emphasize_data(img_path)
    cv2.imwrite('emphasized.jpeg', image)
    nn_set = precompute_distances(image)
    Alpha_pos = np.zeros(2)
    Alpha_score = -float("inf")
    Beta_pos = np.zeros(2)
    Beta_score = -float("inf")
    Delta_pos = np.zeros(2)
    Delta_score = -float("inf")
    Eps, Minpts, csr_matrix = initial_parameters(nn_set, image, 1)
    eps_min = 0.3 * Eps
    eps_max = 1.1 * Eps
    minpts_min = math.ceil(0.5 * Minpts)
    minpts_max = math.floor(1.1 * Minpts)
    # lb = [eps_min, minpts_min]  # 参数下界
    # ub = [eps_max, minpts_max]  # 参数上界
    Max_iter = 20
    SearchAgents_no = 10
    flag = 0
    stop_threshold = 1e-6
    best_tmp = -float("inf")
    pre_best_tmp = -float("inf")

    Positions = np.random.uniform(low=0.0, high=1.0, size=(SearchAgents_no, 2))
    sorted_csr = sort_graph_by_row_values(csr_matrix, warn_when_not_sorted=False)

    for i in range(0, SearchAgents_no):  # 初始化三头狼
        # 计算适应度值
        eps = Positions[i][0] * (eps_max - eps_min) + eps_min
        minpts = int(Positions[i][1] * (minpts_max - minpts_min) + minpts_min)
        dbscan_iter = DBSCAN(eps=eps, min_samples=minpts, metric="precomputed")
        labels_iter = dbscan_iter.fit_predict(sorted_csr.copy())
        masks_iter = labels_iter.reshape(image.shape[0], image.shape[1])
        var = fitness(image, masks_iter)

        if var > Alpha_score:
            Alpha_score = var
            Alpha_pos = Positions[i, :]

        if Alpha_score > var > Beta_score:
            Beta_score = var
            Beta_pos = Positions[i, :]

        if Alpha_score > var > Delta_score and var < Beta_score:
            Delta_score = var
            Delta_pos = Positions[i, :]

    for l in range(0, Max_iter):
        if (best_tmp - pre_best_tmp) > stop_threshold:
            pre_best_tmp = best_tmp
            flag = 0
        else:
            flag += 1
        if flag == 3:
            break
        for i in range(0, SearchAgents_no):
            # 计算适应度值
            eps = Positions[i][0] * (eps_max - eps_min) + eps_min
            minpts = int(Positions[i][1] * (minpts_max - minpts_min) + minpts_min)
            dbscan_iter = DBSCAN(eps=eps, min_samples=minpts, metric="precomputed")
            labels_iter = dbscan_iter.fit_predict(sorted_csr.copy())
            masks_iter = labels_iter.reshape(image.shape[0], image.shape[1])
            var = fitness(image, masks_iter)
            if var > best_tmp:
                best_tmp = var
            print(l, i, eps, minpts, var, labels_iter.max())

            if var > Alpha_score:
                Alpha_score = var
                Alpha_pos = Positions[i, :]

            if Alpha_score > var > Beta_score:
                Beta_score = var
                Beta_pos = Positions[i, :]

            if Alpha_score > var > Delta_score and var < Beta_score:
                Delta_score = var
                Delta_pos = Positions[i, :]

            a = 2 - l * (2 / Max_iter)

            for i in range(0, SearchAgents_no):
                for j in range(0, 2):
                    r1 = random.random()
                    r2 = random.random()

                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                    X1 = Alpha_pos[j] - A1 * D_alpha

                    r1 = random.random()
                    r2 = random.random()

                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                    X2 = Beta_pos[j] - A2 * D_beta

                    r1 = random.random()
                    r2 = random.random()

                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                    X3 = Delta_pos[j] - A3 * D_delta

                    Positions[i, j] = np.clip((X1 + X2 + X3) / 3, 0.0, 1.0)

    best_eps = Alpha_pos[0] * (eps_max - eps_min) + eps_min
    best_minpts = int(Alpha_pos[1] * (minpts_max - minpts_min) + minpts_min)
    print(best_eps, best_minpts)

    name, extension = os.path.splitext(filename)
    filename_best = f"{name}_best{extension}"
    filename_initial = f"{name}_initial{extension}"
    generate_image(best_eps, best_minpts, image, sorted_csr, filename_best, b_folder)
    generate_image(Eps, Minpts, image, sorted_csr, filename_initial, i_folder)
    filepath = os.path.join(b_folder, filename_best)
    best = cv2.imread(filepath, 0)
    return best


if __name__ == '__main__':
    start_time = time.time()
    # 读取datasets文件夹
    datasets_folder = os.path.join(project_folder, 'Datasets')
    dataset_categories = os.listdir(datasets_folder)
    PSNR = []
    SSIM = []
    NRMSE = []
    ClusterNum = []
    csv_path = os.path.join(project_folder, 'result_b=7.csv')
    csv_file = open(csv_path, 'w', newline='', encoding='gbk')
    # 调用open()函数打开csv文件，传入参数：文件名“demo.csv”、写入模式“w”、newline=''、encoding='gbk'
    writer = csv.writer(csv_file)
    # 用csv.writer()函数创建一个writer对象
    writer.writerow(['Category', 'NRMSE', 'PSNR', 'SSIM', 'Clusters_Num'])

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
            print(category + ": " + img)
            # best_image = gwo_dbscan(image_path, img, initial_folder, best_folder)
            best_image, clusters_num = pso_dbscan_parallel(image_path, img, initial_folder, best_folder)
            # best_image = pso_dbscan_parallel(image_path, img, initial_folder, best_folder)
            origin = cv2.imread('emphasized.jpeg', 0)
            psnr, ssim, nrmse = get_metrics(origin, best_image)
            writer.writerow([str(img), str(nrmse), str(psnr), str(ssim), str(clusters_num)])
            PSNR.append(psnr)
            SSIM.append(ssim)
            NRMSE.append(nrmse)
            ClusterNum.append(clusters_num)

        PSNR_mean = np.mean(PSNR)
        SSIM_mean = np.mean(SSIM)
        NRMSE_mean = np.mean(NRMSE)
        ClusterNum_mean = np.mean(ClusterNum)
        PSNR.clear()
        SSIM.clear()
        NRMSE.clear()
        writer.writerow([str(category), str(NRMSE_mean), str(PSNR_mean), str(SSIM_mean), str(ClusterNum_mean)])
        writer.writerow([" ", " ", " ", " "])
    # 记录结束时间
    end_time = time.time()
    # 计算程序运行时间
    execution_time = end_time - start_time
    print("程序运行时间：", execution_time, "秒")
    print("DONE")
