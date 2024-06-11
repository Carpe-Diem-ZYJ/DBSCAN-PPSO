import csv
import os
import cv2
import numpy as np
from dbslib import get_metrics

project_folder = r"E:\Projects\Git Projects\PSO-DBSCAN-SEGMENTATION"


def fitness_threshold(Thresholds=None, img_path=None):
    """Given a list of Thresholds and an image location , returns the fitness using Otsu's Objective function"""
    Thresholds.append(256)
    Thresholds.insert(0, 0)
    Thresholds.sort()
    image = cv2.imread(img_path, 0)
    img_array = np.array(image)

    hist, bins = np.histogram(img_array, bins=256, range=(0, 256))
    hist = hist.tolist()
    rows, cols = img_array.shape[:2]

    Total_Pixels = rows * cols

    for j in range(len(hist)):  # Probabilities
        hist[j] = hist[j] / Total_Pixels

    cumulative_sum = []
    cumulative_mean = []
    global_mean = 0
    Sigma = 0
    for j in range(len(Thresholds)):
        Thresholds[j] = int(Thresholds[j])

    for j in range(len(Thresholds) - 1):
        cumulative_sum.append(sum(hist[Thresholds[j]:Thresholds[j + 1]]) + 0.0000001)  # 每一个区间概率和
        cumulative = 0
        for k in range(Thresholds[j], Thresholds[j + 1]):  # 每一个区间的平均密度
            cumulative = cumulative + k * hist[k]

        cumulative_mean.append(cumulative)  # Cumulative mean of each Class

        global_mean = global_mean + cumulative  # Global Intensity Mean

    for j in range(len(cumulative_mean)):  # Computing Sigma
        Sigma = Sigma + (cumulative_sum[j] *
                         ((cumulative_mean[j] - global_mean) ** 2))

    del Thresholds[0]
    del Thresholds[-1]

    return Sigma


class WhaleOptimizationAlgorithm:
    def __init__(self,  n_whales=10, dim=2, n_iter=100, a_min=0, a_max=2, img_path=None):
        self.n_whales = n_whales
        self.dim = dim
        self.n_iter = n_iter
        self.a_min = a_min
        self.a_max = a_max
        self.image_path = img_path
        image = cv2.imread(img_path, 0)
        low = np.min(image)
        high = np.max(image)
        self.positions = np.random.uniform(low, high, (self.n_whales, self.dim))
        self.positions.sort()
        self.positions = self.positions.tolist()

        # 初始化最佳适应度值和对应的位置
        self.fitness = []
        for j in range(self.n_whales):
            self.fitness.append(fitness_threshold(self.positions[j], img_path))
        self.best_index = np.argmax(self.fitness)
        self.best_position = self.positions[self.best_index]
        self.best_fitness = self.fitness[self.best_index]
        self.best_position = np.array(self.best_position)

    def optimize(self):
        for j in range(self.n_iter):
            a = self.a_max - (j+1) * ((self.a_max - self.a_min) / self.n_iter)
            for whale in range(self.n_whales):
                r = np.random.random()
                A = 2 * a * r - a
                B = 2 * r
                l = -1 + np.random.random() * 2
                p = np.random.random()
                b = 1

                self.positions = np.array(self.positions)
                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(B * self.best_position - self.positions[whale])
                        self.positions[whale] = np.clip(self.best_position - A * D, 0, 255)
                    else:
                        random_index = int(np.random.randint(0, self.n_whales))
                        D = abs(B * self.positions[random_index] - self.positions[whale])
                        self.positions[whale] = np.clip(self.positions[random_index] - A * D, 0, 255)
                else:
                    D = abs(self.best_position - self.positions[whale])
                    new_position = D * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_position
                    self.positions[whale] = np.clip(new_position, 0, 255)

                # 确保positions为整数
                self.positions = np.round(self.positions).astype(int)

                # 将positions转换为Python列表
                self.positions = self.positions.tolist()

                # 更新fitness值
                self.fitness[whale] = fitness_threshold(self.positions[whale], self.image_path)

            current_best_index = np.argmax(self.fitness)
            if self.fitness[current_best_index] > self.best_fitness:
                self.best_position = self.positions[current_best_index]
                self.best_position = np.array(self.best_position)
                self.best_fitness = self.fitness[current_best_index]

        print(j, self.best_position, self.best_fitness)

        return self.best_position, self.best_fitness


if __name__ == '__main__':
    # 读取datasets文件夹
    datasets_folder = os.path.join(project_folder, 'Datasets')
    dataset_categories = os.listdir(datasets_folder)
    PSNR = []
    SSIM = []
    NRMSE = []
    csv_path = os.path.join(project_folder, 'threshold_result.csv')
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
            woa = WhaleOptimizationAlgorithm(n_whales=10, dim=3, n_iter=10, img_path=image_path)
            best_position, best_fitness = woa.optimize()
            origin = cv2.imread(image_path, 0)
            best_image = origin.copy()
            best_position = best_position.tolist()
            best_position.append(256)
            best_position.insert(0, 0)
            for i in range(len(best_position) - 1):
                mask = (best_image >= best_position[i]) & (best_image < best_position[i + 1])
                value = np.mean(best_image[mask])
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
