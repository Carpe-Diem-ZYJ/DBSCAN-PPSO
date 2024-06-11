import csv
import os
import cv2
import numpy as np
from dbslib import get_metrics

project_folder = r"E:\Projects\Git Projects\PSO-DBSCAN-SEGMENTATION"


class FireflyKMeansAlgorithm:
    def __init__(self, range_pop, m=50, dim=3, X=None, gama=1.0, belta0=1.0, alpha=1, ite=100):
        self.range_pop = range_pop
        self.m = m
        self.gama = gama
        self.belta0 = belta0
        self.alpha = alpha
        self.ite = ite
        self.dim = dim
        self.X = X

    def fitness_function(self, centroids):
        # 使用Otsu's criterion作为适应度函数
        rows, cols = self.X.shape[:2]
        num = rows * cols
        image = self.X
        labels = np.zeros((rows, cols), dtype=int)
        mg = np.mean(image)

        for i in range(rows):
            for j in range(cols):
                min_distance = float('inf')
                for k in range(len(centroids)):
                    distance = np.linalg.norm(image[i, j] - centroids[k])
                    if distance < min_distance:
                        min_distance = distance
                        labels[i, j] = k

        means = {}
        for label in range(len(centroids)):
            num_i = sum(sum(i == label for i in labels))
            mask = (labels == label)
            mean = np.mean(image[mask])
            means[label] = (num_i / num, mean)

        max_variance = 0
        for i in range(len(centroids)):
            p = means[i][0]
            m = means[i][1]
            max_variance = max_variance + p * (m - mg) ** 2

        return max_variance  # 负值因为我们要最大化适应度

    def optimize(self):
        # 第一阶段: 使用萤火虫算法寻找最优初始质心点
        pop = np.random.uniform(low=self.range_pop[0], high=self.range_pop[1], size=(self.m, self.dim))
        fitness = np.array([self.fitness_function(x) for x in pop])
        best_pop, best_fit = pop[np.argmax(fitness)].copy(), np.max(fitness)
        best_fitness = np.zeros(self.ite)
        flag = 0

        for i in range(self.ite):
            for j in range(self.m):
                if fitness[j] == np.max(fitness):
                    pop[j] = pop[j] + self.alpha * (np.random.rand() - 0.5)
                    pop[j][pop[j] < self.range_pop[0]] = self.range_pop[0]
                    pop[j][pop[j] > self.range_pop[1]] = self.range_pop[1]
                    fitness[j] = self.fitness_function(pop[j])
                else:
                    for q in range(self.m):
                        if fitness[q] > fitness[j]:
                            d = np.linalg.norm(pop[j] - pop[q])
                            belta = self.belta0 * np.exp((-self.gama) * (d ** 2))
                            pop[j] = pop[j] + belta * (pop[q] - pop[j]) + self.alpha * (np.random.rand() - 0.5)
                            pop[j][pop[j] < self.range_pop[0]] = self.range_pop[0]
                            pop[j][pop[j] > self.range_pop[1]] = self.range_pop[1]
                            fitness[j] = self.fitness_function(pop[j])
                            if fitness[j] > best_fit:
                                best_fit = fitness[j]
                                best_pop = pop[j]

            best_fitness[i] = best_fit
            print(i, best_pop, best_fit)

        # 第二阶段: 对最优初始质心点使用K-means进行局部搜索和收敛
        centroids = best_pop
        labels = np.zeros((self.X.shape[0], self.X.shape[1]), dtype=int)
        change = True

        while change:
            change = False
            for i in range(self.X.shape[0]):
                for j in range(self.X.shape[1]):
                    min_distance = float('inf')
                    for k in range(len(centroids)):
                        distance = np.linalg.norm(self.X[i, j] - centroids[k])
                        if distance < min_distance:
                            min_distance = distance
                            labels[i, j] = k

            new_centroids = np.zeros(centroids.shape)
            counts = np.zeros(len(centroids))

            for i in range(self.X.shape[0]):
                for j in range(self.X.shape[1]):
                    k = labels[i, j]
                    new_centroids[k] += self.X[i, j]
                    counts[k] += 1

            for k in range(len(centroids)):
                if counts[k] > 0:
                    new_centroids[k] /= counts[k]
                else:
                    new_centroids[k] = centroids[k]

            if not np.array_equal(new_centroids, centroids):
                change = True
                centroids = new_centroids

        return centroids, best_fitness


# Example usage
if __name__ == "__main__":
    # Define the problem to be solved
    # Define parameters
    range_pop = [0, 255]  # Value range
    m = 10  # Population size
    dim = 3
    gama = 0.5  # Absorption coefficient of the propagation medium to light
    belta0 = 0.2  # Initial attraction value
    alpha = 1  # Step length perturbation factor
    ite = 10  # Iteration times


    def reassign_intensity(image, labels, cen):
        unique_labels = np.unique(labels)
        new_image = np.zeros_like(image)

        for label in unique_labels:
            mask = labels == label
            pixels_in_label = image[mask]
            mean_intensity = np.mean(pixels_in_label)
            new_image[mask] = mean_intensity
            # new_image[mask] = cen[label]

        return new_image


    datasets_folder = os.path.join(project_folder, 'Datasets')
    dataset_categories = os.listdir(datasets_folder)
    PSNR = []
    SSIM = []
    NRMSE = []
    csv_path = os.path.join(project_folder, 'km_fa_result.csv')
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
            # Initialize Firefly Algorithm
            fa = FireflyKMeansAlgorithm(range_pop=range_pop, m=m, dim=dim, X=origin, gama=gama, belta0=belta0,
                                        alpha=alpha,
                                        ite=ite)
            # Optimize using Firefly Algorithm
            best_pop, best_fitness = fa.optimize()
            # 保存聚类图片
            rows, cols = origin.shape[:2]
            labels = np.zeros((rows, cols), dtype=int)
            image = origin.copy()
            best_image = origin.copy()

            for i in range(rows):
                for j in range(cols):
                    min_distance = float('inf')
                    for k in range(len(best_pop)):
                        distance = np.linalg.norm(origin[i, j] - best_pop[k])
                        if distance < min_distance:
                            min_distance = distance
                            labels[i, j] = k
            best_image = reassign_intensity(image, labels, best_pop)

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
