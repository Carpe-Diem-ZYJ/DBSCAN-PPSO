import csv
import math
import random
import numpy
import numpy as np
import cv2
from PIL import Image
import os

from dbslib import get_metrics

project_folder = r"E:\Projects\Git Projects\PSO-DBSCAN-SEGMENTATION"


class Cluster(object):
    # Constructor for cluster object
    def __init__(self):
        self.pixels = []  # intialize pixels into a list
        self.centroid = None  # set the number of centroids to none

    def addPoint(self, pixel):  # add pixels to the pixel list
        self.pixels.append(pixel)


class fcm(object):
    # __inti__ is the constructor and self refers to the current object.
    def __init__(self, k=2, max_iterations=3, min_distance=5.0, size=200, m=2, epsilon=.001):
        self.k = k  # initialize k clusters
        self.max_iterations = max_iterations  # intialize max_iterations
        self.min_distance = min_distance  # intialize min_distance
        self.degree_of_membership = []
        self.s = 0  # size ** 2
        self.size = (size, size)  # intialize the size
        self.m = m
        self.epsilon = epsilon  # .001
        self.max_diff = 0.0
        self.image = 0

    # image_arr = numpy.array(self.image)
    # self.s = image_arr.size // image_arr.shape[2]

    # Takes in an image and performs FCM Clustering.
    def run(self, image):
        self.image = image
        self.image.thumbnail(self.size)  # 改变图片大小
        image_arr = numpy.array(image)  # 将图像转换为数组的形式

        # 计算一个通道的像素点数
        # print(image_arr.shape)
        if len(image_arr.shape) < 3:  # 判断图片数据的通道数
            self.s = image_arr.size
        else:
            self.s = image_arr.size // image_arr.shape[2]  # 取出其中一个通道的像素点，//计算结果是int类型的
        self.pixels = numpy.array(image.getdata(), dtype=numpy.uint8)  # 一个通道的所有像素点排成一排
        # self.beta = self.calculate_beta(self.image)

        print("********************************************************************")
        for i in range(self.s):
            print(self.pixels[i])

        self.clusters = [None for i in range(self.k)]
        self.oldClusters = None

        for i in range(self.s):
            self.degree_of_membership.append(numpy.random.dirichlet(numpy.ones(self.k), size=1))  # 初始化uij
        randomPixels = random.sample(list(self.pixels), self.k)  # 从中随机选取k个像素点
        print("INTIALIZE RANDOM PIXELS AS CENTROIDS")
        print(randomPixels)
        #    print"================================================================================"
        for idx in range(self.k):  # 聚类中心的个数
            self.clusters[idx] = Cluster()
            self.clusters[idx].centroid = randomPixels[idx]  # 初始化ci
        # if(i ==0):
        for cluster in self.clusters:
            for pixel in self.pixels:
                cluster.addPoint(pixel)

        print("________", self.clusters[0].pixels[0])
        iterations = 0

        self.oldClusters = [cluster.centroid for cluster in self.clusters]
        print("HELLO I AM ITERATIONS:", iterations)
        self.calculate_centre_vector()  # 第一次的更新uij和ci
        self.update_degree_of_membership()
        iterations += 1
        self.J_min = self.calculate_J(self.degree_of_membership, self.clusters)

        # shouldExit(iterations) checks to see if the exit requirements have been met.
        # - max iterations has been reached OR the centers have converged.
        while self.shouldExit(iterations) is False:
            self.oldClusters = [cluster.centroid for cluster in self.clusters]
            print("HELLO I AM ITERATIONS:", iterations)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            iterations += 1

        for cluster in self.clusters:  # 经过上述循环，满足要求之后，输出最后的聚类中中心ci
            print(cluster.centroid)
        return [cluster.centroid for cluster in self.clusters]

    def selectSingleSolution(self):
        self.max_iterations = 10

    def shouldExit(self, iterations):

        '''#/////////////////////////根据迭代次数和uij作为循环结束标志///////////////
		if iterations >= self.max_iterations or self.max_diff < self.epsilon:#迭代次数到达上限或者uij变化小于阈值就退出循环
			return True
		print("delta max_diff:",self.max_diff)#输出此次迭代输出的uij差值
		return False
		'''
        self.clusters = self.calculate_centre_vector()
        self.degree_of_membership = self.update_degree_of_membership()

        self.J = self.calculate_J(self.degree_of_membership, self.clusters)

        # if self.J<self.J_min:
        #     self.J_min=self.J

        self.diff = abs(self.J - self.J_min)
        if self.diff < self.epsilon or iterations >= self.max_iterations:
            return True
        print("delta diff:", self.diff)

        return False

    # if (self.max_diff > self.epsilon):
    #   return False
    # Perform normalization
    # self.normalization()
    # for i in self.s:

    # Euclidean distance (Distance Metric).
    def calcDistance(self, a, b):
        result = math.sqrt((a - b) ** 2)
        return result

    # Calculates the centroids using degree of membership and fuzziness.
    def calculate_centre_vector(self):
        t = []
        for i in range(self.s):
            t.append([])
            for j in range(self.k):
                t[i].append(pow(self.degree_of_membership[i][0][j], self.m))
        # print"\n\nCALC_CENTRE_VECTOR INVOKED:"

        for cluster in range(self.k):
            # print"*********************************************************************************"
            numerator = 0.0
            denominator = 0.0
            for i in range(self.s):
                # print "+++++++++", self.clusters[cluster].pixels[i], t[i][cluster], (t[i][cluster] * self.clusters[cluster].pixels[i])
                numerator += t[i][cluster] * self.clusters[cluster].pixels[i]
                denominator += (t[i][cluster])
            # print " ______ ", numerator/denominator
            self.clusters[cluster].centroid = (numerator / denominator)
        return self.clusters

    # Updates the degree of membership for all of the data points.
    def update_degree_of_membership(self, pixels, clusters, s, k):
        self.max_diff = 0.0
        degree_of_membership = []
        for i in range(s):
            degree_of_membership.append(numpy.random.dirichlet(numpy.ones(k), size=1))  # 初始化uij,这边主要是为了开辟对应的空间存储uij

        for j in range(s):
            for idx in range(k):  # 上一层取一个样本，这个样本到所有中心的距离
                new_uij = self.get_new_value(pixels[j], clusters[idx].centroid, clusters)  # 计算新的uij
                if (j == 0):
                    print("This is the Updatedegree centroid number:", idx,
                          clusters[idx].centroid)  # 计算uij时候，先把所导进来的ci输出显示
                # ////////////////这边的终止条件暂时没用用到//////////
                # diff = new_uij - self.degree_of_membership[j][0][idx]
                # if (diff > self.max_diff):
                #     self.max_diff = diff
                # //////////////////////////////////////////////////
                degree_of_membership[j][0][idx] = new_uij
        # return self.max_diff    #原代码
        return degree_of_membership

    def get_new_value(self, i, j, z):  # 计算新的uij
        sum = 0.0
        val = 0.0
        i = float(i)
        j = float(j)
        p = (2 * 1.0 / (self.m - 1))  # cast to float value or else will round to nearst int
        for k in z:
            num = self.calcDistance(i, j)
            denom = self.calcDistance(i, k.centroid)
            val = num / (denom + 1)
            val = pow(val, p)  # val得p次方
            sum += val
        return 1.0 / (sum + 1)

    def normalization(self):
        max = 0.0
        highest_index = 0
        for i in range(self.s):
            # Find the index with highest probability
            for j in range(self.k):
                if (self.degree_of_membership[i][0][j] > max):
                    max = self.degree_of_membership[i][0][j]
                    highest_index = j
            # Normalize, set highest prob to 1 rest to zero
            for j in range(self.k):
                if (j != highest_index):
                    self.degree_of_membership[i][0][j] = 0
                else:
                    self.degree_of_membership[i][0][j] = 1

    def calculate_J(self, degree_of_membership, clusters, pixels, k, s):
        J = 0
        for j in range(s):
            for i in range(k):
                J = numpy.add(
                    pow(degree_of_membership[j][0][i], self.m) * pow(self.calcDistance(pixels[j], clusters[i].centroid),
                                                                     2), J)
        return J

    # Shows the image.
    def showImage(self):
        self.image.show()

    def showClustering(self, image, pixels, clusters):
        # Calculate average intensity for each cluster
        cluster_intensities = [0 for _ in clusters]
        cluster_pixel_counts = [0] * len(clusters)
        for pixel in pixels:
            shortest = float('Inf')
            nearest_cluster_index = -1
            for i, cluster in enumerate(clusters):
                distance = self.calcDistance(cluster.centroid, pixel)
                if distance < shortest:
                    shortest = distance
                    nearest_cluster_index = i
            cluster_intensities[nearest_cluster_index] += pixel  # Assuming pixel is now a single intensity value
            cluster_pixel_counts[nearest_cluster_index] += 1

        for i, intensity_sum in enumerate(cluster_intensities):
            if cluster_pixel_counts[i] > 0:  # Avoid division by zero
                cluster_intensities[i] = intensity_sum / cluster_pixel_counts[i]

        # Assign average intensity to each pixel
        localPixels = [None] * len(pixels)
        for idx, pixel in enumerate(pixels):
            shortest = float('Inf')
            nearest = None
            for cluster_index, cluster in enumerate(clusters):
                distance = self.calcDistance(cluster.centroid, pixel)
                if distance < shortest:
                    shortest = distance
                    nearest = cluster_index
            localPixels[idx] = cluster_intensities[nearest]

        # Convert localPixels to image format for grayscale
        w, h = image.size
        localPixels = numpy.asarray(localPixels).astype('uint8').reshape(
            (h, w))  # Note the reshape now does not include -1 for the third dimension
        greyMap = Image.fromarray(localPixels, mode='L')  # Specify mode='L' for grayscale

        # Optionally, display or save the image
        # greyMap.show()
        cluster_img = numpy.array(greyMap)

        # Save or return the image
        return cluster_img

    def showScatterPlot(self):
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        sum_of_overlapeed_pixels = 0

        for i in range(self.s):
            # Find the index with highest probability
            status = False
            for j in range(self.k):
                if self.degree_of_membership[i][0][j] >= 0.5:
                    status = True

            if status == False:
                sum_of_overlapeed_pixels = sum_of_overlapeed_pixels + 1

        print("sum of overlapped pixels ", sum_of_overlapeed_pixels)


class PSO_optimization(object):
    random.seed()

    ### Initialization of positions ###
    def initPos(self, image, m, k):  # m是种群的数量，k是聚类中心个数
        # image=np.array(image)
        # self.width = image.shape[1]
        # self.height= image.shape[0]
        # print("image size:",image.shape)

        self.pixels = image.getdata()
        self.Max_Pixel = max(self.pixels)
        self.Min_Pixel = min(self.pixels)
        self.pixels = np.array(self.pixels, dtype=np.uint8)

        # self.particles =np.zeros((m,k),dtype=np.int)
        self.particles = np.zeros((m, k), dtype=float)
        for i in range(m):
            for j in range(k):
                self.particles[i][j] = random.uniform(self.Min_Pixel, self.Max_Pixel + random.randint(0,
                                                                                                      1))  # 在图中随机选择m个不同的数据点，即作为一个种群,randint能取到上下界
                while random.randint(self.Min_Pixel, self.Max_Pixel) + random.uniform(0,
                                                                                      1) in self.particles:  # 防止选择的是同一个种群
                    self.particles[i][j] = random.uniform(self.Min_Pixel, self.Max_Pixel + random.randint(0, 1))
        # list(self.particles[i]).append([x,y])
        # self.particles[i][j]=self.Position2PixelIndex(x,y)
        return self.particles

    ### Initialization of velocity vector ###
    def initVel(self, m, V_max):
        return [[random.uniform(0, V_max) for j in range(k)] for i in range(m)]

    def PSO(self, iterations, m, k, image, W=1, V_max=20, c1=2, c2=2):  # 这边的m，用在FCM时候，m个ci作为一个个体
        self.f = fcm()
        # degree_of_membership=[]

        clusters = [[None for i in range(k)] for i in range(m)]  # m行，k列，存储聚类中心

        positions = self.initPos(image, m, k)  # 初始化种群

        image_arr = np.array(image)  # 将图像转换为数组的形式
        self.s = image_arr.size  # 计算整张图片的像素个数

        degree_of_membership = []
        for i in range(self.s):
            degree_of_membership.append(np.random.dirichlet(np.ones(k), size=1))
        # ///////////计算uij

        for i in range(m):
            for idx in range(k):  # 聚类中心的个数
                clusters[i][idx] = Cluster()
                clusters[i][idx].centroid = positions[i][idx]
        # clusters[i][idx].centroid =pixels[positions[i][idx]]

        # /////////////////////////////PSO迭代计算/////////////////

        for i in range(m):
            degree_of_membership[i] = self.f.update_degree_of_membership(self.pixels, clusters[i], self.s, k)

        # ///////////计算目标函数func的值
        func = []
        for i in range(m):
            func.append(self.f.calculate_J(degree_of_membership[i], clusters[i], self.pixels, k, self.s))

        MinFunc_index = func.index(min(func))  # 返回目标函数中最小的那个种群下标
        global_best = positions[MinFunc_index]
        p_best = positions
        # best = [(f.calculate_J(degree_of_membership[i],clusters[i],pixels,k,s), clusters[i] ) for i in m]
        # ///////////更新positions////

        # best = [(func(image,elem[0],elem[1]),elem[0],elem[1]) for elem in positions]#记录所有种群个体的func计算的值
        # global_best = max(best, key=lambda x:x[0])#这边可能得改为min，计算的其中的最值作为全局最佳值
        vel = self.initVel(m, V_max)
        J_value = []
        for x in range(iterations):
            # p_best=p_best
            print("HELLO I AM ITERATIONS:", x)
            cur_positions = positions  # 先把当前位置保存下来
            positions = np.zeros((m, k), dtype=float)  # []
            for i in range(m):  # 迭代iterations次，选择最佳的个体作为种群的最佳状态，下面是对v和x的更新

                # 	k_pixels=[]
                vel[i] = [
                    np.multiply(W, vel[i]) +  # vi=wvi+c1*rand()*(pbesti-xi)+c2*rand()*(gbesti-xi)
                    np.multiply(np.multiply(c1, [random.uniform(0, 1) for j in range(k)]),
                                (np.subtract(p_best[i], cur_positions[i]))) +
                    np.multiply(np.multiply(c2, [random.uniform(0, 1) for j in range(k)]),
                                (np.subtract(global_best, cur_positions[i])))]
                # for i in range(m)]
                positions[i] = np.add(cur_positions[i], vel[i])  # xi=xi+vi
                # for j in range(k):
                # 	k_pixels.append(self.pixels[self.Position2PixelIndex(self.positions[i][j][0], self.positions[i][j][1])])
                # //////////////////判断位置，保证不越界
                for j in range(k):
                    if positions[i][j] > self.Max_Pixel:  # 防止超出最大边缘
                        positions[i][j] = self.Max_Pixel
                    elif positions[i][j] < self.Min_Pixel:
                        positions[i][j] = self.Min_Pixel

            print("---------------------Next is the p_best value--------------------")
            p_best = self.Positions2p_best(cur_positions, positions, m, k)  # PSO更新完之后得到的positions得根据func在判断是否是p_best
            print("p_best:", p_best)
            print("---------------------Next is the global_best value--------------------")
            global_best = self.P_best2global_best(p_best, m, k)  # 上述m个种群运算完之后，根据p_best计算global_best
            print("global_best", global_best)

            J_value.append(self.Single_func(global_best, k))
            if self.Single_func(global_best, k) < 1:
                break

            # ////////////////////////////判断并重新更新p_best

            # best = [(func(image,elem[0], elem[1]), elem[0], elem[1]) for elem in positions]#重新计算种群中各个个体的数值
            # global_best = max(best+[global_best], key=lambda x: x[0])#重新选择最佳的个体
            vel = self.initVel(m, V_max)
        return global_best, J_value

    def Positions2p_best(self, cur_positions, positions, m, k):
        p_best = np.zeros((m, k), dtype=float)
        cur_func = self.Func_calculate(cur_positions, m, k)  # 计算相应的目标函数值
        update_func = self.Func_calculate(positions, m, k)
        d = np.array(cur_func) - np.array(update_func)

        for i in range(m):
            if d[i] <= 0:
                p_best[i] = cur_positions[i]
            else:
                p_best[i] = positions[i]

        return p_best

    def P_best2global_best(self, p_best, m, k):
        p_bestFunc = self.Func_calculate(p_best, m, k)
        MinFunc_index = p_bestFunc.index(min(p_bestFunc))
        global_best = p_best[MinFunc_index]

        return global_best

    def Func_calculate(self, positions, m, k):
        degree_of_membership = []
        clusters = [[None for i in range(k)] for i in range(m)]

        for i in range(self.s):
            degree_of_membership.append(np.random.dirichlet(np.ones(k), size=1))

        for i in range(m):
            for idx in range(k):  # 聚类中心的个数
                clusters[i][idx] = Cluster()
                # print("Now index x,y:",positions[i][idx][0],positions[i][idx][1])
                clusters[i][idx].centroid = positions[i][idx]  # 根据坐标计算对应的像素值

        for i in range(m):
            degree_of_membership[i] = self.f.update_degree_of_membership(self.pixels, clusters[i], self.s, k)
        # ///////////计算目标函数func的值
        func = []
        for i in range(m):
            func.append(self.f.calculate_J(degree_of_membership[i], clusters[i], self.pixels, k, self.s))
        return func

    def Position2PixelIndex(self, x, y):
        Pixel_index = self.width * y + x
        return Pixel_index

    def Show_image(self, image, pixels, global_best):
        cluster_img = self.f.showClustering(image, pixels, global_best)
        return cluster_img

    def Single_func(self, global_best, k):  # 通过一个聚类中心，然后计算其对应的func值
        clusters = [None for i in range(k)]
        for idx in range(k):  # 聚类中心的个数
            clusters[idx] = Cluster()
            clusters[idx].centroid = global_best[idx]
        degree_of_membership = self.f.update_degree_of_membership(self.pixels, clusters, self.s, k)
        J_value = self.f.calculate_J(degree_of_membership, clusters, self.pixels, k, self.s)
        return J_value


if __name__ == "__main__":
    iterations = 1
m = 5
k = 3
# m是种群的数量，k是聚类中心个数
# 读取datasets文件夹
datasets_folder = os.path.join(project_folder, 'Datasets')
dataset_categories = os.listdir(datasets_folder)
PSNR = []
SSIM = []
NRMSE = []
csv_path = os.path.join(project_folder, 'pso-fcm_result.csv')
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
        origin = Image.open(image_path)
        p = PSO_optimization()
        # 保存聚类图片
        pixels = np.array(origin.getdata(), dtype=np.uint8)
        global_best, J_value = p.PSO(iterations, m, k, origin)  # 返回的是最佳聚类中心的坐标位置，和最佳位置对应的func值
        clusters = [None for i in range(k)]
        for i in range(k):
            clusters[i] = Cluster()
            clusters[i].centroid = global_best[i]
        best_image = p.Show_image(origin, pixels, clusters)
        origin = cv2.imread(image_path, 0)
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

# //////////////////这边读写一遍是为了修改图片的位数，因为8为不能给 Image.open打开////////////
# image1 = cv2.imread("he_6.png")
# cv2.imwrite("he_6b.png", image1)
# image2 = Image.open("he_6b.png")
#
