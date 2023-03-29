# -*- coding: utf-8 -*-
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np
from numpy import random as rand
import gc
import time
import multiprocessing

MATCH_RATE_LIST_BEST = []
MATCH_RATE_LIST_MID = []
MATCH_RATE_LIST_WORST = []
generation_num = 30000 #迭代次数
triangles_num = 10000  # 三角形数量
father_num = 20  # 父代数量
mutate_rate = 0.05  # 变异率
TARGET = "C:\\Users\\Zhong\\Desktop\\a.jpg"
result_address = r"C:\Users\Zhong\Desktop\result"   # 设定生成的图片保存位置为桌面的results文件夹
if not os.path.isdir(result_address): os.makedirs(result_address)   #为保证桌面一定有该文件夹，以免报错，此处判断了上述路径是否存在，若不存在就创建该路径


class Color():
    """
    RGBA模式，RGB三色通道，A透明度通道
    """
    def __init__(self):
        #分别随机生成[0,255]之间的一个整数，作为RGB的值
        #随机生成[95, 115]之间的一个整数，作为A的值
        self.r = rand.randint(0, 255)
        self.g = rand.randint(0, 255)
        self.b = rand.randint(0, 255)
        self.a = rand.randint(95, 115)

class Triangle():
    """
    三角形类，定义三角形三顶点，颜色
    """
    def __init__(self):
        self.color = Color()  # 随机初始化颜色值
        self.ax = rand.randint(0, 255)  # 随机初始化三顶点坐标
        self.ay = rand.randint(0, 255)
        self.bx = rand.randint(0, 255)
        self.by = rand.randint(0, 255)
        self.cx = rand.randint(0, 255)
        self.cy = rand.randint(0, 255)

def Draw_Together(triangles):
    """
        该函数负责将输入的三角形列表合并绘制到一张图片中。
        triangles: 一个父代列表，包含了一个父代包含的所有由三角形类定义的三角形。
    """
    img = Image.new('RGBA', size=(256, 256))   # 新建一个画布
    draw_img = ImageDraw.Draw(img)    # 创建一个img图像上绘图的对象
    draw_img.polygon([(0, 0), (0, 255), (255, 255), (255, 0)], fill=(255, 255, 255, 255))    # 绘制多边形，此处绘制了一个覆盖全画布大小的白色全不透明矩形，作为背景
    for triangle in triangles:
        triangle_img = Image.new('RGBA', size=(256, 256))  # 新建一个画单个三角形的画布
        draw_triangle = ImageDraw.Draw(triangle_img)
        draw_triangle.polygon([(triangle.ax, triangle.ay),
                      (triangle.bx, triangle.by),
                      (triangle.cx, triangle.cy)],
                     fill=(triangle.color.r, triangle.color.g, triangle.color.b, triangle.color.a))    # 在前面定义的画布triangle_img上绘制出指定三角形
        img = Image.alpha_composite(img, triangle_img)  # 将两个图片按各自透明度叠加到一张图中
    return img


def Mutate(generation,triangle):
    """
    在给定变异率下，对传入的三角形在一定幅度内随机变异

    """
    mutate_rate = 0.05
    if mutate_rate > rand.random():
        triangle.ax = np.clip(triangle.ax + rand.randint(-20, 20), 0, 255)
        triangle.ay = np.clip(triangle.ay + rand.randint(-20, 20), 0, 255)
        triangle.bx = np.clip(triangle.bx + rand.randint(-20, 20), 0, 255)
        triangle.by = np.clip(triangle.by + rand.randint(-20, 20), 0, 255)
        triangle.cx = np.clip(triangle.cx + rand.randint(-20, 20), 0, 255)
        triangle.cy = np.clip(triangle.cy + rand.randint(-20, 20), 0, 255)
        triangle.color.r = np.clip(triangle.color.r + rand.randint(-10, 10), 0, 255)
        triangle.color.g = np.clip(triangle.color.g + rand.randint(-10, 10), 0, 255)
        triangle.color.b = np.clip(triangle.color.b + rand.randint(-10, 10), 0, 255)
        triangle.color.a = np.clip(triangle.color.a + rand.randint(-10, 10), 90, 255)
    return triangle

def Cal_match_tate(gene_img ,target_img):
    """
        计算传入的两图像的像素差值的平方和作为其生成图像的环境适应度，
        其值大小越小，说明两图像相似度高，即环境适应度越高。
    """
    # 将生成图像和目标图像的RGB值分别合并为一个一维向量，便于计算
    gene_pixel = np.array([])     #生成图像的像素向量
    for p in gene_img.split()[:-1]:  # split获得一个元组，包含RGBA四个Image对象，[:-1]用来选取前三个结果，即对象的RGB值
        gene_pixel = np.hstack((gene_pixel, np.hstack(np.array(p))))
    # np.array(p)将p对象转化为numpy类型的数组， np.hstack()可将矩阵按行合并为一维向量
    target_pixel = np.array([])
    for p in target_img.split()[:-1]:
        target_pixel = np.hstack((target_pixel, np.hstack(np.array(p))))
    return np.sum(np.square(np.subtract(gene_pixel, target_pixel)))   # 计算环境适应度，并返回

def init_parent(parent_triangles,parent_images,index):
    """
        初始化父代，随机生成20个三角形组成的父代
    """
    parent_triangle = parent_triangles[index]
    for i in range(triangles_num):
        parent_triangle.append(Triangle())
    parent_images.append(Draw_Together(parent_triangle))  #每当一个父代生成好后，就绘制其合并图像，并添加到parent_images中
    parent_triangles[index] = parent_triangle
    return parent_triangles,parent_images

def breed_children(child_triangles,child_images,parent,generation,index):
    child_triangle = child_triangles[index]
    for i in range(triangles_num):
        child_triangle.append(Mutate(generation,parent[i]))
    child_triangles[index] = child_triangle
    child_images.append(Draw_Together(child_triangle))
    return child_triangle,child_images

if __name__ == '__main__':        # 运行主函数
    """
    流程：
    1. 随机初始化20个图像（每个图像由triangle_num个三角形绘成），挑选fitness最小的当做父代
    2. 从父代变异10个子代， 选出父代及十个子代中match_rate最小的当做新父代
    3. 重复步骤2，30000次
    4. 每100代输出一次match_rate,并保存图像到本地
    """
    target_img = Image.open(TARGET).resize((256, 256)).convert("RGBA")   #导入目标图像并改变大小为256*256,色彩模式RGBA
    cpu_num = os.cpu_count()    # 获得物理核心数，用于创建进程池
    p = multiprocessing.Pool(cpu_num)   # 创建一个进程池，用于加速计算环境适应度
    # 流程1
    ## 初始化父代
    parent_images = multiprocessing.Manager().list([])    #储存每个父代的合并后的图像，便于环境适应度计算
    parent_triangles = multiprocessing.Manager().list([])   #储存每个父代所包含的所有三角形
    for i in range(father_num):
        print('正在初始化第%d个父代...' %(i+1))
        parent_triangles.append([])
        p.apply_async(init_parent, args=(parent_triangles,parent_images,i ,))  # 使用进程池加速初始化父代
    p.close()
    p.join()
    ## 计算match_rate，并挑选最小的当父代
    match_rates = []
    for i in range(father_num):
        match_rates.append(Cal_match_tate(parent_images[i], target_img))
    parent_tmp = parent_triangles[match_rates.index(min(match_rates))]   # 定义最终父代为match_rate最小的那个初始化父代
    del parent_images        # 删除保存所有初始化父代图像及三角形的列表，因为后面用不到了
    del parent_triangles
    del p
    gc.collect()
    # 流程2,3
    parent = multiprocessing.Manager().list(parent_tmp)   # 将父代转换为可共享的列表
    for generation in range(generation_num):
        ## 从父代变异10个子代
        child_images = multiprocessing.Manager().list([])    #储存每个子代的合并后的图像，便于环境适应度计算
        child_triangles = multiprocessing.Manager().list([])   #储存每个子代所包含的所有三角形
        nums = max(11,cpu_num)
        po = multiprocessing.Pool(nums)   # 创建一个进程池，用于加速计算环境适应度
        start_time = time.time()
        for i in range(10):
            child_triangles.append([])
            po.apply_async(breed_children, args=(child_triangles, child_images, parent, generation,i,))
        po.close()
        po.join()
        end_time = time.time()
        print("generation",generation,"waste time: ", end_time-start_time)
        match_rates = []
        for i in range(10):
            match_rates.append(Cal_match_tate(child_images[i], target_img))
        MATCH_RATE_LIST_BEST.append(max(match_rates))
        MATCH_RATE_LIST_WORST.append(min(match_rates))
        if Cal_match_tate(Draw_Together(parent), target_img) > min(match_rates):
            parent = child_triangles[match_rates.index(min(match_rates))]
        match_rate = max(match_rates)+min(match_rates)
        MATCH_RATE_LIST_MID.append(match_rate/2)
        del child_images
        del child_triangles
        del po
        gc.collect()

        # 流程4
        if generation % 100 == 0:     # 每迭代100代就输出一次环境适应度值，并保存当前父代图像到本地
            match_rate = Cal_match_tate(Draw_Together(parent), target_img)
            plt.plot(range(1, len(MATCH_RATE_LIST_MID) + 1, 1), MATCH_RATE_LIST_MID, 'r', label='mid')
            plt.plot(range(1, len(MATCH_RATE_LIST_BEST)  + 1, 1), MATCH_RATE_LIST_BEST, 'b', label='best')
            plt.plot(range(1, len(MATCH_RATE_LIST_WORST)  + 1, 1), MATCH_RATE_LIST_WORST, 'g', label='worst')
            plt.legend()
            plt.xlabel('generation')
            plt.ylabel('match_rate')
            plt.savefig(os.path.join(result_address, 'match_rate.png'))
            plt.close()
            print('第%d代的match_rate：\t%d' %(generation, match_rate))
            save_img = Draw_Together(parent)
            save_img.save(os.path.join(result_address, '%d.png' % (generation)))
