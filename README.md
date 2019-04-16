# PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-
计算图像对的特征匹配，并估计基础矩阵。算出空间中对应的匹配点，求出投影矩阵。
# 一.外极几何
多视图几何--既是利用在不同视点所拍摄图像间的关系，来研究照相机之间或者特征之间的关系。如果有一个场景的两个视图以及视图中的对应图像点，那么根据照相机间的空间相对位置关系、照相机的性质以及三维场景点的位置，可以得到对这些图像点的一些几何关系约束。  
三维场景点X经过4×4的单应性矩阵H变换为HX后，则HX在照相机PH^-1里得到的图像点和X在照相机P里得到的图像点相同。  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/1.jpg)  
因此，当我们分析双视图几何关系时，总是可以将照相机间的相对位置关系用单应性矩阵加以简化。这里的单应性矩阵通常只做刚体变换，即值通过单应性矩阵变换了坐标系。一个比较好的做法是，将原点和坐标轴与第一个照相机对齐。  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/2.jpg)  
其中K1和K2是标定矩阵，R是第二个照相机的旋转矩阵，t是第二个照相机的平移向量。  
同一个图像点经过不同的投影矩阵产生的不同投影点必须满足如下：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/3.jpg)       (①)  
其中：![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/4.jpg)  
矩阵St为反对称矩阵：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/5.jpg)  
公式①为外极约束条件，矩阵F为基础矩阵，基础矩阵（后面会细讲）可以由两照相机的参数矩阵（相对于旋转R和平移t）表示。  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/7.jpg)  
本质上两幅图之间的对极几何是图像平面与以基线为轴的平面束的交的几何，这种几何被广泛同于双目成像原理中 
如图所示，摄像机由相机中心C,C’以及各自的成像平面表示，对于任意一个空间中的点X,在两个像平面上的点分别为x,x’，第一幅图像上的点x反向投影成空间三维的一条射线，它由摄像机中心和x确定，这条射线向第二个图像平面上投影得到一条直线l’，显然x的投影x’必然在l’上.  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/6.jpg)  
如图所示，我们把相机中心的连线叫做基线(baseline),基线与两幅图像平面交于对极点e和e’，任何一条包含基线的平面pi是一张对极平面，分别与图像平面相交于l和l’，实际上，当三维位置X变化时，对应的实际上是该对极平面绕基线”旋转”，这个旋转得到的平面簇叫做对极平面束，由所有对极平面和成像平面相交得到的对极限相交于对极点.  
# 二.基础矩阵  
基础矩阵F是一个3×3的秩为2的矩阵。一般记基础矩阵F为：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/8.jpg)  
给一对匹配点x1(x1,y1)、x2(x2,y2),由矩阵乘法可知有:  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/9.jpg)  
# 三.八点法
确定基础矩阵的最简单的方法即为8点法。存在一对匹配点x1,x2，当有8对这样的点时如下图所示：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/10.jpg)  
则有如下方程：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/11.jpg)  
另左边矩阵为A，右边矩阵为f，即 Af=0
优化方法一般使用最小二乘法，即优化:  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/12.jpg)  
RANSAN算法可以用来消除错误匹配的的点，找到基础矩阵F，算法思想如下:  
（1）随机选择8个点；   
（2）用这8个点估计基础矩阵F；   
（3）用这个F进行验算，计算用F验算成功的点对数n；   
重复多次，找到使n最大的F作为基础矩阵。  
# 四.三维点照相机矩阵
三角剖分   
给定照相机参数模型，图像点可以通过三角剖分来恢复出这些点的三维位置。   
对于两个照相机P1和P2的视图，三维实物点X的投影为x1和x2   
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/13.jpg)  
我们可以通过SVD算法来得到三维点的最小二乘估计值。  
如果已经找到了一些三维点及其图像投影，我们可以使用直接线性变换的方法来计算照相机矩阵P。本质上，这是三角剖分方法的逆问题，有时我们将其称为照相机反切法。利用该方法恢复照相机矩阵同样也是一个最小二乘问题。  
每个三维点（齐次坐标下）按照λixi=PXi投影到图像点xi=[xi,yi,1]，相应的点满足下面的关系：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/14.jpg)  
其中，p1、p2和p3是矩阵P的三行，上面的式子可以写得更紧凑，如所示 : Mv=0   
然后，我们可以使用SVD分解估计出照相机矩阵。  
# 五.实例
## 1).室外普通场景：
特征：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/15.jpg)  
八点法求得F：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/16.jpg)  
红框为所得基础矩阵：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/17.jpg)  
点和投影矩阵：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/18.jpg)  
阈值为le^-3级别  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/19.jpg)

## 2).室外复杂场景：
特征：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/20.jpg)  
八点法求得F：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/21.jpg)  
红框为所得基础矩阵：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/22.jpg)  
点和投影矩阵：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/23.jpg)  
阈值为le^-1级别  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/24.jpg)

## 3).室内场景：
特征：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/25.jpg)  
八点法求得F：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/26.jpg)  
红框为所得基础矩阵：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/27.jpg)  
点和投影矩阵：  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/28.jpg)  
阈值为le^-1级别  
![image](https://github.com/Nocami/PythonComputerVision-7-EpipolarGeometry--FundamentalMatrix-/blob/master/images/29.jpg)
## 4).简述
通过三组不同场景图像的对比，我们可以做出基本判断：  
在室外普通场景下，匹配到的特征点数量多，可以以较高的精度（阈值级别较小）用八点法计算出基础矩阵以及三维点相机矩阵；  
面对室外较复杂场景时，精度大大下降；在处理室内图片数据集时，甚至出现了不能正确计算的情况，效果较差。  
# 六.源码及说明
~~~python
# coding: utf-8

# In[1]:
from pylab import *
from PIL import Image

# If you have PCV installed, these imports should work
from PCV.geometry import homography, camera, sfm
from PCV.localdescriptors import sift
from PIL import Image
from numpy import *
from pylab import *
import numpy as np


# In[2]:

#import camera
#import homography
#import sfm
#import sift
camera = reload(camera)
homography = reload(homography)
sfm = reload(sfm)
sift = reload(sift)


# In[3]:

# Read features
im1 = array(Image.open('images/1.jpg'))
sift.process_image('images/1.jpg', 'im1.sift')

im2 = array(Image.open('images/2.jpg'))
sift.process_image('images/2.jpg', 'im2.sift')


# In[4]:

l1, d1 = sift.read_features_from_file('im1.sift')
l2, d2 = sift.read_features_from_file('im2.sift')


# In[5]:

matches = sift.match_twosided(d1, d2)


# In[6]:

ndx = matches.nonzero()[0]
x1 = homography.make_homog(l1[ndx, :2].T)
ndx2 = [int(matches[i]) for i in ndx]
x2 = homography.make_homog(l2[ndx2, :2].T)

d1n = d1[ndx]
d2n = d2[ndx2]
x1n = x1.copy()
x2n = x2.copy()


# In[7]:

figure(figsize=(16,16))
sift.plot_matches(im1, im2, l1, l2, matches, True)
show()


# In[26]:

#def F_from_ransac(x1, x2, model, maxiter=5000, match_threshold=1e-6):
def F_from_ransac(x1, x2, model, maxiter=5000, match_threshold=1e-6):
    """ Robust estimation of a fundamental matrix F from point
    correspondences using RANSAC (ransac.py from
    http://www.scipy.org/Cookbook/RANSAC).

    input: x1, x2 (3*n arrays) points in hom. coordinates. """

    from PCV.tools import ransac
    data = np.vstack((x1, x2))
    d = 10 # 20 is the original
    # compute F and return with inlier index
    F, ransac_data = ransac.ransac(data.T, model,
                                   8, maxiter, match_threshold, d, return_all=True)
    return F, ransac_data['inliers']


# In[27]:

# find F through RANSAC
model = sfm.RansacModel()
F, inliers = F_from_ransac(x1n, x2n, model, maxiter=5000, match_threshold=1e-1)
print F


# In[28]:

P1 = array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2 = sfm.compute_P_from_fundamental(F)


# In[29]:

print P2
print F


# In[30]:

# P2, F (1e-4, d=20)
# [[ -1.48067422e+00   1.14802177e+01   5.62878044e+02   4.74418238e+03]
#  [  1.24802182e+01  -9.67640761e+01  -4.74418113e+03   5.62856097e+02]
#  [  2.16588305e-02   3.69220292e-03  -1.04831621e+02   1.00000000e+00]]
# [[ -1.14890281e-07   4.55171451e-06  -2.63063628e-03]
#  [ -1.26569570e-06   6.28095242e-07   2.03963649e-02]
#  [  1.25746499e-03  -2.19476910e-02   1.00000000e+00]]


# In[31]:

# triangulate inliers and remove points not in front of both cameras
X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2)


# In[32]:

# plot the projection of X
cam1 = camera.Camera(P1)
cam2 = camera.Camera(P2)
x1p = cam1.project(X)
x2p = cam2.project(X)


# In[33]:

figure(figsize=(16, 16))
imj = sift.appendimages(im1, im2)
imj = vstack((imj, imj))

imshow(imj)

cols1 = im1.shape[1]
rows1 = im1.shape[0]
for i in range(len(x1p[0])):
    if (0<= x1p[0][i]<cols1) and (0<= x2p[0][i]<cols1) and (0<=x1p[1][i]<rows1) and (0<=x2p[1][i]<rows1):
        plot([x1p[0][i], x2p[0][i]+cols1],[x1p[1][i], x2p[1][i]],'c')
axis('off')
show()


# In[34]:

d1p = d1n[inliers]
d2p = d2n[inliers]


# In[35]:

# Read features
im3 = array(Image.open('images/3.jpg'))
sift.process_image('images/3.jpg', 'im3.sift')
l3, d3 = sift.read_features_from_file('im3.sift')


# In[36]:

matches13 = sift.match_twosided(d1p, d3)


# In[37]:

ndx_13 = matches13.nonzero()[0]
x1_13 = homography.make_homog(x1p[:, ndx_13])
ndx2_13 = [int(matches13[i]) for i in ndx_13]
x3_13 = homography.make_homog(l3[ndx2_13, :2].T)


# In[38]:

figure(figsize=(16, 16))
imj = sift.appendimages(im1, im3)
imj = vstack((imj, imj))

imshow(imj)

cols1 = im1.shape[1]
rows1 = im1.shape[0]
for i in range(len(x1_13[0])):
    if (0<= x1_13[0][i]<cols1) and (0<= x3_13[0][i]<cols1) and (0<=x1_13[1][i]<rows1) and (0<=x3_13[1][i]<rows1):
        plot([x1_13[0][i], x3_13[0][i]+cols1],[x1_13[1][i], x3_13[1][i]],'c')
axis('off')
show()


# In[39]:

P3 = sfm.compute_P(x3_13, X[:, ndx_13])


# In[40]:

print P3


# In[41]:

print P1
print P2
print P3


# In[22]:

# Can't tell the camera position because there's no calibration matrix (K)


~~~
其中，In[27]：中的“match_threshold=1e-1”，即指阈值精度，可根据图片调整相应的精度。  
代码运行于 python 2，所需库文件可以参考本人稍早前发布的"PythonComputerVision"系列博客。  
  
    
### 特别说明：
入境图片： JMU 集美大学  

原理中部分参考： http://blog.sina.com.cn/s/blog_61cc74300102ys0x.html   https://blog.csdn.net/lhanchao/article/details/51834223  
代码来源：  https://github.com/moizumi99/CVBookExercise/tree/master/Chapter-5  
