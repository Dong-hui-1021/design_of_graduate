import pywt
import cv2
import numpy as np
# 读取灰度图
img=cv2.imread("test1.jpg",0)
#对img进行haar小波变换,变量分别是低频，水平高频，垂直高频，对角线高频
cA,(cH,cV,cD)=pywt.dwt2(img,"haar")
print(cA)
#小波变换之后，低频分量对应的图像
cv2.imwrite('lena.png',np.uint8(cA/np.max(cA)*255))
# 小波变换之后，水平方向上高频分量对应的图像
cv2.imwrite('lena_h.png',np.uint8(cH/np.max(cH)*255))
# 小波变换之后，垂直方向上高频分量对应的图像
cv2.imwrite('lena_v.png',np.uint8(cV/np.max(cV)*255))
# 小波变换之后，对角线方向上高频分量对应的图像
cv2.imwrite('lena_d.png',np.uint8(cD/np.max(cD)*255))
# 根据小波系数重构的图像
rimg=pywt.idwt2((cA,(cH,cV,cD)),"haar")
cv2.imwrite("rimg.png",np.uint8(rimg))
