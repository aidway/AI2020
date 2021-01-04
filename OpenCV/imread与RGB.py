```
python3

'''
cv2.imread的读取顺序是BGR，而plt.imshow的顺序是RGB，可通过以下三种方式转换：
'''

# 1. 图像转换
img = cv2.imread("./imgs/test2-1.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)


# 2. 更改显示顺序
img = cv2.imread("./imgs/test2-1.png")[:,:,[2,1,0]]
plt.imshow(img)


# 3. 官方手段
img = cv2.imread("./imgs/test2-1.png")[...,::-1]
plt.imshow(img)

# 4. 如果原图是灰度图
plt.imshow(img, cmap=plt.cm.gray)

```