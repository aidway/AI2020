## 安装
1，如果只需要主要模块
pip install opencv-python

2，如果需要更全的模块
pip install opencv-contrib-python

3，如果资源较少，不需要任何GUI功能
pip install opencv-python-headless

4，如果资源较少，不需要任何GUI功能，包含contrib模块
pip install opencv-contrib-python-headless
```shell
python -m pip install opencv-python==4.1.0.25 -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

python -m pip install opencv-python-headless -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

python -m pip install opencv-contrib-python==4.1.2.30 -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

python -m pip install pyqt -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
 
```

## 问题
1.缺少共享库(libSM.so.6, libXrender.so.1, libXext.so.6)  
```shell
yum install libSM
```


