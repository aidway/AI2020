# pybm3d github地址
https://github.com/ericmjonas/pybm3d

# 安装步骤
## 环境
centos7.6

## 安装libfftw3
1. 下载
http://www.fftw.org/download.html
2. 安装
```shell
# 基本安装(必选)
[root@tdh01 tmp]# cd fftw-3.3.9/
[root@tdh01 fftw-3.3.9]# ./configure
[root@tdh01 fftw-3.3.9]# make
[root@tdh01 fftw-3.3.9]# make install
# 安装lfftw3(可选)
[root@tdh01 fftw-3.3.9]# ./configure --enable-shared  
[root@tdh01 fftw-3.3.9]# make
[root@tdh01 fftw-3.3.9]# make install
[root@tdh01 fftw-3.3.9]# make clean
# 安装lfftw3l(可选)
[root@tdh01 fftw-3.3.9]# ./configure --enable-shared --enable-single --enable-long-double  
[root@tdh01 fftw-3.3.9]# make
[root@tdh01 fftw-3.3.9]# make install
[root@tdh01 fftw-3.3.9]# make clean
[root@tdh01 fftw-3.3.9]# ./configure --enable-shared --enable-long-double  
[root@tdh01 fftw-3.3.9]# make
[root@tdh01 fftw-3.3.9]# make install
[root@tdh01 fftw-3.3.9]# make clean
# 安装lfftw3f(必选)
[root@tdh01 fftw-3.3.9]# ./configure --enable-shared --enable-float  
[root@tdh01 fftw-3.3.9]# make
[root@tdh01 fftw-3.3.9]# make install
[root@tdh01 fftw-3.3.9]# make clean
# 配置环境变量
[root@tdh01 fftw-3.3.9]# export LD_LIBRARY_PATH=/usr/local/lib
[root@tdh01 fftw-3.3.9]# export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
```
## 安装pybm3d
```shell
[root@tdh01 fftw-3.3.9]# pip install pybm3d
```