## conda版本
4.7.12

## 需求
将A机器（可连外网）的conda环境ocr迁移到B机器（离线机器，不可连外网）。

## 步骤
1.将A机器的anaconda3/envs/ocr目录拷贝到 B机器的/tmp/目录  
2.将A机器的anaconda3/pkgs的文件拷贝到 B机器的anaconda3/pkgs/目录  
3.将B机器anaconda3/pkgs/中的urls和urls.txt删除  
4.B机器  
```shell
conda create -n ocr --clone /tmp/ocr --offline
```

## 问题
1.could not parse explicit url  
V4.6.14已知bug  
https://github.com/conda/conda/issues/9272  
2.RuntimeError ...using a remote connection in offline mode  
将创建环境的命令修改为：  
```shell
conda create -n ocr --clone /tmp/ocr
```
3.conda activate激活环境报错CommandNotFoundError: Your shell has not been properly configured  
执行以下命令：  
```shell
source activate ocr
```
