# 环境

* OS:debian 9.4

| 软件   | 版本  |
| :------- | ---- |
| debian     | 9.4   |
| tensorflow | 1.9.0 |



# 问题

执行ocr合同清单检测脚本时，报Segmentation fault，见下图：
![d](.\\imgs\\Segmentation.png)

![d](E:\\git\\AI2020\\problem\\OCR_Segmentation_fault\\core_file.png)

# 处理过程

## 解析core文件

```shell
gdb -c core.25700
```

![d](E:\\git\\AI2020\\problem\\OCR_Segmentation_fault\\gdb1.png)

* 结论：无明显发现。

## 查看系统资源限制

![d](E:\\git\\AI2020\\problem\\OCR_Segmentation_fault\\ulimit.png)

		* 尝试修改stack大小，无用。
		* 查看程序执行期间操作系统cpu、memory、open files，无异常。





