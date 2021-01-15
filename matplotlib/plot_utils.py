import matplotlib.pyplot as plt

def rgb2hex(rgb):
    '''
    sample: rgb2hex((66,115,197))
    '''
    return '#%02x%02x%02x' % rgb

def hex2rgb(value):
    '''
    sample: hex2rgb('4273c5')
    '''
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    
    
def show_color():
    '''
    function: show color list
    '''
    y = [5,5,5,5,5,5,5   ]
    x = range(1,len(y)+1)
    
    plt.bar(x, y, color=['#0e72cc','#6ca30f','#f59311','#fa4343','#16afcc','#85c021','#d12a6a'])  

    plt.title('color list')
    print(['1:0e72cc','2:6ca30f','3:f59311','4:fa4343','5:16afcc','6:85c021','7:d12a6a'])
    plt.show()
    
# https://blog.csdn.net/sinat_41299610/article/details/106912048
# https://mp.weixin.qq.com/s?__biz=MzA5NDAxMzg3OQ==&mid=2651949936&idx=3&sn=4540c0d3197e4285a71456b2bab41520&chksm=8bb03ba8bcc7b2be9e4fc8fe6d9ac1e1209bbc67a4ad6a25c28383e9f74d79580059ede3d543&mpshare=1&scene=1&srcid=0113LDvLSbf8Xxrvpx1tX419&sharer_sharetime=1610503142338&sharer_shareid=7c89213c3e6ce9f7537ee097570db754&key=bd73c23a62b97dffa4e41bdff8b3835b5f2227d10df260b3502d609399f353a00f9c1b829c1787735579700c0949c00b4d4ffce444bd9eae48bcba5bb52eabaeb8a3a96914b9687cceb878555d7a96bbc190a925902e70935caed16fe0094144b11d28ac35ead5fb9b19b686c3fcfa1b473b0fb6df991737787bcdfbb5b7d8a9&ascene=1&uin=MTE5NjMwMDkwMg%3D%3D&devicetype=Windows+10+x64&version=6300002f&lang=zh_CN&exportkey=ATyU2XbIM%2FLlZbiscKEPAXo%3D&pass_ticket=9nzRh7g1LwdKmJiClXQraIkXaW1a03JdAJ5YGYaD4Lm2dWwah50TF1a7RpoK0X4%2F&wx_header=0