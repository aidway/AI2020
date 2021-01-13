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
    
    
    