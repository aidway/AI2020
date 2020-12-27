import matplotlib.pyplot as plt

'''
всм╪╪Д╬Ю subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
'''


plt.figure(figsize=(5,4))
plt.subplots_adjust(hspace=0.4, wspace = 0.35)

plt.subplot(221)
plt.plot()
plt.title('img1')
plt.subplot(222)
plt.plot()
plt.title('img2')
plt.subplot(223)
plt.plot()
plt.title('img3')
plt.subplot(224)
plt.plot()
plt.title('img4')


