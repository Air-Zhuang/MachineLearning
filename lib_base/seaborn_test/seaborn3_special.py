import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series,DataFrame

x=np.linspace(1,14,100)
y1=np.sin(x)
y2=np.sin(x+2)*1.25

print("==============seaborn设置图形显示效果=================")
def sinplot():
    plt.plot(x,y1)
    plt.plot(x,y2)


style = ['darkgrid', 'dark', 'white', 'whitegrid', 'tricks']        #seaborn提供五种背景风格
context=['paper','notebook','talk','poster']                        #seaborn提供四种纸张风格

sns.set_style(style[0])                                             #设置背景风格
sns.set_context(context[3])                                         #设置纸张风格

# sns.set()                                                         #清空样式

sinplot()
plt.show()

print("==============seaborn调色功能=================")
def sinplot2():
    x=np.linspace(0,14,100)
    plt.figure(figsize=(8,6))                                       #图大小
    for i in range(4):
        plt.plot(x,np.sin(x+i)*(i+0.75),label='sin(x+%s)*(%s+0.75)' % (i,i))
    plt.legend()


sns.set()                                                           #清空一下样式,显示seaborn原始的样式
sinplot2()
plt.show()

pal_style=['deep','muted','pastel','bright','dark','colorblind']    #seaborn默认提供的色板主题
sns.palplot(sns.color_palette('dark'))                              #深色主题色板
plt.show()

sns.set_palette(sns.color_palette('bright'))                        #设置风格为明亮的主题色板
sinplot2()
plt.show()

with sns.color_palette('dark'):                                     #临时设置深色的主题色板
    sinplot2()
    plt.show()