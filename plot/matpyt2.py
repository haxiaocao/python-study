import matplotlib.pyplot as plt
import numpy as np

# reference:https://www.runoob.com/w3cnote/matplotlib-tutorial.html
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']


def plot_default():
    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    C, S = np.cos(X), np.sin(X)

    plt.plot(X, C)
    plt.plot(X, S)

    plt.show()


def plot_default_detail():
    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    C, S = np.cos(X), np.sin(X)

    # figure(figsize=(10,6), dpi=80)
    # plot:color="green", linewidth=1.0, linestyle="-", marker='*',markersize=15
    # decide the X,Y Range automatically.
    # xmin ,xmax = X.min(), X.max()
    # ymin, ymax = Y.min(), Y.max()

    # dx = (xmax - xmin) * 0.2
    # dy = (ymax - ymin) * 0.2

    # xlim(xmin - dx, xmax + dx)
    # ylim(ymin - dy, ymax + dy)
    # yticks([-1, 0, +1],[r'$-1$', r'$0$', r'$+1$'])
    plt.subplot(1, 1, 1)
    plt.plot(X, C, 'r', label="Cosin")
    plt.plot(X, S, 'g', label="Sin")

    plt.xlim(-4, 4)
    plt.xticks(np.linspace(-4, 4, 9, endpoint=True))

    plt.ylim(-1, 1)
    plt.yticks(np.linspace(-1, 1, 5, endpoint=True))

    plt.show()


def some_features():
    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    C, S = np.cos(X), np.sin(X)

    # figure(figsize=(10,6), dpi=80)
    # plot:color="green", linewidth=1.0, linestyle="-", marker='*',markersize=15
    # decide the X,Y Range automatically.
    # xmin ,xmax = X.min(), X.max()
    # ymin, ymax = Y.min(), Y.max()

    # dx = (xmax - xmin) * 0.2
    # dy = (ymax - ymin) * 0.2

    # xlim(xmin - dx, xmax + dx)
    # ylim(ymin - dy, ymax + dy)
    # yticks([-1, 0, +1],[r'$-1$', r'$0$', r'$+1$'])
    plt.subplot(1, 1, 1)
    plt.plot(X, C, 'r', label="Cosin")
    plt.plot(X, S, 'g', label="Sin")

    plt.xlim(-4, 4)
    plt.xticks(np.linspace(-4, 4, 9, endpoint=True))

    plt.ylim(-1, 1)
    plt.yticks(np.linspace(-1, 1, 5, endpoint=True))
    #plt.savefig('pyt.png', dpi=72)

    # 添加图例 legend
    plt.legend(loc='upper left')

    # gca() 函数以及 gcf() 函数来获取当前的坐标轴和图像
    # Spines
    # 为了将脊柱放在图的中间，我们必须将其中的两条（上和右）设置为无色，然后调整剩下的两条到合适的位置——数据空间的 0 点。
    ##l = plt.axhline(y=.5, xmin=0.25, xmax=0.75)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    # 如果坐标轴的label被遮住，改善效果
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.1))

    # 特殊点 添加 注释
    # annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',
    #  xy=(t, np.cos(t)), xycoords='data',
    #  xytext=(-90, -50), textcoords='offset points', fontsize=16,
    #  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.scatter(2, np.cos(2), color='y')
    plt.annotate("p", [2, np.cos(2)+0.1])

    plt.show()


def diff_color():
    n = 256
    X = np.linspace(-np.pi, np.pi, n, endpoint=True)
    Y = np.sin(2*X)

    plt.plot(X, Y+1, color='lightblue', label='', alpha=1.00)
    plt.fill_between(X, Y+1, 1, where=Y+1 > 1,
                     facecolor='#97FFFF', interpolate=True)
    # plt.fill_between(X, Y+1, 1, where=Y+1 < 1, facecolor='yellow')
    plt.axhline(y=1, c='lightblue')

    plt.plot(X, Y-1, color='blue', label='', alpha=1.00)
    plt.fill_between(X, Y-1, -1, where=Y-1 < -1, facecolor='green')
    plt.fill_between(X, Y-1, -1, where=Y-1 > -1, facecolor='orange')
    # plt.xticks([])
    # plt.yticks([])
    plt.show()


def scatter_im():
    n = 1024
    X = np.random.normal(0, 1, n)
    Y = np.random.normal(0, 1, n)
    T = np.arctan2(Y, X)

    plt.axes([0.025, 0.055, 0.92, 0.92])
    plt.scatter(X, Y, s=75, alpha=.5, c=T)
    plt.xlim(-1.5, 1.5), plt.xticks([])
    plt.ylim(-1.5, 1.5), plt.yticks([])
    plt.title("散点图")
    plt.show()


def bar_im():
    n = 12
    X = np.arange(n)
    Y1 = (1-X/float(n)) * np.random.uniform(0.5, 1.0, n)
    Y2 = (1-X/float(n)) * np.random.uniform(0.5, 1.0, n)

    plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
    plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

    for x, y in zip(X, Y1):
        plt.text(x, y+0.03, '%.2f' % y, ha='center', va='bottom')
    for x, y in zip(X, Y2):
        plt.text(x, -y-0.03, '%.2f' % y, ha='center', va='top')

    plt.ylim(-1.2, +1.2)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def contour_im():
    def f(x, y): return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

    n = 256
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)

    # 坐标轴[距离左边，距离下边，坐标轴宽度，坐标轴高度],范围是(0,1) 倍数
    # 确定起始位置：距离左边，距离下边  确定图形结束位置：坐标轴宽度，坐标轴高度
    plt.axes([0.025, 0.025, 0.97, 0.97])
    plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap='hot')
    C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)
    plt.clabel(C, inline=1, fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.show()


'''
灰度图
'''


def imshow_gray():
    def f(x, y): return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

    n = 10
    x = np.linspace(-3, 3, 4*n)
    y = np.linspace(-3, 3, 3*n)
    X, Y = np.meshgrid(x, y)
    # 坐标轴[距离左边，距离下边，坐标轴宽度，坐标轴高度],范围是(0,1) 倍数
    # 确定起始位置：距离左边，距离下边  确定图形结束位置：坐标轴宽度，坐标轴高度
    plt.axes([0.025, 0.025, 0.95, 0.95])
    # print(plt.axes)
    plt.imshow(f(X, Y), cmap='bone', origin='lower', interpolation='bilinear')

    # shrink 总长度的一部分，从两端收缩
    plt.colorbar(shrink=.8)
    plt.xticks([]), plt.yticks([])
    #plt.legend(loc='upper right')
    # plt.savefig('imshow.png',dpi=50)
    plt.show()


def imshow_gray_frompic():
    import matplotlib.image as mpimg
    picdir = r'H:\图片_图像\61mONPx23jL._SL500_AC_SS350_.jpg'
    img = mpimg.imread(picdir)
    plt.imshow(img, cmap='gray')
    plt.title('Original  image')
    plt.show()


def pie_im():
    n = 10
    labels = ['娱乐', '育儿', '饮食', '房贷', '交通', '教育', '消费', '人情', '购物', '其它']
    Z = np.random.uniform(0, 1, n)
    # print(Z)
    explode = (0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.05)
    plt.pie(Z, labels=labels, shadow=False, explode=explode,
            autopct='%1.1f%%', startangle=90)
    plt.title("花钱方式Ways", color='r')
    plt.legend(loc='upper right', fontsize=10,
               borderaxespad=0.01, bbox_to_anchor=(1.25, 0.8))
    plt.show()


def quiver_im():
    n = 8
    X, Y = np.mgrid[0:n, 0:n]
    plt.quiver(X, Y)
    plt.show()


def grid_im():
    plt.figure('grid 表格major and minor line')
    axes = plt.gca()
    axes.set_xlim(0, 4)
    axes.set_ylim(0, 3)
    # 去掉显示坐标值
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    axes.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    axes.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    axes.yaxis.set_major_locator(plt.MultipleLocator(1))
    axes.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    axes.grid(1, which='minor', axis='x', linestyle='--',
              color='green', linewidth=0.25)
    axes.grid(1, which='major', axis='x', linestyle='-',
              color='green', linewidth=0.75)
    axes.grid(1, which='minor', axis='y', linestyle='--',
              color='green', linewidth=0.25)
    axes.grid(1, which='major', axis='y', linestyle='-',
              color='green', linewidth=0.75)
    plt.show()


def multi_grid():
    plt.subplot(2, 1, 1)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 4)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 5)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 6)
    plt.xticks([])
    plt.yticks([])

    plt.show()


def im_axes():
    plt.axes([0.1, 0.1, .8, .8])
    plt.xticks([]), plt.yticks([])
    plt.text(0.6, 0.5, 'axes([0.1,0.1,.8,.8])',
             ha='center', va='center', size=10, alpha=.5)

    plt.axes([0.2, 0.2, .3, .3])
    plt.xticks([]), plt.yticks([])
    plt.text(1, 0.1, 'axes([0.2,0.2,.3,.3])',
             ha='center', va='center', size=16, alpha=.5)

    # #plt.savefig("../figures/axes.png",dpi=64)
    plt.show()
