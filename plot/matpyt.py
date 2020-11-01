import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import gridspec as gridspec

# 如果使用 ipython,添加 -pylab，可以直接交互方式使用
# 用于设置字体
plt.rcParams['font.family'] = ['STKaiti']

# lw:linewidth,
# fc:facecolor
# s: point style
# c: color
# cmap: color mapping
# xlim,ylim: lim->limit
# 填充颜色需要使用 fill
# Alias	Property
# 'lw''linewidth'
# 'ls''linestyle'
# 'c''color'
# 'fc''facecolor'
# 'ec''edgecolor'
# 'mew''markeredgewidth'
# 'aa''antialiased'

'''
输出字体，用于解决中文显示问题
'''


def out_font():
    a = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
    for i in a:
        print(i)


def plotone():
    x = np.arange(1, 15, 2)
    y = 2*x+5
    plt.plot(x, y, 'b')
    # plt.plot(x, y, 'r*', marker='o')
    plt.title("matplotlib demo")
    plt.xlabel("x 坐标")
    plt.ylabel("y axis")

    for xy in zip(x, y):
        X, Y = xy
        plt.annotate("(%s,%s)" % xy, xy=xy, xytext=(X, Y+1), weight='bold', color='aqua',
                     arrowprops=dict(arrowstyle='-|>',
                                     connectionstyle='arc3', color='red'),
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1, alpha=0.4))
    # plt.annotate("")

    # 2,2: data coordinate
    plt.text(10, 10, 'Here you are', family='fantasy', fontsize=12,
             style='italic', color='mediumvioletred')

    plt.show()


def plottwo():
    x = np.arange(0,  3 * np.pi,  0.1)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    # figure()命令在这儿可以不写，因为figure(1)将会被默认执行
    #plt.figure("figure name ")
    plt.subplot(2, 1, 1)
    plt.plot(x, y_sin)
    plt.title('Sine')
    #plt.xlabel('X axis')

    plt.subplot(2, 1, 2)
    plt.plot(x, y_cos)
    plt.title('Cosine')
    plt.xlabel('X axis')
    plt.show()


def bar():
    x = [5, 8, 10]
    y = [12, 16, 6]
    x2 = [6, 9, 11]
    y2 = [6, 15, 7]
    plot1 = plt.bar(x, y)
    plot2 = plt.bar(x2, y2, color='r')

    plt.title('Bar graph')
    plt.ylabel('Y axis')
    plt.xlabel('X axis')

    #plt.legend('best',(plot1, plot2), ('red line','green circles'))
    plt.show()


def histogram():
    a = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])
    # hist, bins = np.histogram(a, bins=[0, 20, 40, 60, 80, 100])
    # print(hist)
    # print(bins)
    plt.hist(a, bins=[0, 20, 40, 60, 80, 100])
    plt.title('histogram Value')
    plt.show()

# plotone()
# plottwo()
# bar()
# histogram()


def drawline():
    t = np.arange(-1, 2, .01)
    s = np.sin(2*np.pi*t)

    plt.plot(t, s)
    plt.axis([-1, 2, -3, 2])
    # 画一条直线
    plt.axhline(y=0, c='r')
    # xmin,xmax是倍数，范围(0,1),0最左，1最右边
    plt.axhline(y=-1, c='r', xmin=0, xmax=0.75, linewidth=4)
    plt.axvline(x=0.5, c='g')
    plt.axvline(x=0, c='g', ymin=0.2, ymax=0.5, linewidth=4)

    # 画一段范围
    plt.axhspan(0, .75, facecolor='0.5', alpha=0.5)
    plt.axvspan(-1, 0, facecolor='0.5', alpha=0.5)

    # plt.hlines(hline, xmin=plt.gca().get_xlim()[0], xmax=plt.gca().get_xlim()[1], linestyles=line_style, colors=color)
    plt.text(0, 1, "1的text数值")

    plt.show()


def mul_subplot():
    n = np.array([0, 1, 2, 3, 4, 5])
    x = np.linspace(-0.75, 1., 100)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    axes[0].scatter(x, x + 0.25*np.random.randn(len(x)))
    axes[0].set_title("scatter plot")

    axes[1].step(n, n**2, lw=2)
    axes[1].set_title("step plot")

    axes[2].bar(n, n**2, align="center", width=0.5, alpha=0.5)
    axes[2].set_title("bar plot")

    axes[3].fill_between(x, x**2, x**3, color="green", alpha=0.5)
    axes[3].set_title("fill entity between two Y plot")

    plt.show()


def scatter_im():
    N = 50
    x = np.random.rand(N)
    y = np.random.rand(N)
    # 点的颜色
    color = 2*np.pi*np.random.rand(N)
    # 点的大小
    area = np.pi*(15*np.random.rand(N))**2

    plt.scatter(x, y, c=color, alpha=.5, cmap=plt.cm.jet, s=area)
    #plt.annotate(class_label, (data_arr[:, 0][i], data_arr[:, 1][i]))
    plt.show()


def bar_and_grid():
    a = list(range(10))
    b = np.random.randint(1, 100, 10)
    plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(4, 1)
    # 画图表
    ax1 = plt.subplot(gs[:3, 0])
    ax1.bar(a, b)
    plt.xlabel('x Label')
    plt.ylabel('y Label')
    plt.title('bar title')

    # 插入表格
    ax2 = plt.subplot(gs[3, 0])
    plt.axis('off')
    rowLables = ['第一行', '第二行', '第三行']
    cellText = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    table = plt.table(cellText=cellText, rowLabels=rowLables,
                      loc='center', cellLoc='center', rowLoc='center')
    # print(type(table))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    # 表格缩放
    table.scale(0.7, 2)

    plt.show()
