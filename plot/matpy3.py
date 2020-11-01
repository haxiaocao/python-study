import squarify
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# 一文搞懂matplotlib中的颜色设置
# https://blog.csdn.net/weixin_43569478/article/details/107724780
# matplotlib各种画图 https://blog.csdn.net/kun1280437633/article/details/80841364?utm_medium=distribute.pc_relevant.none-task-blog-title-7&spm=1001.2101.3001.4242
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False


def GDP():
    GDP = [12406.8, 13908.57, 9386.87, 9143.64]
    plt.title('城市GDP')
    plt.bar(range(4), GDP, color='steelblue', alpha=0.5, width=.3)
    plt.ylabel('GDP')
    plt.xlabel('城市')
    plt.xticks(range(4), ['北京市', '上海市', '天津市', '重庆市'])
    plt.ylim([0, 20000])
    plt.yticks(np.arange(0, 25000, 5000))
    for x, y in enumerate(GDP):
        plt.text(x, y + 100, '%s ' % round(y, 1), ha='center', color='b')

    # 去掉边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 调整label的显示位置
    ax.yaxis.set_label_coords(0.01, 1.02)
    ax.yaxis.label.set_color('red')
    ax.xaxis.set_label_coords(1.01, -0.03)
    ax.xaxis.label.set_color('red')
    plt.show()


def price():
    price = [39.5, 39.9, 45.4, 38.9, 33.34]

    plt.title('不同平台书的最低价比较', color='red')
    plt.barh(range(5), price, color='steelblue', alpha=0.5, height=0.5)
    plt.yticks(range(5), ['亚马逊', '当当网', '中国图书网', '京东', '天猫'])
    plt.xlabel('价格')
    plt.xlim(0, 50)

    for x, y in enumerate(price):
        # here x,y是颠倒过来的
        plt.text(y - 2, x, '%s' % y, va='center')

    plt.show()


def two_bar_YiWan():
    plt.title('亿万财富家庭数Top5城市分布')
    labels = ['北京', '上海', '香港', '深圳', '广州']
    plt.xticks(range(5), labels)
    plt.yticks(np.arange(0, 30000, 5000))
    plt.ylim(0, 25000)
    plt.xlabel('TOP 5 城市')
    plt.ylabel('家庭数量')
    bar_width = 0.3

    Y2016 = [17000, 15000, 11000, 5000, 3900]
    plt.bar(np.arange(5),
            Y2016,
            color='green',
            alpha=0.8,
            width=bar_width,
            label='2016')
    for x, y in enumerate(Y2016):
        # here x,y是颠倒过来的
        plt.text(x, y, '%s' % y, ha='center')

    # 水平移动，两条
    Y2017 = [17400, 14800, 12000, 5200, 4020]
    plt.bar(np.arange(5) + bar_width,
            Y2017,
            color='steelblue',
            alpha=0.8,
            width=bar_width,
            label='2017')
    for x, y in enumerate(Y2017):
        # here x,y是颠倒过来的
        plt.text(x + bar_width, y, '%s' % y, ha='center')

    plt.legend()
    # plt.tick_params(top = 'off', right = 'off')
    plt.show()


def pie_score():
    edu = [0.2515, 0.3724, 0.3336, 0.0368, 0.0057]
    labels = ['中专', '大专', '本科', '硕士', '其他']
    colors = ['#9999ff', '#ff9999', '#7777aa', '#2442aa', '#dd5555']
    plt.pie(edu,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=180,
            wedgeprops={
                'linewidth': 1.5,
                'edgecolor': 'green'
            },
            textprops={
                'fontsize': 12,
                'color': 'k'
            },
            explode=(0, 0.075, 0, 0, 0.1))
    plt.title('芝麻信用失信用户教育水平分布')
    plt.legend(loc='upper right',
               fontsize=8,
               borderaxespad=0.1,
               bbox_to_anchor=(1.15, 0.9))

    plt.show()


def treemap():
    # 树地图
    name = [
        '国内增值税', '国内消费税', '企业所得税', '个人所得税', '进口增值税、消费税', '出口退税', '城市维护建设税',
        '车辆购置税', '印花税', '资源税', '土地和房税', '车船税烟叶税等'
    ]
    income = [1361, 1042, 320, 291, 175, 111, 414, 63, 3908, 856, 801, 868]
    # color不够，循环来凑?
    colors = [
        'steelblue', '#9999ff', 'red', 'indianred', 'yellow', 'orange', 'green'
    ]
    plot = squarify.plot(
        sizes=income,  # 指定绘图数据
        label=name,  # 指定标签
        color=colors,  # 指定自定义颜色
        alpha=0.6,  # 指定透明度
        value=income,  # 添加数值标签
        edgecolor='white',  # 设置边界框为白色
        linewidth=3  # 设置边框宽度为3
    )

    plt.rc('font', size=18)
    plot.set_title('2017年8月中央财政收支情况', fontdict={'fontsize': 15})
    plt.axis('off')
    # # 去除上边框和右边框刻度
    # plt.tick_params(top = 'off', right = 'off')

    plt.show()


def histogram():
    # seed(): 设置种子是为了输出图形具有可重复性
    np.random.seed(20170717)
    mu, sigma = 100, 15

    x = mu + sigma * np.random.randn(10000)
    plt.hist(
        x,
        20,  #直接平分的个数.
        density=True,  #即normed,true为标准化，否则就是频率count.
        alpha=0.75,
        facecolor='g',  #color
        rwidth=1,
        edgecolor='k',  #指定直方图的边界色
        #  align='mid',
        orientation='vertical',
        histtype='bar')
    plt.title('直方图')

    # 为了测试数据集是否近似服从正态分布，需要在直方图的基础上再绘制两条线，一条表示理论的正态分布曲线，另一条为核密度曲线，目的就是比较两条曲线的吻合度，越吻合就说明数据越近似于正态分布。

    # 生成正态曲线的数据
    from scipy.stats import norm
    normalValue = np.linspace(x.min(), x.max(), 100)
    normal = norm.pdf(normalValue, x.mean(), x.std())
    line1, = plt.plot(normalValue, normal, 'r-', linewidth=1)

    # 生成概率密度曲线的数据
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(x)
    kde_value = np.linspace(x.min(), x.max(), 1000)
    line2, = plt.plot(kde_value, kde(kde_value), 'y-', linewidth=2)

    plt.legend()
    plt.show()

    #累计频度直方图histogram
    plt.hist(
        x,
        bins=np.linspace(x.min(), x.max(), 10),
        # density=True,
        alpha=0.75,
        color='g',  #color
        rwidth=1,
        cumulative=True,  #累计频率直方图
        edgecolor='k',  #指定直方图的边界色
        #  align='mid',
        orientation='vertical',
        histtype='bar')
    plt.title('累计频度直方图')
    plt.show()

    ## datasets为多个一维数据集
    x2 = mu + sigma * np.random.randn(3000)
    plt.hist((x, x2), 20, density=True, alpha=.4, color=('r', 'g'))
    plt.title('多个一维数据集')
    plt.show()

    ##datasets 为 2D-ndarray
    t = np.random.randn(3600)
    t.shape = (900, 4)
    xx = mu + sigma * t
    p, bins, patches = plt.hist(xx, bins=20, density=True, alpha=0.4)
    plt.title('2D-ndarray数据集')
    plt.show()

    ###histogram 叠加
    xx = mu + 0.1 * sigma * np.random.randn(3000)
    color = ('r', 'g')
    plt.hist(
        (x, x2),
        20,
        density=True,
        alpha=.4,
        color=color,
        #  label=['Num1', 'Num2'],
        stacked=True,
        fill=True,
        edgecolor='k')
    plt.legend(['Num1', 'Num2'], loc='upper left')
    plt.title('两个DataSet叠加')
    plt.show()


def plot_dash():
    # register the converters explicitly
    # from pandas.plotting import register_matplotlib_converters
    # register_matplotlib_converters()

    num = 50
    article_num = np.random.randint(1000, 10000, num)
    date_seq = pd.date_range(start='20170101', periods=num)
    #设置绘图风格
    plt.style.use('ggplot')
    #To list all available styles, use:
    print(plt.style.available)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(date_seq,
             article_num,
             linestyle='-',
             linewidth=1,
             marker='o',
             markerfacecolor='r',
             markeredgecolor='g',
             markersize=6)
    plt.title('每日公众号阅读数')
    plt.xlabel('日期')
    plt.ylabel('人数')
    # 剔除图框上边界和右边界的刻度
    # plt.tick_params(top='on', right='on', direction='in')

    fig.autofmt_xdate(rotation=45)

    #设置日期格式
    ax = plt.gca()
    import matplotlib.dates as mdates
    myFmt = mdates.DateFormatter('%Y/%m/%d')
    ax.xaxis.set_major_formatter(myFmt)

    #设置x轴每个刻度的间隔天数
    import matplotlib.ticker as ticker
    xlocator = ticker.MultipleLocator(5)  # tick, interval
    ax.xaxis.set_major_locator(xlocator)

    plt.show()


def plot_dash_multple():
    # register the converters explicitly
    # from pandas.plotting import register_matplotlib_converters
    # register_matplotlib_converters()

    num = 50
    article_num = np.random.randint(1000, 10000, num)
    date_seq = pd.date_range(start='20170101', periods=num)
    #设置绘图风格
    plt.style.use('ggplot')
    #To list all available styles, use:
    print(plt.style.available)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(date_seq,
             article_num,
             linestyle='-',
             linewidth=1,
             marker='o',
             markerfacecolor='r',
             markeredgecolor='g',
             markersize=6,
             label='阅读量')

    article_count = article_num * 0.7 - 100
    plt.plot(date_seq,
             article_count,
             linestyle='-',
             linewidth=1,
             marker='o',
             markerfacecolor='b',
             markeredgecolor='black',
             markersize=6,
             label='阅读人次')

    plt.title('每日公众号阅读数')
    plt.xlabel('日期')
    plt.ylabel('人数')
    # 剔除图框上边界和右边界的刻度
    # plt.tick_params(top='on', right='on', direction='in')

    fig.autofmt_xdate(rotation=45)

    #设置日期格式
    ax = plt.gca()
    import matplotlib.dates as mdates
    myFmt = mdates.DateFormatter('%Y/%m/%d')
    ax.xaxis.set_major_formatter(myFmt)

    #设置x轴每个刻度的间隔天数
    import matplotlib.ticker as ticker
    xlocator = ticker.MultipleLocator(5)  # tick, interval
    ax.xaxis.set_major_locator(xlocator)

    plt.legend()
    plt.show()


def plot_stackplot():
    num = 50
    article_num = np.random.randint(1000, 10000, num)
    date_seq = pd.date_range(start='20170101', periods=num)
    #设置绘图风格
    plt.style.use('ggplot')
    #To list all available styles, use:
    print(plt.style.available)

    fig = plt.figure(figsize=(10, 6))
    labels = ['阅读数量', '阅读人数']
    colors = ['b', 'y']
    article_count = article_num * 0.7 - 100
    plt.stackplot(date_seq,
                  article_num,
                  article_count,
                  labels=labels,
                  colors=colors)

    fig.autofmt_xdate(rotation=45)

    # plt.xticks()
    #设置日期格式
    ax = plt.gca()
    import matplotlib.dates as mdates
    myFmt = mdates.DateFormatter('%Y/%m/%d')
    ax.xaxis.set_major_formatter(myFmt)

    plt.legend()
    plt.show()


def plot_scatter_multiple():
    num = 50
    y = np.random.randint(1000, 10000, num)
    x = np.linspace(100, 1000, num=num)
    colormap = np.array(['r', 'g', 'b'])
    # categories = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # Supervised
    categories = [i % 3 for i in y]

    #颜色映射 legend
    import matplotlib.patches as mpatches
    pop_a = mpatches.Patch(color='r', label='case 1')
    pop_b = mpatches.Patch(color='g', label='case 2')
    pop_c = mpatches.Patch(color='b', label='case 3')

    #这里也可以通过分类，scatter多次，每次一组.
    #marker 参考: matplotlib.markers.MarkerStyle
    plt.scatter(
        x,
        y,
        marker='s',
        s=30,  # 散点的大小（如果实现气泡的效果，调整）
        c=colormap[categories],
        alpha=0.8,
        edgecolors='black',
        # label=labels[categories] #此路不通，哈.
    )

    # plt.tick_params(top='on', right='on')
    plt.legend(handles=[pop_a, pop_b, pop_c])
    plt.show()


def plot_buttle():
    # create data
    x = np.random.rand(40)
    y = np.random.rand(40)
    z = np.random.rand(40)

    # use the scatter function
    plt.scatter(x, y, s=z * 1000, alpha=0.5)
    plt.show()


def plot_linear_regression():
    num = 50
    y = np.random.randint(1000, 2000, num)
    x = np.linspace(100, 1000, num=num)
    #设置绘图风格
    plt.style.use('ggplot')
    #To list all available styles, use:
    print(plt.style.available)

    plt.scatter(
        x,
        y,
        marker='o',
        s=30,  # 散点的大小（如果实现气泡的效果，调整）
        alpha=0.8,
        edgecolors='black',
        label='观测点'  #此路不通，哈.
    )

    from sklearn.linear_model import LinearRegression

    reg = LinearRegression().fit(x.reshape(-1, 1), y)
    pred = reg.predict(x.reshape(-1, 1))
    plt.plot(x, pred, linewidth=2, label='线性回归线', color='g')
    plt.legend(loc='upper left')
    plt.show()
