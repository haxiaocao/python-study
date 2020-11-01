import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def mask_plot_part():
    corr = np.corrcoef(np.random.randn(10, 200))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True)
    plt.show()


def plot_whole_heatmap():
    # kwargs :  https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.axes.Axes.pcolormesh.html
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False

    num = 31
    data = np.random.randint(0, 100, size=num)
    date = pd.date_range(start='20201001', periods=31)
    df = pd.DataFrame({'date': date, 'score': data})
    df.score = df.score.astype('int')

    def week_of_month(target_date):
        for i in range(1, num + 1):
            d = datetime.datetime(target_date.year, target_date.month, i)
            if d.day - d.weekday() > 0:
                startdate = d
                break
        # 返回日期所属月份的第一周
        return (target_date - startdate).days // 7 + 1

    df['weekday'] = df.date.apply(pd.datetime.weekday)
    df['week_of_month'] = df.date.apply(week_of_month)
    #构建数据表-日历
    target = pd.pivot_table(data=df.iloc[:, 1:],
                            values='score',
                            index='week_of_month',
                            columns='weekday')
    # 缺失值填充（不填充的话pcolor函数无法绘制）
    target.fillna(0, inplace=True)
    # 删除表格的索引名称
    # target.index.name = None
    target.sort_index(ascending=False, inplace=True)

    # plt.figure(figsize=(10,10)) # set the whole figure size. 
    #常规做法，直接颜色图映射
    #plt.pcolor(target, cmap=plt.cm.Blues, edgecolors='white')
    #seaborn模块中的heatmap函数重新绘制一下热力，显示数据和颜色图例
    ax = sns.heatmap(
        target,  # 指定绘图数据
        cmap=plt.cm.Reds,  # 指定填充色
        linewidths=.1,  # 设置每个单元方块的间隔
        annot=True,  # 显示数值
        square=False,
        cbar=True,
    )
    plt.xticks(np.arange(7) + 0.5, ['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
    plt.yticks(np.arange(5) + 0.5, ['第五周', '第四周', '第三周', '第二周', '第一周'],
               rotation=0)
    # 旋转y刻度0度，即水平显示

    plt.tick_params(direction=None)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.title('十月份得分一览')
    plt.show()

plot_whole_heatmap()