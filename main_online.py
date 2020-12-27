import pandas as pd
import numpy as np

import features_compute
import myKmeans as mk

model = 2
my_points = [[1.2,2.1],[1.6,1.8],[2.2,2.9],[3.1,4.2]]
w_l = [0.22,0.33,0.44,0.55]
w_g = [0.22,0.33,0.44,0.55]
#以上4个参数的取值均只是为了测试代码能否正常工作而随机赋值，实际以自动建模框架传入的参数为准。
#
matrix = pd.read_csv('all_result.csv',usecols=[2,3,4,5,6],header=None,
                     names=['P','DP1','DP2','Temp','WLR'])
A = 0.000491
beta = 0.6
c_d = 0.95
Epsilon = 1

p1 = matrix.iloc[:,1]
p2 = matrix.iloc[:,2]
P = matrix.iloc[:,0]
T = matrix.iloc[:,3]
WLR = matrix.iloc[:,4]

p1[p1<0] = 0
p1_h = p1
p2_h = p2

rou_g = features_compute.compute_density(matrix)
rou_o = 850

q_LL = [0.0]*(len(matrix)//600*600)
q_GG = [0.0]*(len(matrix)//600*600)
for k in range(0,len(matrix)//600,600):

    n = k//600
    DP1 = p1_h[k:k+600].reset_index(drop=True)
    DP2 = p2_h[k:k+600].reset_index(drop=True)
    tt = T[k:k+600].reset_index(drop=True)
    pp = P[k:k+600].reset_index(drop=True)
    wlr = WLR[k:k+600].reset_index(drop=True)
    Den_GG = rou_g[k:k+600].reset_index(drop=True)
    m_p = pp.mean()
    m_t = tt.mean()
    m_1 = DP1.mean()
    m_2 = DP2.mean()
    m_w = wlr.mean()
    st1 = DP1.std()
    st2 = DP2.std()
    rou_l = rou_o + m_w*(100-rou_o)

    for i in range(600):
        q_L = [0.0]*600
        q_G = [0.0]*600
        kk = 600*n
        if model == 1:
            for point in range(len(my_points)):
                if m_1<=my_points[point]:
                    ff = w_l[point]
                    gg = w_g[point]
                    break
                else:
                    ff =sum(w_l)/len(w_l)
                    gg = sum(w_g)/len(w_g)   #对于大于分位点最大值的信号，暂时采用各类系数的平均值进行后续计算

        elif model == 2:
            minDist = np.inf  # 初始设置值为无穷大
            minIndex = -1
            # 计算得到该600帧数据的前差压、后差压均值与聚类中心最近的中心点，根据中心点选择对应的模型系数
            for j in range(len(my_points)):
                distJ = mk.distEclud(np.array(my_points[j]), np.array([m_1,m_2]))  #计算欧式距离
                if distJ < minDist:
                    minDist = distJ  # 更新 最小的距离
                    minIndex = j
            ff = w_l[minIndex]
            gg = w_g[minIndex]

        q_L[i] = ff * (c_d * Epsilon * A * (2 * DP1[i] * 1000 / rou_l) ** 0.5) / (1 - beta ** 4) ** (1 / 2) * 3600
        q_G[i] = gg * (c_d * Epsilon / (1 - beta ** 4) ** 0.5 * A * (2 / Den_GG[i] * DP1[i] * 1000) ** 0.5 * 3600) \
                 * 293 * (pp[i] + 0.1) / 0.1 / (tt[i] + 273)

        if st1<0.1 and st2<0.1 and m_1<0.18:
                q_L[i] = 0
                q_G[i] = 0
        elif DP1[i] < 0:
                q_L[i] = 0
                q_G[i] = 0
            # % QL1 = sum(QL) * 24 / i;
        if q_L[i] > (c_d*Epsilon*A*(2*DP1[i]*1000/rou_l)**0.5)/(1-beta**4)**(1/2)*3600:
            q_L[i] = (c_d*Epsilon*A*(2*DP1[i]*1000/rou_l)**0.5)/(1-beta**4)**(1/2)*3600

        if q_G[i] > (c_d*Epsilon/(1-beta**4)**0.5*A*(2/Den_GG[i]*DP1[i]*1000)**0.5*3600)\
                         *293*(pp[i]+0.1)/0.1/(tt[i]+273):
            q_G[i] = (c_d*Epsilon/(1-beta**4)**0.5*A*(2/Den_GG[i]*DP1[i]*1000)**0.5*3600)\
                         *293*(pp[i]+0.1)/0.1/(tt[i]+273)

        if q_L[i] < 0:
            q_L[i] = 0
        if q_G[i] < 0:
            q_G[i] = 0

    q_LL[k: k + 599] = [round(i, 5) for i in q_L]
    q_GG[k: k + 599] = [round(i, 5) for i in q_G]










