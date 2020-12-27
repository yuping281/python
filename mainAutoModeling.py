import pandas as pd

import myAutoModeling as mam

def computeWeights(data,label):
    model = -1
    mean_features, duration_list = mam.feature_eng(data, label)
    cpl = mam.myClassification(mean_features, class_num)  ##判断类别的DP1大小值,在在线模型中需要作为参数输入

    w_l_1, r_l_1, loss_l_1 = mam.fitting_liquid_data(mean_features, class_num, duration_list,
                                                     targets=label['Ref. Q_L'].values)
    w_g_1, r_g_1, loss_g_1 = mam.fitting_liquid_data(mean_features, class_num, duration_list,
                                             targets=label['Ref. Q_g'].values, learningRate=0.0000005)
    ## w_g为气相模型系数
    myCentroids = mam.myClassification(mean_features, class_num, model='kmeans')
    w_l_2, r_l_2, loss_l_2 = mam.fitting_liquid_data(mean_features, class_num, duration_list,
                                                     targets=label['Ref. Q_L'].values)
    ## w_l为液相模型系数
    w_g_2, r_g_2, loss_g_2 = mam.fitting_gas_data(mean_features, class_num, duration_list,
                                                  targets=label['Ref. Q_g'].values,
                                                  learningRate=0.0000005)
    if loss_l_1<=loss_l_2:
        model = 1
        return w_l_1,w_g_1,cpl,model
    else:
        model = 2
        return w_l_2,w_g_2,myCentroids,model
if __name__ == '__main__':
    label = pd.read_csv('Ref_6005.csv', header=0, engine='python')
    data = pd.read_csv('data_6005.csv', header=0)
    class_num = mam.select_categories(label)
    label['Ref. Q_L'] /= 24.0   ##因为标定值是吨/天，转化成吨/小时
    label['Ref. Q_g'] /= 24.0   ##因为标定值是标方/天，转化成标方/小时
    w_l,w_g,my_points,model = computeWeights(data,label)
    #这4个参数是需要传给EMB板卡中基础模型的参数,其中w_l,w_g分别为计算得到的模型系数，my_points在使用dp1直接
    #分类的方法中代表dp1的分类分位点，在使用KMeans聚类方法时代表聚类后各个中心点坐标，model为选用何种分类方法
    #的标志符，1为根据dp1直接分类，2为使用KMeans聚类


