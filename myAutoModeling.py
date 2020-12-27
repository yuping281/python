import pandas as pd
import numpy as np

import features_compute
import linearRegression as lg
import myKmeans as mk

def feature_eng(data,label,time_range=600):
    duration_list = list(label['Duration'])
    den_G = features_compute.compute_density(data)
    data['Den_G'] = den_G
    data['Difference'] = data.DP1 - data.DP2
    mean_features = pd.DataFrame()

    for i in range(0, len(data), time_range):
        newfeatures = pd.DataFrame({
            'p_mean': [data.iloc[i:i + time_range, 1].mean()],
            't_mean': [data.iloc[i:i + time_range, 4].mean()],
            'DP1_mean': [data.iloc[i:i + time_range, 2].mean()],
            'DP2_mean': [data.iloc[i:i + time_range, 3].mean()],
            'DP1_std': [data.iloc[i:i + time_range, 2].std()],
            'DP2_std': [data.iloc[i:i + time_range, 3].std()],
            'WLR_mean': [data.iloc[i:i + time_range, 5].mean()],
            'Den_G_mean': [data.iloc[i:i + time_range, 6].mean()],
            'Diff_mean': [data.iloc[i:i + time_range, 7].mean()]})
        mean_features = pd.concat([mean_features, newfeatures], ignore_index=True)
    return mean_features,duration_list

def select_categories(label):
    """
    根据样本标签的数量选择分类的类别数
    """
    if len(label) < 10:
        raise ValueError('Labels are not enough!')
    elif (len(label) >= 10) and (len(label) < 20):
        return 5
    elif (len(label) >= 20) and (len(label) < 40):
        return 8
    elif (len(label) > 40) and (len(label) <= 60):
        return 10
    elif (len(label) > 60) and (len(label) <= 100):
        return 15
    else:
        return 20

def classificationPointList(data,n):
    cpl = [0]*(n)
    for i in range(n):
        cpl[i] = data.DP1_mean.min()+(data.DP1_mean.max()-data.DP1_mean.min())/n*(i+1)
    return cpl

def myClassification(data,n,model='default'):
    labels = []
    if model=='default':
        cpl = classificationPointList(data, n)
        for i in range(len(data)):
            for k in range(len(cpl)):
                if data.DP1_mean[i]<=cpl[k]:
                    labels.append(k)
                    break
            # if data.DP1_mean[i]>=cpl[-1]:
            #     labels.append(k+1)
        data['Label'] = labels
        return cpl
    elif model=='kmeans':
        dataset = np.array(data.iloc[:,2:4])
        myCentroids, clustAssing = mk.KMeans(dataset,n)
        data['Label'] = clustAssing[:,0]
        return myCentroids


def compute_general_Q(p, t, dp1, dp2, std1, std2, den_g, model='single_or', beta=0.480769, A=0.000491):
    """
    输入前后差压均值、标准差
    返回该前差压对应的虚高流量值
    model: 可选参数为'single_or','dual_or'，分别代表使用单虚高模型与双虚高模型
    """
    if model == 'single_or':
        q_L = np.zeros(len(dp1), )
        q_G = np.zeros(len(dp1), )
        for i in range(len(dp1)):
            if ((dp1[i] < 0.1) and (dp2[i] < 0.1)) and (std1[i] < 0.01 and std2[i] < 0.01):
                q_L[i] = 0
                q_G[i] = 0

            elif dp1[i] <= 0:
                q_L[i] = 0
                q_G[i] = 0

            else:
                q_L[i] = (0.99 / (1 - beta ** 4) ** 0.5 * A * (2 / 880 * dp1[i] * 1000) ** 0.5 * 3600)
                q_G[i] = (0.99 / (1 - beta ** 4) ** 0.5 * A * (2 / den_g[i] * dp1[i] * 1000) ** 0.5 * 3600) * 293 * (
                            p[i] + 0.1) / 0.1 / (t[i] + 273)
        return q_L, q_G
    elif model == 'dual_or':
        q_L1 = np.zeros(len(dp1), )
        q_G1 = np.zeros(len(dp1), )
        q_L2 = np.zeros(len(dp1), )
        q_G2 = np.zeros(len(dp1), )
        for i in range(len(dp1)):
            if ((dp1[i] < 0.1) and (dp2[i] < 0.1)) and (std1[i] < 0.01 and std2[i] < 0.01):
                q_L1[i] = 0
                q_G1[i] = 0
                q_L2[i] = 0
                q_G2[i] = 0

            elif dp1[i] <= 0:
                q_L1[i] = 0
                q_G1[i] = 0
                q_L2[i] = 0
                q_G2[i] = 0

            else:
                q_L1[i] = (0.99 / (1 - beta ** 4) ** 0.5 * A * (2 / 880 * dp1[i] * 1000) ** 0.5 * 3600)
                q_G1[i] = (0.99 / (1 - beta ** 4) ** 0.5 * A * (2 / den_g[i] * dp1[i] * 1000) ** 0.5 * 3600) * 293 * (
                            p[i] + 0.1) / 0.1 / (t[i] + 273)
                q_L2[i] = (0.99 / (1 - beta ** 4) ** 0.5 * A * (2 / 880 * dp2[i] * 1000) ** 0.5 * 3600)
                q_G2[i] = (0.99 / (1 - beta ** 4) ** 0.5 * A * (2 / den_g[i] * dp2[i] * 1000) ** 0.5 * 3600) * 293 * (
                            p[i] + 0.1) / 0.1 / (t[i] + 273)
        return q_L1, q_L2, q_G1, q_G2


def fitting_liquid_data(features_per_min, K, counts, targets, model='single_or', learningRate=0.00005, alpha=1.0):
    """
    输入features_per_min为标定时间段每分钟的特征值，K为对标定数据分类的类别数，counts为各个标定时间段时长（包含分钟数）的列表,targets为标定
    时间段的真实流量标签，类型为array
    输出为各类别的拟合权重系数Weights_or和各标定时间段按类别的累计流量矩阵regression_fit
    """
    if model == 'single_or':
        n = len(counts)
        ql_train_gmodel_or = compute_general_Q(features_per_min.p_mean.values,
                                               features_per_min.t_mean.values,
                                               features_per_min.DP1_mean.values,
                                               features_per_min.DP2_mean.values,
                                               features_per_min.DP1_std.values,
                                               features_per_min.DP2_std.values,
                                               features_per_min.Den_G_mean.values,
                                               model=model)[0]
        ql_per_min = ql_train_gmodel_or
        all_labels = features_per_min['Label'].values

        regression_fit = np.zeros((n, K))
        for i in range(n):
            range_data = ql_per_min[:counts[i]]
            labels = all_labels[:counts[i]]
            ql_per_min = np.delete(ql_per_min, [j for j in range(counts[i])])
            all_labels = np.delete(all_labels, [j for j in range(counts[i])])
            for kk in range(len(range_data)):
                for label in range(K):
                    if labels[kk] == label:
                        regression_fit[i][label] += range_data[kk]
            regression_fit[i] = regression_fit[i] / counts[i]

        Weights_or,loss = lg.liner_Regression \
            (regression_fit, targets.reshape(n, 1), learningRate=learningRate, Loopnum=2000000, alpha=alpha)
        return Weights_or, regression_fit,loss
    # =============================================================================
    #     '''
    #     使用双虚高模型计算
    #     '''
    # =============================================================================
    elif model == 'dual_or':
        n = len(counts)
        ql_train_gmodel_or1, ql_train_gmodel_or2 = compute_general_Q(features_per_min.p_mean.values,
                                                                     features_per_min.t_mean.values,
                                                                     features_per_min.DP1_mean.values,
                                                                     features_per_min.DP2_mean.values,
                                                                     features_per_min.DP1_std.values,
                                                                     features_per_min.DP2_std.values,
                                                                     features_per_min.Den_G_mean.values,
                                                                     model=model)[0:2]
        ql_per_min1, ql_per_min2 = ql_train_gmodel_or1, ql_train_gmodel_or2
        all_labels = features_per_min['Label'].values

        regression_fit = np.zeros((n, 2 * K))
        for i in range(n):
            range_data1 = ql_per_min1[:counts[i]]
            range_data2 = ql_per_min2[:counts[i]]
            labels = all_labels[:counts[i]]
            ql_per_min1 = np.delete(ql_per_min1, [j for j in range(counts[i])])
            ql_per_min2 = np.delete(ql_per_min2, [j for j in range(counts[i])])
            all_labels = np.delete(all_labels, [j for j in range(counts[i])])
            for kk in range(len(range_data1)):
                for label in range(K):
                    if labels[kk] == label:
                        regression_fit[i][label] += range_data1[kk]
                        regression_fit[i][label + K] += range_data2[kk]
            regression_fit[i] = regression_fit[i] / counts[i]

        Weights_or,loss = lg.liner_Regression(regression_fit, targets.reshape(n, 1), learningRate=learningRate,
                                         Loopnum=2000000, alpha=alpha)
        return Weights_or, regression_fit,loss


def fitting_gas_data(features_per_min, K, counts, targets, model='single_or', learningRate=0.0000005, alpha=1.0):
    """
    输入features_per_min为标定时间段每分钟的特征值，K为对标定数据分类的类别数，counts为各个标定时间段时长（包含分钟数）的列表,targets为标定
    时间段的真实流量标签，类型为array
    输出为各类别的拟合权重系数Weights_or和各标定时间段按类别的累计流量矩阵regression_fit
    """
    if model == 'single_or':
        n = len(counts)
        qg_train_gmodel_or = compute_general_Q(features_per_min.p_mean.values,
                                               features_per_min.t_mean.values,
                                               features_per_min.DP1_mean.values,
                                               features_per_min.DP2_mean.values,
                                               features_per_min.DP1_std.values,
                                               features_per_min.DP2_std.values,
                                               features_per_min.Den_G_mean.values,
                                               model=model)[1]
        qg_per_min = qg_train_gmodel_or
        all_labels = features_per_min['Label'].values

        regression_fit = np.zeros((n, K))
        for i in range(n):
            range_data = qg_per_min[:counts[i]]
            labels = all_labels[:counts[i]]
            qg_per_min = np.delete(qg_per_min, [j for j in range(counts[i])])
            all_labels = np.delete(all_labels, [j for j in range(counts[i])])
            for kk in range(len(range_data)):
                for label in range(K):
                    if labels[kk] == label:
                        regression_fit[i][label] += range_data[kk]
            regression_fit[i] = regression_fit[i] / counts[i]

        Weights_or,loss = lg.liner_Regression(regression_fit, targets.reshape(n, 1), learningRate=learningRate,
                                         Loopnum=2000000, alpha=alpha)
        return Weights_or, regression_fit,loss
    elif model == 'dual_or':
        n = len(counts)
        qg_train_gmodel_or1, qg_train_gmodel_or2 = compute_general_Q(features_per_min.p_mean.values,
                                                                     features_per_min.t_mean.values,
                                                                     features_per_min.DP1_mean.values,
                                                                     features_per_min.DP2_mean.values,
                                                                     features_per_min.DP1_std.values,
                                                                     features_per_min.DP2_std.values,
                                                                     features_per_min.Den_G_mean.values,
                                                                     model=model)[2:4]
        qg_per_min1, qg_per_min2 = qg_train_gmodel_or1, qg_train_gmodel_or2
        all_labels = features_per_min['Label'].values

        regression_fit = np.zeros((n, 2 * K))
        for i in range(n):
            range_data1 = qg_per_min1[:counts[i]]
            range_data2 = qg_per_min2[:counts[i]]
            labels = all_labels[:counts[i]]
            qg_per_min1 = np.delete(qg_per_min1, [j for j in range(counts[i])])
            qg_per_min2 = np.delete(qg_per_min2, [j for j in range(counts[i])])
            all_labels = np.delete(all_labels, [j for j in range(counts[i])])
            for kk in range(len(range_data1)):
                for label in range(K):
                    if labels[kk] == label:
                        regression_fit[i][label] += range_data1[kk]
                        regression_fit[i][label + K] += range_data2[kk]
            regression_fit[i] = regression_fit[i] / counts[i]

        Weights_or,loss = lg.liner_Regression(regression_fit, targets.reshape(n, 1), learningRate=learningRate,
                                         Loopnum=2000000, alpha=alpha)
        return Weights_or, regression_fit,loss


# def my_smooth(dst, span):
#     """
#     smooth函数python实现
#     """
#     src = dst.copy()
#     if span <= 0:
#         ex = Exception('输入非法区间值')
#         raise ex
#     # 如果输入的区间数为偶数，将区间值减一变为奇数
#     if (span % 2 == 0):
#         span -= 1
#
#     if (span > len(dst)):
#         ex = Exception('输入区间值大于列表长度')
#         raise ex
#
#     for i in range(len(dst)):
#         r = int((span - 1) / 2)
#
#         # 对两端元素不够区间窗口长度的，减小窗口半径
#         while (i - r < 0 or i + r >= len(dst)):
#             r -= 1
#
#         src[i] = sum(dst[i - r:i + r + 1]) / (2 * r + 1)
#     return src


# def compute_features(data):
#     """
#     data = pd.read_csv(filename,header=None,delimiter=' ',names=['P','DP1','DP2','Temp'],engine='python')
#     计算标定值时间范围内的特征值
#     data为该段时间范围内的原始信号（命名规则：['P','DP1','DP2','Temp']）
#     """
#     den_G = features_compute.compute_density(data)
#     data['Den_G'] = den_G
#     data['Difference'] = data.DP1 - data.DP2
#     # data['smooth'] = my_smooth(data.DP1, 10)
#     # data['ratio'] = data.DP1 / data.DP2
#     # data.drop(columns=['P', 'Temp'], inplace=True)
#
#     DP1_p, DP1_f = features_compute.compute_freq_power(data.DP1)
#     DP2_p, DP2_f = features_compute.compute_freq_power(data.DP2)
#     Den_p, Den_f = features_compute.compute_freq_power(data.Den_G)
#     Diff_p, Diff_f = features_compute.compute_freq_power(data.Difference)
#     # print(DP1_p)
#     # if 'features' not in locals():
#     features = pd.DataFrame(
#         {'p_mean': [data.P.mean()],
#          't_mean': [data.Temp.mean()],
#          'DP1_mean': [data.DP1.mean()],
#          'DP1_std': [data.DP1.std()],
#          'DP1_var': [data.DP1.var()],
#          # 'DP1_zrcs': [features_compute.compute_zrcs(data.DP1)],
#          # 'DP1_avgcs': [features_compute.compute_avgcs(data.DP1)],
#          'DP1_skew': [features_compute.compute_skew(data.DP1)],
#          'DP1_kurt': [features_compute.compute_kurt(data.DP1)],
#          'DP1_f1': [features_compute.compute_f1(DP1_p, DP1_f)],
#          'DP1_f2': [features_compute.compute_f2(DP1_p, DP1_f)],
#          # 'DP1_E': [features_compute.compute_Entropy(DP1_p)],
#          'DP1_SF': [features_compute.compute_SF(DP1_p)],
#          'DP2_mean': [data.DP2.mean()],
#          'DP2_std': [data.DP2.std()],
#          'DP2_var': [data.DP2.var()],
#          # 'DP2_zrcs': [features_compute.compute_zrcs(data.DP2)],
#          # 'DP2_avgcs': [features_compute.compute_avgcs(data.DP2)],
#          'DP2_skew': [features_compute.compute_skew(data.DP2)],
#          'DP2_kurt': [features_compute.compute_kurt(data.DP2)],
#          'DP2_f1': [features_compute.compute_f1(DP2_p, DP2_f)],
#          'DP2_f2': [features_compute.compute_f2(DP2_p, DP2_f)],
#          # 'DP2_E': [features_compute.compute_Entropy(DP2_p)],
#          'DP2_SF': [features_compute.compute_SF(DP2_p)],
#          'Den_mean': [data.Den_G.mean()],
#          'Den_std': [data.Den_G.std()],
#          'Den_var': [data.Den_G.var()],
#          # 'Den_zrcs': [features_compute.compute_zrcs(data.Den_G)],
#          'Den_skew': [features_compute.compute_skew(data.Den_G)],
#          'Den_kurt': [features_compute.compute_kurt(data.Den_G)],
#          'Den_f1': [features_compute.compute_f1(Den_p, Den_f)],
#          'Den_f2': [features_compute.compute_f2(Den_p, Den_f)],
#          # 'Den_E': [features_compute.compute_Entropy(Den_p)],
#          'Den_SF': [features_compute.compute_SF(Den_p)],
#          'Diff_mean': [data.Difference.mean()],
#          'Diff_std': [data.Difference.std()],
#          'Diff_var': [data.Difference.var()],
#          # 'Diff_zrcs': [features_compute.compute_zrcs(data.Difference)],
#          'Diff_skew': [features_compute.compute_skew(data.Difference)],
#          'Diff_kurt': [features_compute.compute_kurt(data.Difference)],
#          'Diff_f1': [features_compute.compute_f1(Diff_p, Diff_f)],
#          'Diff_f2': [features_compute.compute_f2(Diff_p, Diff_f)],
#          # 'Diff_E': [features_compute.compute_Entropy(Diff_p)],
#          'Diff_SF': [features_compute.compute_SF(Diff_p)],
#          'pulse': [abs(data.DP1 - data.smooth).mean()],
#          'ratio_mean': [data.ratio.mean()]})
#
#     # else:
#     #     newfeature = pd.DataFrame(
#     #         {'DP1_mean': [data.DP1.mean()], 'DP1_std': [data.DP1.std()], 'DP1_var': [data.DP1.var()], 'DP1_zrcs': \
#     #             [features_compute.compute_zrcs(data.DP1)], 'DP1_avgcs': [features_compute.compute_avgcs(data.DP1)],
#     #          'DP1_skew': [features_compute.compute_skew(data.DP1)],
#     #          'DP1_kurt': [features_compute.compute_kurt(data.DP1)], \
#     #          'DP1_f1': [features_compute.compute_f1(DP1_p, DP1_f)],
#     #          'DP1_f2': [features_compute.compute_f2(DP1_p, DP1_f)], 'DP1_E': [features_compute.compute_Entropy(DP1_p)], \
#     #          'DP1_SF': [features_compute.compute_SF(DP1_p)], 'DP2_mean': [data.DP2.mean()], 'DP2_std': [data.DP2.std()],
#     #          'DP2_var': \
#     #              [data.DP2.var()], 'DP2_zrcs': [features_compute.compute_zrcs(data.DP2)],
#     #          'DP2_avgcs': [features_compute.compute_avgcs(data.DP2)],
#     #          'DP2_skew': [features_compute.compute_skew(data.DP2)], 'DP2_kurt': \
#     #              [features_compute.compute_kurt(data.DP2)], 'DP2_f1': [features_compute.compute_f1(DP2_p, DP2_f)],
#     #          'DP2_f2': [features_compute.compute_f2(DP2_p, DP2_f)], 'DP2_E': \
#     #              [features_compute.compute_Entropy(DP2_p)], 'DP2_SF': [features_compute.compute_SF(DP2_p)],
#     #          'Den_mean': [data.Den_G.mean()], 'Den_std': \
#     #              [data.Den_G.std()], 'Den_var': [data.Den_G.var()],
#     #          'Den_zrcs': [features_compute.compute_zrcs(data.Den_G)], 'Den_skew': \
#     #              [features_compute.compute_skew(data.Den_G)], 'Den_kurt': [features_compute.compute_kurt(data.Den_G)],
#     #          'Den_f1': [features_compute.compute_f1(Den_p, Den_f)], \
#     #          'Den_f2': [features_compute.compute_f2(Den_p, Den_f)], 'Den_E': [features_compute.compute_Entropy(Den_p)],
#     #          'Den_SF': [features_compute.compute_SF(Den_p)], \
#     #          'Diff_mean': [data.Difference.mean()], 'Diff_std': [data.Difference.std()],
#     #          'Diff_var': [data.Difference.var()], \
#     #          'Diff_zrcs': [features_compute.compute_zrcs(data.Difference)],
#     #          'Diff_skew': [features_compute.compute_skew(data.Difference)], 'Diff_kurt': \
#     #              [features_compute.compute_kurt(data.Difference)],
#     #          'Diff_f1': [features_compute.compute_f1(Diff_p, Diff_f)],
#     #          'Diff_f2': [features_compute.compute_f2(Diff_p, Diff_f)], \
#     #          'Diff_E': [features_compute.compute_Entropy(Diff_p)], 'Diff_SF': [features_compute.compute_SF(Diff_p)], \
#     #          'pulse': [abs(data.DP1 - data.smooth).mean()], 'ratio_mean': [data.ratio.mean()]})
#     #     features = features.append(newfeature, ignore_index=True)
#     return features
#
#
# def compute_features_by_time(data_all, time_range=600):
#     """
#     data_all = pd.read_csv(filename,header=None,delimiter=' ',names=['P','DP1','DP2','Temp'],engine='python')
#     计算标定值时间范围内time_range时间段的平均特征值（默认为600帧，即一分钟）
#     data_all为该段时间范围内的原始信号（命名规则：['P','DP1','DP2','Temp']）
#     """
#     # den_G = features_compute.compute_density(data_all)
#     # data_all['Den_G'] = den_G
#     # data_all['Difference'] = data_all.DP1 - data_all.DP2
#     # data_all['smooth'] = my_smooth(data_all.DP1, 10)
#     # data_all['ratio'] = data_all.DP1 / data_all.DP2
#     # data_all.drop(columns=['P', 'Temp'], inplace=True)
#     counts = len(data_all) // time_range
#     for n in range(0, len(data_all) // time_range * time_range, time_range):
#         data = data_all.iloc[n:n + time_range, :]
#         DP1_p, DP1_f = features_compute.compute_freq_power(data.DP1)
#         DP2_p, DP2_f = features_compute.compute_freq_power(data.DP2)
#         Den_p, Den_f = features_compute.compute_freq_power(data.Den_G)
#         Diff_p, Diff_f = features_compute.compute_freq_power(data.Difference)
#         # print(DP1_p)
#         if 'features_split' not in locals():
#             features_split = pd.DataFrame(
#                 {'p_mean': [data.P.mean()],
#                  't_mean': [data.Temp.mean()],
#                  'DP1_mean': [data.DP1.mean()],
#                  'DP1_std': [data.DP1.std()],
#                  'DP1_var': [data.DP1.var()],
#                  # 'DP1_zrcs': [features_compute.compute_zrcs(data.DP1)],
#                  # 'DP1_avgcs': [features_compute.compute_avgcs(data.DP1)],
#                  'DP1_skew': [features_compute.compute_skew(data.DP1)],
#                  'DP1_kurt': [features_compute.compute_kurt(data.DP1)],
#                  'DP1_f1': [features_compute.compute_f1(DP1_p, DP1_f)],
#                  'DP1_f2': [features_compute.compute_f2(DP1_p, DP1_f)],
#                  # 'DP1_E': [features_compute.compute_Entropy(DP1_p)],
#                  'DP1_SF': [features_compute.compute_SF(DP1_p)],
#                  'DP2_mean': [data.DP2.mean()],
#                  'DP2_std': [data.DP2.std()],
#                  'DP2_var': [data.DP2.var()],
#                  # 'DP2_zrcs': [features_compute.compute_zrcs(data.DP2)],
#                  # 'DP2_avgcs': [features_compute.compute_avgcs(data.DP2)],
#                  'DP2_skew': [features_compute.compute_skew(data.DP2)],
#                  'DP2_kurt': [features_compute.compute_kurt(data.DP2)],
#                  'DP2_f1': [features_compute.compute_f1(DP2_p, DP2_f)],
#                  'DP2_f2': [features_compute.compute_f2(DP2_p, DP2_f)],
#                  # 'DP2_E': [features_compute.compute_Entropy(DP2_p)],
#                  'DP2_SF': [features_compute.compute_SF(DP2_p)],
#                  'Den_mean': [data.Den_G.mean()],
#                  'Den_std': [data.Den_G.std()],
#                  'Den_var': [data.Den_G.var()],
#                  # 'Den_zrcs': [features_compute.compute_zrcs(data.Den_G)],
#                  'Den_skew': [features_compute.compute_skew(data.Den_G)],
#                  'Den_kurt': [features_compute.compute_kurt(data.Den_G)],
#                  'Den_f1': [features_compute.compute_f1(Den_p, Den_f)],
#                  'Den_f2': [features_compute.compute_f2(Den_p, Den_f)],
#                  # 'Den_E': [features_compute.compute_Entropy(Den_p)],
#                  'Den_SF': [features_compute.compute_SF(Den_p)],
#                  'Diff_mean': [data.Difference.mean()],
#                  'Diff_std': [data.Difference.std()],
#                  'Diff_var': [data.Difference.var()],
#                  # 'Diff_zrcs': [features_compute.compute_zrcs(data.Difference)],
#                  'Diff_skew': [features_compute.compute_skew(data.Difference)],
#                  'Diff_kurt': [features_compute.compute_kurt(data.Difference)],
#                  'Diff_f1': [features_compute.compute_f1(Diff_p, Diff_f)],
#                  'Diff_f2': [features_compute.compute_f2(Diff_p, Diff_f)],
#                  # 'Diff_E': [features_compute.compute_Entropy(Diff_p)],
#                  'Diff_SF': [features_compute.compute_SF(Diff_p)],
#                  'pulse': [abs(data.DP1 - data.smooth).mean()],
#                  'ratio_mean': [data.ratio.mean()]})
#         else:
#             newfeature = pd.DataFrame(
#                 {'p_mean': [data.P.mean()],
#                  't_mean': [data.Temp.mean()],
#                  'DP1_mean': [data.DP1.mean()],
#                  'DP2_mean': [data.DP2.mean()],
#                  'Den_mean': [data.Den_G.mean()],
#                  'Diff_mean': [data.Difference.mean()],
#                  'Diff_std': [data.Difference.std()],
#                  'Diff_var': [data.Difference.var()],
#                  # 'Diff_zrcs': [features_compute.compute_zrcs(data.Difference)],
#                  'Diff_skew': [features_compute.compute_skew(data.Difference)],
#                  'Diff_kurt': [features_compute.compute_kurt(data.Difference)],
#                  'Diff_f1': [features_compute.compute_f1(Diff_p, Diff_f)],
#                  'Diff_f2': [features_compute.compute_f2(Diff_p, Diff_f)],
#                  # 'Diff_E': [features_compute.compute_Entropy(Diff_p)],
#                  'Diff_SF': [features_compute.compute_SF(Diff_p)],
#                  'pulse': [abs(data.DP1 - data.smooth).mean()],
#                  'ratio_mean': [data.ratio.mean()]})
#             features_split = features_split.append(newfeature, ignore_index=True)
#     return features_split, counts


# def feature_extraction(data):
#     """
#     输入标定时间段原始信号，输出分别为标定时间段特征值与标定时间段每分钟特征值
#     """
#     features_all = compute_features(data)
#     features_per_min = compute_features_by_time(data)
#     return features_all, features_per_min


# def my_feature_selection(features_all, targets, feature_num=2):
#     """
#     输入标定时间段的特征矩阵、标定值以及希望提取的相关特征个数n（默认为2）
#     返回一个长度为n的列表，列表中的元素为相关特征的索引值
#     """
#     from sklearn.feature_selection import f_regression, SelectKBest
#     from sklearn.preprocessing import StandardScaler
#     if feature_num == 'all':
#         index_ = [i for i in range(len(features_all.columns))]
#         return index_
#     else:
#         index_ = []
#         scaler = StandardScaler()
#         selector = SelectKBest(f_regression, feature_num)
#         features_sca = scaler.fit_transform(features_all)
#         selected_feature = selector.fit_transform(features_sca, targets)
#         for i in range(feature_num):
#             index = list(features_sca[0]).index(selected_feature[0][i])
#             index_.append(index)
#         return index_



