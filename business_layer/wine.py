import operator
import numpy as np
from numpy import genfromtxt
import copy
import warnings
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from pandas.core.frame import DataFrame
from sklearn import metrics
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from tkinter import _flatten

warnings.filterwarnings("ignore")

data = genfromtxt(r"C:\Users\17894\Desktop\wine.csv", delimiter=',')
# print(data)
num_data = data.shape[0]  # 样本数
# print(num_data)
num_attribute = data.shape[1] - 1  # 属性个数
data_no_label = data[:, :-1]  # 没有标签的数据的矩阵
lable = DataFrame(data).iloc[:, -1]
Alpha = 0.6
Beta = 0.3


def get_decision_class():  # 获得决策类
    label_num = []
    for i in range(num_data):  # 将所有数据的标签转为一个列表
        label_num.append(data[i, -1])
    # print(lable_list)
    label = list(set(label_num))  # 数据中的标签转为list方便后面处理
    d_class = [[] for i in range(len(label))]
    # print(lable)
    # print(d_class)
    for i in range(num_data):  # 将样本按照标签进行分类，返回相应的决策类
        for j in range(len(label)):
            if data[i, -1] == label[j]:
                d_class[j].append(i)
    # print(d_class)
    return d_class


def get_condition_class(data1):  # 获得相应的条件类
    c_class = [[] for i in range(num_data)]
    for i in range(num_data):
        for j in range(num_data):
            if np.all(data1[i] == data1[j]):  # 判断两个样本是否相等
                c_class[i].append(j)
        c_class[i].sort()
    temp = []
    for i in c_class:  # 去除重复的条件类
        i.sort()
        if i not in temp:
            temp.append(i)
    c_class = temp
    # print(c_class)
    return c_class


def max_inclusion(data1):  # 计算在相关条件属性的条件熵
    d_class = get_decision_class()  # 决策类
    c_class = get_condition_class(data1)  # 条件类
    max_include = []
    for i in range(len(c_class)):
        len_mixed = []
        for j in range(len(d_class)):
            mixed = list(set(c_class[i]).intersection(set(d_class[j])))
            len_mixed.append(len(mixed))
        max_len_mixed = max(len_mixed)
        # print(max_len_mixed)
        max_include.append(max_len_mixed / len(c_class[i]))
    # print(max_include)
    return max_include


def mutual_information(data1, col_index):  # 计算在相关条件属性集的互信息
    c_class = get_condition_class(data1)  # 条件类
    d_class = get_decision_class()  # 决策类
    temp_data = data1[:, col_index]  # 相应条件属性构成的决策表
    # print(temp_data)
    max_include = max_inclusion(temp_data)
    # print(max_include)
    H = 0  # 决策类的信息熵
    H_T = 0  # 条件熵
    for i in range(len(c_class)):
        for j in range(len(max_include)):
            H_T = H_T - max_include[j] * np.log(max_include[j]) * (len(c_class[i]) / num_data)
        #  a = a + max_include[j] * np.log(max_include[j])
        #  H_T = H_T + len(c_class[i]) / num_data * a
    for i in range(len(d_class)):
        H = H - (len(d_class[i]) / num_data) * np.log(len(d_class[i]) / num_data)
    # print(-H_T)
    I = H - H_T
    return I


All_MH = mutual_information(data_no_label, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


def get_view_reduction():  # 进行视角的约简，得到约简后的视角集合（三维的）
    view_list = []
    view_reduction = []
    view_MH = []
    temp_view = []
    view_index = [[[0],[0,1],[0,1,2]], [[3],[3,4],[3,4,5]], [[6],[6,7],[6,7,8]], [[9],[9,10],[9,10,11],[9,10,11,12]]]   # []视角，[[]]层次
    # print(compute_k_DMH(data_no_label, view_index[1]+view_reduction))
    # print(len(view_index))
    while (len(view_index)):  # 进行视角的选择
        for i in range(len(view_index)):
            # print(i)
            index = view_reduction + view_index[i]
            temp_index = list(_flatten(index))  # 将二维转换为一维进行视角的计算
            # print(temp_index)
            view_MH.append(mutual_information(data_no_label, temp_index))
        print(view_MH)

        min_view_MH = min(view_MH)
        min_view_index = view_index[view_MH.index(min_view_MH)]  # 找出最小的视角
        view_reduction.extend(min_view_index)  # 找出之后加入视角约简集合，之后继续进行视角的选择
        view_list.append(min_view_index)
        del view_index[view_MH.index(min_view_MH)]  # 删除已经加入red的集合
        view_MH = []
        if mutual_information(data_no_label, list(_flatten(view_reduction))) <= All_MH:
            break
        # print(view_reduction)
    # print(view_list)
    print("视角约简结果为:")
    print(view_list)
    return view_list


def get_level_reduction():  # 进行层次约简

    view_reduction_index = get_view_reduction()
    level_reduction_index = copy.deepcopy(view_reduction_index)
    view_red_MH = []  # 约简之后的每个视角的MH

    level_red = []  # 返回层次约简之后的结果

    for i in range(len(view_reduction_index)):  # 根据视角的数目进行循环
        level_red.append([])

        temp_i_view = view_reduction_index[i]  # 用来进行数据处理的第i个视角
        view_red_MH.append(mutual_information(data_no_label, list(_flatten(temp_i_view))))  # 获取每个视角的MH
        level_SIG = []  # 每个层次的重要度
        first_level = [[], []]
        # level_list = [[], []]   # [0]暂存层次的重要度，[1]暂存层次的标签
        for k in temp_i_view:  # 选择第一个层次加入（MH最小的）
            first_level[0].append(mutual_information(data_no_label, k))
            first_level[1].append(k)
        min_index = first_level[0].index(min(first_level[0]))
        level_red[i].append(first_level[1][min_index])
        b = len(temp_i_view)
        for m in range(b - 1):  # 选择层次重要度最大的加入
            level_list = [[], []]  # [0]暂存层次的重要度，[1]暂存层次的标签
            for l in temp_i_view:  # 删除已经加入约简集合中的属性
                if l in level_red[i]:
                    temp_i_view.remove(l)
            for j in temp_i_view:  # 计算层次重要度
                a = copy.deepcopy(level_red[i])
                # print(a)
                # print("1")
                # print(level_red[i])
                level_list[0].append(
                    mutual_information(data_no_label, level_red[i]) - mutual_information(data_no_label,
                                                                                           a.append(j)))  # 计算层次的重要度
                level_list[1].append(j)  # 将层次标签加入
            max_index = level_list[0].index(max(level_list[0]))  # 选择重要度最大的层次加入
            level_red[i].append(level_list[1][max_index])
            if mutual_information(data_no_label, list(_flatten(level_red[i]))) <= view_red_MH[i]:
                break

    level_red = [ele for ele in level_red if ele != []]  # 删除空列表
    print("层次约简结果为")
    print(level_red)

    return level_red


def get_attribute_reduction():  # 在层次约简之后的层次集合中进行属性的约简
    level_reduction_index = get_level_reduction()  # 获得层次约简的结果,三维的
    attribute_reduction_index = copy.deepcopy(level_reduction_index)  # 存储约简之后的结果
    attribute_red = []  # 返回约简之后的结果
    view_i_MH = []  # 每个视角的MH
    attribute_SIG = []
    for i in range(len(level_reduction_index)):
        attribute_red.append([])
        temp_attribute_red = []
        temp_i_view = list(_flatten(level_reduction_index[i]))  # 将用来计算的二维数据转换为一维,用来进行数据处理的第i个视角
        attribute_i_view = list(_flatten(attribute_reduction_index[i]))  # 将返回的约简结果转换为一维
        view_i_MH.append(mutual_information(data_no_label, temp_i_view))  # 计算每个视角的MH
        # print(view_i_MH[i])
        first_attribute = [[], []]
        # attribute_list = [[], []]  # [0]暂存属性的重要度，[1]暂存属性的标签
        for k in temp_i_view:  # 选择第一个属性加入（MH最小的）
            first_attribute[0].append(mutual_information(data_no_label, k))
            first_attribute[1].append(k)
        min_index = first_attribute[0].index(min(first_attribute[0]))
        temp_attribute_red.append(first_attribute[1][min_index])
        b = len(temp_i_view)
        for m in range(b - 1):  # 选择属性重要度最大的加入
            attribute_list = [[], []]  # [0]暂存属性的重要度，[1]暂存属性的标签
            for l in temp_i_view:  # 删除已经加入约简集合中的属性
                if l in temp_attribute_red:
                    temp_i_view.remove(l)
            for j in temp_i_view:  # 计算属性重要度
                a = copy.deepcopy(temp_attribute_red)
                attribute_list[0].append(
                    mutual_information(data_no_label, temp_attribute_red) - mutual_information(data_no_label,
                                                                                                 a.append(
                                                                                                     j)))  # 计算层次的重要度
                attribute_list[1].append(j)  # 将层次标签加入
            max_index = attribute_list[0].index(max(attribute_list[0]))  # 选择重要度最大的层次加入
            temp_attribute_red.append(attribute_list[1][max_index])
            if mutual_information(data_no_label, temp_attribute_red) <= view_i_MH[i]:
                break
        # print(temp_attribute_red)
        attribute_red.append(temp_attribute_red)
    attribute_red = [ele for ele in attribute_red if ele != []]  # 删除空列表
    # print(attribute_red)
    print("约简结果为：")
    print(attribute_red)
    return attribute_red


# get_decision_class(data1, num_data)
# get_condition_class(data_no_label, num_data)
# max_inclusion(data_no_label)
# max_entropy(data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17])
red = get_attribute_reduction()
for i in range(len(red)):
    red[i].insert(0, -1)


# print(red)

def acc(attr_data):
    temp_view_data = attr_data  # 约简之后的视角集合，二维
    view_index = []
    temp_lable = []  # 存储不同视角下预测的标签
    predict = []  # 融合之后的数据标签
    tree_predict = []  # 暂存视角下的预测结果
    for i in range(len(temp_view_data)):
        tree_predict.append([])
        view_index.append([])
        view_index[i] = temp_view_data[i]  # 获取视角index进行区分
        data_i = DataFrame(data[:, view_index[i]])  # 不同视角下的带有标签的数据
        train_data = data_i.iloc[:, 1:]  # 训练数据
        train_target = data_i.iloc[:, 0]  # 标签
        # clf = DecisionTreeClassifier()  # 获取决策树作为分类器
        # clf = KNeighborsClassifier()  # 获取KNN分类器
        # clf = svm.SVC(C = 0.8, kernel = 'rbf', gamma = 20)
        clf = AdaBoostClassifier()  # 获取AdaBoost分类模型
        a = cross_val_predict(clf, train_data, train_target, cv=10)  # 获取一个视角下的预测结果
        # print(type(a))
        tree_predict[i] = a.tolist()

    # print(tree_predict)
    for j in range(num_data):
        temp_lable.append([])
    # print(num_data)
    # print(tree_predict[0][1])

    for k in range(len(temp_view_data)):
        for j in range(num_data):
            temp_lable[j].append(tree_predict[k][j])
    # print(temp_lable)
    # print(temp_lable[0])
    # print(temp_lable[0][1])
    for j in range(num_data):
        predict.append([])
        predict[j] = max(set(temp_lable[j]), key=temp_lable[j].count)
    # print(predict)
    # print(lable)
    print("accuracy：")
    accuracyavg = accuracy_score(lable, predict)
    print(accuracyavg)


acccuracy = acc(red)
