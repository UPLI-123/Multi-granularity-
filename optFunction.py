import numpy as np
import copy
from tkinter import _flatten
from utils import getgroup, remove_list
from queue import Queue

#  todo 粒结构优化代码
def opt_Graular(df, view_index, ci_list, res_list, df2):
    Alpha = 0.6
    Beta = 0.3
    #  样本数
    num_data = len(df)
    #  属性个数
    num_attribute = len(df.columns) - 1
    #  获取标签信息
    data_no_label = np.mat(df.iloc[:, 0:num_attribute])
    #  获取属性信息
    label = np.mat(df.iloc[:, -1])
    # print(label)
    # print(num_attribute)
    # level_red
    view_reduction_index, level_red, attr_red = get_attribute_reduction(view_index, Alpha=Alpha, Beta=Beta, num_data=num_data, num_attribute=num_attribute,
                                  data_no_label=data_no_label, label=label, data=df)


    str_data = ""
    level_red = sorted(level_red)
    begin_list = []
    end_list = []
    for i, data in enumerate(level_red):  # 视图信息作为下标
        begin_list.append(0)
        end_list.append(len(data) - 1)
        for j, d in enumerate(data):  # 层次信息作为上标
            str_data = str_data + "<p>π<sub>{}</sub><sup>{}</sup>:{}</p>".format(i + 1, j + 1, getgroup(df2,d))
            pass
        pass
    str_data += "<hr/>"
    str_data += "<p>多视角多层次粒结构约简：</p>"
    q = Queue(maxsize=0)
    q.put(begin_list)
    res_list = []
    res_list.append([begin_list])
    while q.empty() is not None:
        #  记录 当前层所记录的数据条数
        q_size = q.qsize()
        if q_size == 0:
            break
        q_list = []  # 记录当前层的结果
        while q_size > 0:
            temp = q.get()
            for t in range(len(temp)):
                temp[t] = temp[t] + 1
                if temp[t] <= end_list[t]:
                    q_list.append(temp.copy())
                    q.put(temp.copy())
                elif temp == end_list:
                    q_list.append(end_list)
                    q = Queue(maxsize=0)
                    break
                    pass
                temp[t] = temp[t] - 1
                pass
            q_size = q_size - 1
            pass
        #  清除q_list中的重复元素
        q_list = remove_list(q_list)
        if len(q_list) != 0:
            res_list.append(q_list)
        pass
    print(res_list)
    #  将rss_list 调整格式显示到界面上
    for res in res_list:
        str_data += '<p>'
        for r in res:
            str_data += '('
            for i in range(len(r)):
                str_data = str_data + "π<sub>{}</sub><sup>{}</sup>".format(i + 1, r[i] + 1)
                if i == len(r) - 1:
                    str_data = str_data + ")"
                    pass
                else:
                    str_data = str_data + ","
                    pass
            str_data += '\t'
            pass
        str_data += '</p>'
        pass
    return str_data, level_red
    pass


#  进行属性约简
def get_attribute_reduction(view_index, **kwargs):
    view_res, level_reduction_index = get_level_reduction(view_index, kwargs)
    level_res = copy.deepcopy(level_reduction_index)
    attribute_reduction_index = copy.deepcopy(level_reduction_index)  # 存储约简之后的结果
    attribute_red = []  # 返回约简之后的结果
    view_i_MH = []  # 每个视角的MH
    data_no_label = kwargs['data_no_label']
    attribute_SIG = []
    for i in range(len(level_reduction_index)):
        attribute_red.append([])
        temp_attribute_red = []
        temp_i_view = list(_flatten(level_reduction_index[i]))  # 将用来计算的二维数据转换为一维,用来进行数据处理的第i个视角
        attribute_i_view = list(_flatten(attribute_reduction_index[i]))  # 将返回的约简结果转换为一维
        view_i_MH.append(mutual_information(data_no_label, temp_i_view, kwargs))  # 计算每个视角的MH
        # print(view_i_MH[i])
        first_attribute = [[], []]
        # attribute_list = [[], []]  # [0]暂存属性的重要度，[1]暂存属性的标签
        for k in temp_i_view:  # 选择第一个属性加入（MH最小的）
            first_attribute[0].append(mutual_information(data_no_label, k, kwargs))
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
                a.append(j)
                attribute_list[0].append(
                    mutual_information(data_no_label,
                                       a, kwargs) -
                    mutual_information(data_no_label, temp_attribute_red, kwargs))  # 计算层次的重要度
                attribute_list[1].append(j)  # 将层次标签加入
            max_index = attribute_list[0].index(max(attribute_list[0]))  # 选择重要度最大的层次加入
            temp_attribute_red.append(attribute_list[1][max_index])
            if mutual_information(data_no_label, temp_attribute_red, kwargs) >= view_i_MH[i]:
                break
        # print(temp_attribute_red)
        attribute_red.append(temp_attribute_red)
    attribute_red = [ele for ele in attribute_red if ele != []]  # 删除空列表
    # print(attribute_red)
    print("约简结果为：")
    print(attribute_red)
    return view_res, level_res, attribute_red
    pass


#  进行层次约简
def get_level_reduction(view_index, kwargs):
    view_reduction_index = get_view_reduction(view_index, kwargs)
    data_no_label = kwargs['data_no_label']
    level_reduction_index = copy.deepcopy(view_reduction_index)
    view_red_MH = []  # 约简之后的每个视角的MH
    level_red = []  # 返回层次约简之后的结果
    for i in range(len(view_reduction_index)):  # 根据视角的数目进行循环
        level_red.append([])

        temp_i_view = view_reduction_index[i]  # 用来进行数据处理的第i个视角
        view_red_MH.append(mutual_information(data_no_label, list(_flatten(temp_i_view)),kwargs))  # 获取每个视角的MH
        level_SIG = []  # 每个层次的重要度
        first_level = [[], []]
        # level_list = [[], []]   # [0]暂存层次的重要度，[1]暂存层次的标签
        for k in temp_i_view:  # 选择第一个层次加入（MH最小的）
            first_level[0].append(mutual_information(data_no_label, k,kwargs))
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
                a.append(j)
                level_list[0].append(mutual_information(data_no_label,list(_flatten(a)), kwargs) -
                    mutual_information(data_no_label, list(_flatten(level_red[i])), kwargs)) # 计算层次的重要度
                level_list[1].append(j)  # 将层次标签加入
            max_index = level_list[0].index(max(level_list[0]))  # 选择重要度最大的层次加入
            level_red[i].append(level_list[1][max_index])
            if mutual_information(data_no_label, list(_flatten(level_red[i])), kwargs) >= view_red_MH[i]:
                break

    level_red = [ele for ele in level_red if ele != []]  # 删除空列表
    print("层次约简结果为")
    print(level_red)
    # print(level_reduction_index)
    return level_reduction_index, level_red
    # return level_red
    pass


#  进行视图约简
def get_view_reduction(view_index, kwargs):
    view_list = []
    view_reduction = []
    view_MH = []
    data_no_label = kwargs['data_no_label']
    num_attribute = kwargs['num_attribute']
    All_MH = mutual_information(data_no_label, [i for i in range(num_attribute)], kwargs)
    print("全部视图的互信息:")
    print(All_MH)
    # 进行视角的选择
    while (len(view_index)):
        for i in range(len(view_index)):
            index = view_reduction + view_index[i]
            #  将二维转化为一维进行计算
            temp_index = list(_flatten(index))
            # print(temp_index)
            view_MH.append(mutual_information(data_no_label, temp_index, kwargs))
            pass
        print(view_MH)
        min_view_MH = min(view_MH)
        min_view_index = view_index[view_MH.index(min_view_MH)]  # 找出最小的视角
        view_reduction.extend(min_view_index)  # 找出之后加入视角约简集合，之后继续进行视角的选择
        view_list.append(min_view_index)
        del view_index[view_MH.index(min_view_MH)]  # 删除已经加入red的集合
        view_MH = []
        if mutual_information(data_no_label, list(_flatten(view_reduction)), kwargs) >= All_MH:
            break
        pass
    print("视角约简结果为:")
    print(view_list)
    return view_list
    pass

#  计算互信息
def mutual_information(data1, col_index, kwargs):  # 计算在相关条件属性集的互信息
    print(col_index)
    temp_data = data1[:, col_index]  # 相应条件属性构成的决策表
    c_class = get_condition_class(temp_data, kwargs)  # 条件类
    # print("条件类:")
    # print(c_class)
    d_class = get_decision_class(kwargs)  # 决策类
    # print("决策类:")
    # print(d_class)
    max_include, max_value = max_inclusion(temp_data, kwargs)
    # print(max_include)
    num_data = kwargs['num_data']
    H = 0  # 决策类的信息熵
    H_T = 0  # 条件熵
    for i in range(len(c_class)):
        H_T = H_T + max_include[i] * np.log(max_include[i])*max_value[i]

        #  a = a + max_include[j] * np.log(max_include[j])
        #  H_T = H_T + len(c_class[i]) / num_data * a
    for i in range(len(d_class)):
        H = H + (len(d_class[i]) / num_data) * np.log(len(d_class[i]) / num_data)
    # print(H_T)
    # print(H)
    I = H - H_T
    return I

def get_decision_class(kwargs):  # 获得决策类
    label_num = []
    num_data = kwargs['num_data']
    data = kwargs['data']
    data = np.mat(data)
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


def get_condition_class(data1, kwargs):  # 获得相应的条件类
    num_data = kwargs['num_data']
    c_class = [[] for i in range(num_data)]
    # print(data1)
    # print(data1[0,:])
    for i in range(num_data):
        for j in range(num_data):
            if np.all(data1[i,:] == data1[j,:]):  # 判断两个样本是否相等
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

def max_inclusion(data1, kwargs):  # 计算在相关条件属性的条件熵
    d_class = get_decision_class(kwargs)  # 决策类
    c_class = get_condition_class(data1, kwargs)  # 条件类
    num_data = kwargs['num_data']
    max_include = []
    max_value = []
    for i in range(len(c_class)):
        # len_mixed = []
        for j in range(len(d_class)):
            mixed = list(set(c_class[i]).intersection(set(d_class[j])))
            if len(mixed) != 0:
                max_include.append(len(mixed) / len(d_class[j]))
                max_value.append(len(d_class[j]) / num_data)
                break
                pass
            else:
                continue
                pass
    return max_include, max_value
