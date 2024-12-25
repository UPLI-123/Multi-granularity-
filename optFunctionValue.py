import numpy as np
from optFunction import get_attribute_reduction
from utils import getgroup, remove_list
from queue import Queue
import pandas as pd


#  按照属性值来进行优化
def opt_Graular_Value(df, view_index, list_view, res_list, view_atr):

    # print(df)
    # print(type(df))
    # print(type(np.mat(df)))
    Alpha = 0.6
    Beta = 0.3
    #  样本数
    num_data = len(df)
    #  属性个数
    num_attribute = len(df.columns) - 1
    #  获取标签信息
    new_data_no_label = np.mat(df.iloc[:, 0:num_attribute])
    data_no_label = new_data_no_label.copy()
    #  获取属性信息f
    label = np.mat(df.iloc[:, -1])
    # print("-----------")
    # print(view_index)
    # print(view_atr)
    #  res_list  和 view_atr 来进行将样本转化为属性
    atr_view_list = []
    for res in res_list:
        for r in res:
            atr_view = []
            for i in range(len(r)):
                temp = view_atr[i]
                atr_view.append(temp[r[i]])
                pass
            atr_view_list.append(atr_view)
            pass

        pass
    # print("new demo:")
    # print(atr_view_list)
    # print(ci_list)
    # print(res_list)
    # print(data_no_label)
    # 将data_no_lablel 进行变化 变成离散值的情况
    ci_sum = 0
    count = 0
    # 将连续值情况转化为离散值的情况
    for view in list_view:
        for ci_list in view:
            ci_sum += len(ci_list[0]) / 2
            for ci in ci_list:
                len1 = len(ci) / 2
                begin = int(ci_sum) - int(len1)
                for i in range(begin, int(ci_sum)):
                    for k in range(num_data):
                        if new_data_no_label[k, i] >= ci[0+(i-begin)*2] and new_data_no_label[k, i] <= ci[1+(i-begin)*2]:
                            data_no_label[k, i] = count
                            pass
                        pass
                    count = count + 1
                    pass
                pass
            pass
        pass
    #  获得新的df
    # 1. 合并data_no_label 和 label
    # 1.1  获取条件属性
    columns = [f'Attr{i + 1}' for i in range(len(df.columns)-1)]
    data_no_label_df = pd.DataFrame(data_no_label, columns=columns)
    # data_no_label_numpy = data_no_label.to_numpy()
    # data_no_label_contiguous = np.ascontiguousarray(data_no_label_numpy)
    # data_no_label_matrix = np.mat(data_no_label_contiguous)
    # 1.2  获取判断属性
    label_df = pd.DataFrame(label.T, columns=['label'])
    # label_numpy = label.to_numpy().reshape(-1,1)
    # label_contiguous = np.ascontiguousarray(label_numpy)
    # label_matrix = np.mat(label_contiguous)
    # 2. 准备合并后的列名
    # 3. 生成新的new_df
    new_df = pd.concat([data_no_label_df, label_df], axis=1)
    # print(type(df))
    # print(type(new_df))

    # print(data_no_label)
    # print(res_list)
    #  进行优化算法
    #  进行视角优化
    view_reduction_index, level_red, attr_red = get_attribute_reduction(view_index=view_atr, Alpha=Alpha,
                                                                        Beta=Beta, num_data=num_data,
                                                                        num_attribute=num_attribute,
                                                                        data_no_label=data_no_label, label=label,
                                                                 data=new_df)
    level_red = sorted(level_red)
    str_data = ""
    begin_list = []
    end_list = []
    for i, data in enumerate(level_red):  # 视图信息作为下标
        begin_list.append(0)
        end_list.append(len(data) - 1)
        for j, d in enumerate(data):  # 层次信息作为上标
            str_data = str_data + "<p>π<sub>{}</sub><sup>{}</sup>:{}</p>".format(i + 1, j + 1, getgroup(data_no_label, d))
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
    return str_data, view_reduction_index, new_df
    pass


