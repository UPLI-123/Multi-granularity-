# 用来存储一些通用的判断方法
import re
import itertools
from queue import Queue
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA  # 进行PCA 降维

#  判断是否符合输入的格式
def is_comma_separated_numbers(s):
    # print("sss")
    # print(s)
    # 正则表达式匹配以逗号分隔的数字
    pattern = r'^[\d]+(,[\d]+)*$'
    return re.match(pattern, s) is not None


# 粒结构构建方法（按照属性进行构造）
def build_Graular(view, ci, df2):
    '''
    :param view:  视图的划分信息
    :param ci:    层次的划分信息
    :param df2:   数据集
    :return:
    '''
    #  构建每一个视图中的属性信息
    #  view_list 是每个视图的属性信息
    view_list = []
    sum = 0
    for v in view:
        view_part = []
        for i in range(sum, sum + v):
            view_part.append(i)
            pass
        sum = sum + v
        view_list.append(view_part)
        pass
    #  构建层次信息
    #  c_list是构建成功的每个视图的层次信息
    ci_list = []
    for i, st in enumerate(ci):
        ci_part = []
        st_list = [int(s) for s in st.split(',')]
        temp = 0
        for j in st_list:
            ci_part.append(view_list[i][temp:j+temp])
            temp = temp + j
            pass
        ci_list.append(ci_part)
        pass
    print(ci_list)
    #  进行层次信息的扩展
    ci_b_list = []
    for ci in ci_list:
        temp_list = []
        temp_view_list = []
        for c in ci:
            temp_list = temp_list + c
            temp_view_list.append(temp_list.copy())
            pass
        ci_b_list.append(temp_view_list)
        pass
    # print(ci_b_list)
    #  构建多层次 多粒度结构 （求笛卡尔积）
    result = list(itertools.product(*ci_b_list))
    # result = r"10<sup>2</sup>"
    result = [list(r) for r in result]
    print(result)
    str_data = ""
    # for idx, data in enumerate(data_info):
    #     str_data =  str_data + "第{}项:".format(idx) + str(data) + "\n"
    #     pass
    #  将每个信息进行定义
    begin_list = []
    end_list = []
    for i, data in enumerate(ci_b_list):  # 视图信息作为下标
        begin_list.append(0)
        end_list.append(len(data) - 1)
        for j, d in enumerate(data):  # 层次信息作为上标
            str_data = str_data + "<p>π<sub>{}</sub><sup>{}</sup>:{}</p>".format(i + 1, j + 1, getgroup(df2,d))
            pass
        pass
    str_data += "<hr/>"
    # 来显示格结构，第i层格结构 为 i-1 层格结构加1
    # 根据视图信息来初始化值
    # 利用树的层次遍历来实现该功能
    #  创建队列 Queue
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
                str_data = str_data + "π<sub>{}</sub><sup>{}</sup>".format(i+1, r[i]+1)
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
    return str_data, result, ci_b_list, res_list, ci_list
    pass


#  清除list中重复的元素
def remove_list(q_list):
    l = len(q_list)
    res_list = []
    for q in q_list:
        if q not in res_list:
            res_list.append(q)
    return res_list

    pass


# 粒结构构建方法（按照属性值进行构造）
def build_Graular_Value(list_view, df, df2):
    num_data = len(df)
    #  属性个数
    num_attribute = len(df.columns) - 1
    #  获取标签信息
    data_no_label = np.mat(df.iloc[:, 0:num_attribute])
    new_data_no_lable = data_no_label.copy()  # 用来存储转换后的结构，进行拷贝防止计算过程中的干扰
    ci_sum = 0
    count = 0
    #  将连续值转为离散值
    for view in list_view:
        for ci_list in view:
            ci_sum += len(ci_list[0]) / 2
            for ci in ci_list:
                len1 = len(ci) / 2
                begin = int(ci_sum) - int(len1)
                for i in range(begin, int(ci_sum)):
                    for k in range(num_data):
                        if data_no_label[k, i] >= ci[0 + (i - begin) * 2] and data_no_label[k, i] <= ci[
                            1 + (i - begin) * 2]:
                            new_data_no_lable[k, i] = count
                            pass
                        pass
                    count = count + 1
                    pass
                pass
            pass
        pass

    level_red = []
    temp_index = 0
    #  存储新的视图信息
    #  将list_view 进行转化
    for view in list_view:
        level_red_temp = []
        end_index =temp_index
        for ci_list in view:
            temp_ci_list =[]
            end_index = end_index + int(len(ci_list[0]) / 2)
            for index in range(temp_index, end_index):
                temp_ci_list.append(index)
                pass
            temp_index = end_index
            level_red_temp.append(temp_ci_list)
            pass
        level_red.append(level_red_temp)
        pass
    #  对层次进行 深度的扩充
    ci_e_list = [] #  ci_e_list 层次扩充的存储
    for v in  level_red:
        v_list = []
        temp_v = []
        for value in v:
            temp_v+=value
            v_list.append(temp_v.copy())
            pass
        ci_e_list.append(v_list.copy())
        pass
    str_ans = ""
    begin_list = []
    end_list = []
    for i, data in enumerate(ci_e_list):  # 视图信息作为下标
        begin_list.append(0)
        end_list.append(len(data) - 1)
        for j, d in enumerate(data):  # 层次信息作为上标
            str_ans = str_ans + "<p>π<sub>{}</sub><sup>{}</sup>:{}</p>".format(i + 1, j + 1,
                                                                                 getgroup(new_data_no_lable, d))
            pass
        pass
    str_ans += "<hr/>"
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
        str_ans += '<p>'
        for r in res:
            str_ans += '('
            for i in range(len(r)):
                str_ans = str_ans + "π<sub>{}</sub><sup>{}</sup>".format(i + 1, r[i] + 1)
                if i == len(r) - 1:
                    str_ans = str_ans + ")"
                    pass
                else:
                    str_ans = str_ans + ","
                    pass
            str_ans += '\t'
            pass
        str_ans += '</p>'
        pass

    return str_ans, [], list_view, res_list, ci_e_list
    pass

# 按照属性值 构造层次信息
def build_Ci_Value(ci_list, df2, ci_sum):
    new_ci_list = []
    for ci in ci_list:
        new_ci = []
        for d in range(len(df2)):
            if isADD(df2, ci, ci_sum, d):
                new_ci.append(d)
            pass
        new_ci_list.append(new_ci)
        pass
    return new_ci_list
    pass


#  用来判断是否符合添加要求
def isADD(df2, ci, ci_sum, d):
    len1 = len(ci) / 2
    #  开始位置
    begin = ci_sum - int(len1)
    # print(begin)
    # if begin > 5:
    #     print(111111)
    #     pass
    for i in range(begin, ci_sum):
        if df2[d,i] < ci[0+(i-begin)*2]:
            return False
        if df2[d,i] > ci[1+(i-begin)*2]:
            return False
        pass
    return True
    pass

def getgroup(df2, d):
    '''
    :param df2: 数据集矩阵
    :param d: 需要进行判断的属性值
    :return: 返回划分之后的属性分组
    '''
    group_list = []
    tag = []
    # 初始化一个标志
    for i in range(len(df2)):
        tag.append(False) # 判断一下当前样本是否已经被分组了
        pass

    for i in range(len(df2)):
        if tag[i] == True: # 当前对象已经被成功分组了,无需考虑后面的分组了
            continue
            pass
        else:   # 当前对象没有考虑分组情况，因此需要创建一个组来进行存储
            temp_group = []
            temp_group.append(i)
            tag[i] = True
            pass
        for j in range(i+1, len(df2)):
            flag = True  # 用来判断对象i和j是否属于同一个组
            for atr in d:  # 使用属性值来进行判断
                if df2[i,atr]!=df2[j,atr]:
                    flag = False
                    break
                pass
            if flag:
                temp_group.append(j)
                tag[j] = True
                pass
            pass
        group_list.append(temp_group.copy())
    return group_list
    pass

def showFinal(level_red, best, df, methods, tag):
    '''
    :param level_red: 层次结构
    :param best: 最佳的种群
    :methods: 多模态融合算法的导入
    :tag: 判断一下是否为属性聚合
    :return:
    '''
    #  求出df2
    fn = len(df.columns)  # 染色体长度
    obn = len(df)  # 数据集样本个数
    # print(fn, obn)
    df2 = df.iloc[0:obn, 0:fn]
    df2 = np.mat(df2)

    str_data = ""
    level_red = sorted(level_red)
    begin_list = []
    end_list = []
    for i, data in enumerate(level_red):  # 视图信息作为下标
        begin_list.append(0)
        end_list.append(len(data) - 1)
        for j, d in enumerate(data):  # 层次信息作为上标
            str_data = str_data + "<p>π<sub>{}</sub><sup>{}</sup>:{}</p>".format(i + 1, j + 1, getgroup(df2, d))
            pass
        pass
    str_data += "<hr/>"
    #  显示问题求解的结果
    str_data += "<p>问题求解的结果：</p>"
    str_data+='<p>('
    for i, j in enumerate(best):
        str_data = str_data + "π<sub>{}</sub><sup>{}</sup>".format(i + 1, int(j))
        if i == len(best) -1:
            str_data = str_data +")</p>"
            pass
        else:
            str_data = str_data+","
            pass
        pass
    str_data += "<hr/>"
    #  显示问题求解的结果
    str_data += "<p>多模态数据融合结构分析：</p>"
    if tag == 1 :  # 属性拼接
        #  将best 转化为属性的集合
        red = []
        for i, level in enumerate(level_red):
            red = red + level[int(best[i]-1)]
            pass
        red.insert(0,-1)
        temp = []
        temp.append(red)
        acccuracy = acc(temp,df2, methods)
        str_data += "accuracy: {}%".format(acccuracy*100)
        return str_data
    elif tag == 2:  # 结果聚合
        count_sum = 0  # 记录每次精度的总和
        for i, level in enumerate(level_red):
            red = level[int(best[i]-1)]
            red.insert(0,-1)
            temp = []
            temp.append(red)
            count_sum += acc(temp, df2, methods)
            pass
        acccuracy = count_sum / len(level_red)
        str_data += "accuracy: {}%".format(acccuracy * 100)
        return str_data
        pass
    else:  # 特征融合
        # 1. 构建输入的视图  将三维向量转化为二维向量
        temp = []
        n_components = 10
        for i, level in enumerate(level_red):
            red = level[int(best[i] - 1)]
            n_components = min(n_components, len(red))
            red.insert(0, -1)
            temp.append(red)
            pass
        #  降维的目标维度为最小值
        acccuracy = acc_with_pca(temp, df2, methods, n_components=n_components)
        str_data += "accuracy: {}%".format(acccuracy * 100)
        return str_data
        pass
    pass

# 求精度
def acc(attr_data, df2, methods):
    temp_view_data = attr_data  # 约简之后的视角集合，二维
    view_index = []
    temp_lable = []  # 存储不同视角下预测的标签
    predict = []  # 融合之后的数据标签
    tree_predict = []  # 暂存视角下的预测结果
    for i in range(len(temp_view_data)):
        tree_predict.append([])
        view_index.append([])
        view_index[i] = temp_view_data[i]  # 获取视角index进行区分
        data_i = df2[:, view_index[i]] # 不同视角下的带有标签的数据
        train_data = data_i[:, 1:]  # 训练数据
        train_target = data_i[:, 0]  # 标签



        if methods == 'DecisionTree':
            clf = DecisionTreeClassifier()  # 获取决策树作为分类器
            pass
        elif methods == 'AdaBoost':
            clf = AdaBoostClassifier()  # 获取AdaBoost分类模型
            pass
        elif methods == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=2)  # 获取KNN分类器
            pass
        elif methods == 'SVC':
            clf = svm.SVC(C = 0.8, kernel = 'rbf', gamma = 20)
            pass
        a = cross_val_predict(clf, train_data, train_target, cv=2)  # 获取一个视角下的预测结果
        # print(type(a))
        tree_predict[i] = a.tolist()

    num_data = len(df2)
    # print(tree_predict)
    for j in range(num_data):
        temp_lable.append([])
    # print(num_data)
    # print(tree_predict[0][1])
    lable = df2[:,-1]
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
    return accuracyavg
    pass

# 求精度--特征融合
def acc_with_pca(attr_data, df2, methods, n_components=2):
    temp_view_data = attr_data  # 约简之后的视角集合，二维
    view_index = []  # 存储视角的索引
    temp_label = []  # 存储不同视角下预测的标签
    predict = []  # 融合之后的数据标签
    tree_predict = []  # 暂存视角下的预测结果

    for i in range(len(temp_view_data)):
        tree_predict.append([])
        view_index.append([])
        view_index[i] = temp_view_data[i]  # 获取视角index进行区分
        data_i = df2[:, view_index[i]]  # 不同视角下的带有标签的数据
        train_data = data_i[:, 1:]  # 训练数据
        train_target = data_i[:, 0]  # 标签

        # 使用 PCA 降维
        pca = PCA(n_components=n_components)
        train_data_pca = pca.fit_transform(train_data)

        if methods == 'DecisionTree':
            clf = DecisionTreeClassifier()  # 获取决策树作为分类器
        elif methods == 'AdaBoost':
            clf = AdaBoostClassifier()  # 获取AdaBoost分类模型
        elif methods == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=2)  # 获取KNN分类器
        elif methods == 'SVC':
            clf = svm.SVC(C=0.8, kernel='rbf', gamma=20)  # 获取SVC分类器
        else:
            raise ValueError(f"Unsupported method: {methods}")

        # 使用交叉验证进行预测
        a = cross_val_predict(clf, train_data_pca, train_target, cv=2)
        tree_predict[i] = a.tolist()

    num_data = len(df2)
    for j in range(num_data):
        temp_label.append([])

    lable = df2[:, -1]
    for k in range(len(temp_view_data)):
        for j in range(num_data):
            temp_label[j].append(tree_predict[k][j])

    # 进行融合，使用投票机制
    for j in range(num_data):
        predict.append(max(set(temp_label[j]), key=temp_label[j].count))

    print("accuracy：")
    accuracyavg = accuracy_score(lable, predict)
    print(accuracyavg)
    return accuracyavg