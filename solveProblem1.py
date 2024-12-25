import numpy as np
import copy
import collections
import multiprocessing
import random
import time
import matplotlib.pyplot as plt

#  用与实现问题求解层的逻辑打地面
def solveByAttribute1(df, view_index_list):
    '''
    按照属性值划分，求解问题
    @df: 读取的数据集
    @viewnum: 划分视图的数量
    @view_index: 输入的格结构
    :return:
    '''
    #  产生新的df(因为视图的约简导致视图的属性减少)

    # 初始化数据
    fn = len(df.columns) - 1  # 染色体长度
    obn = len(df)  # 数据集样本个数
    df1 = df.iloc[0:obn, 0:fn]  # 包含判断属性的样本
    df2 = np.mat(df1)
    dci1 = df.iloc[0:obn, fn:fn + 1]  # 包含决策属性的样本
    dci2 = np.mat(dci1)
    viewnum = len(view_index_list)
    #  为了将决策的属性的操作简单一下
    dci = np.zeros(obn)
    for b in range(obn):
        dci[b] = dci2[b, 0]
        pass
    view_index = view_index_list

    trun = 1  # 需要执行的轮次
    #  todo 默认只有6个判断属性
    listdci1 = []
    listdci2 = []
    listdci3 = []
    listdci4 = []
    listdci5 = []
    listdci6 = []
    for i in range(obn):
        if dci[i] == 0:
            listdci1.append(i)
        if dci[i] == 1:
            listdci2.append(i)
        if dci[i] == 2:
            listdci3.append(i)
        if dci[i] == 3:
            listdci4.append(i)
        if dci[i] == 4:
            listdci5.append(i)
        if dci[i] == 5:
            listdci6.append(i)
            pass
        pass
    # 6个判断属性
    listdci1 = set(listdci1)
    listdci2 = set(listdci2)
    listdci3 = set(listdci3)
    listdci4 = set(listdci4)
    listdci5 = set(listdci5)
    listdci6 = set(listdci6)
    # 其他的一些初始化属性
    pc1 = 0.8
    pc2 = 0.2
    pm = 0.1  # pc为变异的概率
    t2 = 5  # 遗传算法迭代的次数
    t1 = 40
    n = 20  # 种群的个体数,要求大于20以保证具有随机性

    temp_yuzhi = np.zeros(5)
    locality_yuzhi_fuben = 0.4
    # 其他方法对比
    wulidu = np.zeros(5)
    wuweizhi = np.zeros(5)
    satisfy = np.zeros(5)
    satisfycount = np.zeros(5)
    best_poscount = np.zeros(5)
    for num_1 in range(5):
        locality_yuzhi=0
        locality_yuzhi=locality_yuzhi_fuben+0.2*num_1
        print(locality_yuzhi)
        best_fitness=np.zeros(trun)
        best_pos=np.zeros(trun)
        pospercent=np.zeros(trun)
        superfine = np.zeros(viewnum)  # superfine 用来记录当前视图下的深度
        for i in range(viewnum):
            superfine[i] = len(view_index[i])
            pass
        superfine_pos = Jd1(superfine,
                            obn, fn, df2, view_index,
                            listdci1, listdci2, listdci3,
                            listdci4, listdci5, listdci6)
        print("最细粒度的正域划分率为%f" % superfine_pos)
        cengcicount = 0
        choice = np.zeros(viewnum)
        pop = copy.deepcopy(superfine)
        # 从特征向量x中提取出相应的特征
        zhenyucheck = np.zeros(obn)
        for i1 in range(viewnum):
            count = 0
            dcicount = np.zeros(obn)
            p1 = pop[i1]
            p1 = p1.astype(int)
            if p1 <= 1:
                choice[i1] = 1  # 如果当前视角下最细粒层为1层，该视角不进行筛选，直接将最终结果对应的视角层数置为1
            else:
                while p1 >= 1:
                    obndci = np.zeros(obn)  # obndci用来记录个体所在等价类的类标签数量
                    Feature = {}  # 字典Feature用来存 x选择的是哪d个特征
                    k = 0
                    for i2 in range(p1):
                        print(view_index[i1][i2])
                        Feature[k] = view_index[i1][i2]
                        k = k + 1
                        pass
                    df3 = np.zeros((obn, 1))
                    df3_index = []  # 用来记录转化后的属性所属的位置
                    df3_count = 0
                    for l in range(k):
                        # if Feature[l]!=0:
                        df3_index_temp = []
                        p = Feature[l]
                        ll = len(p)
                        for lll in range(ll):
                            df3_index_temp.append(df3_count)
                            df3_count = df3_count + 1
                            pass
                        df3_index.append(df3_index_temp)
                        q = df2[:, p]
                        q = q.reshape(obn, ll)  # 在不改变数据内容的情况下，改变一个数组的格式
                        df3 = np.append(df3, q, axis=1)
                    df3 = np.delete(df3, 0, axis=1)


                    #  存储所有等价类
                    listo = []  # 存储所有等价类
                    listnum = 0
                    save3 = np.ones(obn)
                    for i in range(obn):
                        lists = []
                        judge = 1
                        judge2 = 1
                        # f = 0
                        if save3[i] == 0:
                            continue
                        # save2[f] = i
                        lists.append(i)
                        for j in range(i + 1, obn, 1):
                            if save3[j] == 0:
                                continue
                                pass
                            judge = 1
                            for r in range(k):
                                t = df3_index[r]
                                for tt in t:
                                    if df3[i, tt] != df3[j, tt]:
                                        judge = 0
                                        break
                                        pass
                                    pass
                            if judge == 1:
                                # f = f + 1
                                # save2[f] = j
                                save3[j] = 0
                                lists.append(j)
                        listo.append(lists)
                        listnum = listnum + 1
                    for g in range(listnum):  # 分别遍历每一个等价类
                        dcinum = 0
                        dcilist = []
                        se = set(listo[g])
                        le = len(listo[g])  # le记录当前等价类中的对象个数
                        for u in range(le):
                            if u == 0:
                                dcilist.append(dci2[listo[g][u]])  # 如果当前对象是等价类中的第一个对象，就将它的决策属性值加入到dcilist中去
                                dcinum = dcinum + 1
                            else:
                                check = True
                                for v in range(dcinum):  # 将当前对象的决策属性值与之前已经保存到dcilist中的决策属性值相比较
                                    if dci2[listo[g][u]] == dcilist[v]:
                                        check = False
                                        break
                                if check == True:  # 如果当前对象的决策属性值与之前已经保存到dcilist中的决策属性值都不相同就将当前对象的决策属性值加入到dcilist中去
                                    dcilist.append(dci2[listo[g][u]])
                                    dcinum = dcinum + 1
                                    pass
                                pass
                            pass
                        for v in range(le):
                            obndci[listo[g][v]] = dcinum
                            pass
                        pass
                    check = True
                    checkobn = np.ones(obn)
                    locality_yuzhi_count = obn
                    for g1 in range(obn):
                        if dcicount[g1] == 0:
                            dcicount = copy.deepcopy(obndci)
                        if dcicount[g1] != obndci[g1] and checkobn[g1]==1:  #检查在当前层次组合下，个体所在等价类的类标签数量是否变化
                            # choice[i1]=p1+1
                            for gg in range(listnum):
                                if g1 in listo[gg]:
                                    le=len(listo[gg])
                                    for ggg in range(le):
                                        checkobn[listo[gg][ggg]]=0
                                    locality_yuzhi_count = locality_yuzhi_count - le #将不满足广义决策协调的个体数量从总数量中减去
                                    break
                            '''check = False
                            break'''
                            pass
                        pass
                    if  (locality_yuzhi_count/obn)<locality_yuzhi:
                        choice[i1] = p1 + 1
                        if p1==superfine[i1]:
                            choice[i1]=p1
                        break
                    else:
                        # dcicount=copy.deepcopy(obndci)
                        p1 = p1 - 1
                    if p1 == 0:
                        choice[i1] = 1
                    pass
                pass
            pass
        for g in range(viewnum):
            cengcicount = cengcicount + choice[g]
        wu_pos=Jd1(choice,
                   obn, fn, df2, view_index,
                   listdci1, listdci2, listdci3,
                   listdci4, listdci5, listdci6)
        print("Wu方法正域划分率为：%.4f" %wu_pos)
        temp_yuzhi[num_1]=wu_pos/superfine_pos
        print("与最细粒层的正域划分率之比为：%.4f"%temp_yuzhi[num_1])
        print(choice)
        '''wuweizhi=np.zeros(turn)
        for g in range(turn):
            wuweizhi[g]=cengcicount/fn'''
        wuweizhi[num_1] = cengcicount / fn
        wulidu[num_1] = wuweizhi[num_1] * fn
        print("粒度为：%.4f" % wulidu[num_1])
        for i in range(trun):
            new_view_index = []
            #  初始化种群
            population = start(viewnum, n, view_index)
            new_view_index.append(population)
            # new_view_index.append(population)
            # 调用并行化遗传算法
            best_people, best_pos[i], satisfy = genetic_algorithm_parallel(trun, new_view_index, wu_pos, superfine_pos,
                                                                           superfine,
                                                                           listdci1, listdci2, listdci3,
                                                                           listdci4, listdci5, listdci6,
                                                                           pc1, pc2, pm,
                                                                           t2, t1, n,
                                                                           obn, fn, df2, view_index)

            print("并行算法下的最粗满意粒度层：", best_people)
            #     population = copy.deepcopy(start())
            #     best_people, best_pos[i], satisfy = GA(population, wu_pos, superfine_pos, superfine)
            satisfycount[num_1] = satisfycount[num_1] + satisfy
            print("满足阈值的最粗粒层为：%.4f" % (satisfy))
            print("第%d次选择最好粒层组合为：" % i, best_people)
            best_poscount[num_1] = best_poscount[num_1] + best_pos[i]
            print("第%d次选出的问题求解层正域划分率为：" % i, best_pos[i])
            print("误差为%.4f", (best_pos[i] - wu_pos) / wu_pos)

            pass
        pass
    return best_people
    pass


def Jd1(x, obn, fn, df2, view_index
        , listdci1, listdci2, listdci3
        , listdci4, listdci5, listdci6
        ):
    pop=copy.deepcopy(x)
    # 从特征向量x中提取出相应的特征
    zhenyucheck=np.zeros(obn)
    # kcount=0
    total = 0  # 用来记录属性数
    for i1 in range(len(x)):
        Feature = {}  # 字典Feature用来存 x选择的是哪d个特征
        k = 0
        p1=pop[i1]
        p1=p1.astype(int)
        for i2 in range(p1):
            print(view_index[i1][i2])
            Feature[k] = view_index[i1][i2]
            k=k+1
            pass
        df3 = np.zeros((obn, 1))
        df3_index = []  # 用来记录转化后的属性所属的位置
        df3_count = 0
        for l in range(k):
            # if Feature[l]!=0:
            df3_index_temp = []
            p = Feature[l]
            ll = len(p)
            for lll in range(ll):
                df3_index_temp.append(df3_count)
                df3_count = df3_count+1
                pass
            df3_index.append(df3_index_temp)
            q = df2[:, p]
            q = q.reshape(obn, ll)  # 在不改变数据内容的情况下，改变一个数组的格式
            df3 = np.append(df3, q, axis=1)
        df3 = np.delete(df3, 0, axis=1)
        listo=[]
        listnum=0
        save3=np.ones(obn)
        for i in range(obn):
            lists = []
            judge = 1
            judge2 = 1
            #f = 0
            if save3[i]==0:
                continue
            #save2[f] = i
            lists.append(i)
            for j in range(i + 1, obn, 1):
                if save3[j]==0:
                    continue
                    pass
                judge = 1
                for r in range(k):
                    t = df3_index[r]
                    for tt in t:
                        if df3[i, tt] != df3[j, tt]:
                            judge = 0
                            break
                            pass
                        pass
                if judge == 1:
                    #f = f + 1
                    #save2[f] = j
                    save3[j]=0
                    lists.append(j)
            if len(lists)!=0:
                listo.append(lists)
                listnum = listnum + 1
        for g in range(listnum):
            se = set(listo[g])
            le = len(listo[g])
            if se.issubset(listdci1) is True:  # issubset判断集合的所有元素是否都包含在指定集合中
                for ll in range(le):
                    zhenyucheck[listo[g][ll]] = 1
                continue
            if se.issubset(listdci2) is True:
                for ll in range(le):
                    zhenyucheck[listo[g][ll]] = 1
                continue
            if se.issubset(listdci3) is True:
                for ll in range(le):
                    zhenyucheck[listo[g][ll]] = 1
                continue
            if se.issubset(listdci4) is True:
                for ll in range(le):
                    zhenyucheck[listo[g][ll]] = 1
                continue
            if se.issubset(listdci5) is True:
                for ll in range(le):
                    zhenyucheck[listo[g][ll]] = 1
                continue
            if se.issubset(listdci6) is True:
                for ll in range(le):
                    zhenyucheck[listo[g][ll]] = 1
                continue
    col=collections.Counter(zhenyucheck)
    lengthcount=col[1]      #lengthcount是zhenyucheck中1的个数
    fit =lengthcount/obn
    return fit

# 主程序
def genetic_algorithm_parallel(turn, new_view_index, wu_pos, superfine_pos, superfine,
                               listdci1, listdci2, listdci3,
                               listdci4, listdci5, listdci6,
                               pc1, pc2, pm,
                               t, t1, n,
                               obn, fn, df2,view_index
                               ):
    '''
    input:
        new_view_index:一个深度相同的层，具体为一个嵌套的二维数组，例如[[1,2], [2,3]]
        wu_pos, superfine_pos, superfine:GA的变量
    output:并行计算下最粗的满意粒度层
    '''
    # best_pos = np.zeros(turn)
    # best_poscount = np.zeros(len(new_view_index))
    # satisfycount = np.zeros(len(new_view_index))

    #Map
    args_list = [(new_view_index[j], wu_pos, superfine_pos, superfine,
                  listdci1, listdci2, listdci3,
                  listdci4, listdci5, listdci6,
                  pc1, pc2, pm,
                  t, t1, n,
                  obn, fn, df2, view_index) for j in range(len(new_view_index))] # 将并行化变量准备好
    # print(args_list)
    with multiprocessing.Pool(3) as pool: # 并行化操作
        results = pool.map(parallel_GA, args_list)
    #Reduce
    # print(results)
    result = min(results, key=lambda x: x[0].size) # 找出最粗粒度层输出，即数组长度最小的结果

    # for result in results:
    return result


# 并行化GA函数
def parallel_GA(args):
    '''
    input: GA函数所需要的变量共同组成的数组
    output: GA函数执行结果
    '''
    population, temp_yuzhi, superfine_pos, superfine,\
        listdci1, listdci2, listdci3,\
        listdci4, listdci5, listdci6,\
        pc1, pc2, pm,\
        t, t1, n,\
        obn, fn, df2, view_index = args

    return GA(population, temp_yuzhi, superfine_pos, superfine,
              listdci1, listdci2, listdci3, listdci4, listdci5, listdci6,
              pc1, pc2, pm,
              t, t1, n,
              obn, fn, df2, view_index)

def GA(pop,temp_yuzhi,superfine_pos,superfine,
       listdci1, listdci2, listdci3,
       listdci4, listdic5, listdci6,
       pc1, pc2, pm,
       t, t1, n,
       obn, fn, df2, view_index):
    satisfy = 0
    # print('length', pop[0].size)
    viewnum = len(pop[0])

    population = copy.deepcopy(pop)
    fitness_change = np.zeros(t)  # 记录每一代的适应度最大的染色体的适应度值
    fitness_change2 = np.zeros(t1)  # 记录每一代的第二种适应度最大的染色体的适应度值
    bestpop = np.zeros((t, viewnum))  # 记录每一代最好的染色体
    bestpop2 = np.zeros((t1, viewnum))
    fitness = np.zeros(n)  # fitness为每一个个体的适应度值

    for i in range(t):
        if(i==0):
            print(i)
            timestart=time.time()
        if i == 0:
            for j in range(n):
                fitness[j] = Jd2(population[j],view_index)  # 计算每一个体的适应度值
            fitness_change[i] = min(fitness)  # 找出每一代的适应度最大的染色体的适应度值
            c = fitness.argmin()
            bestpop[i] = copy.deepcopy(population[c])
        population,fitness= selection(population, fitness, n)  # 通过概率选择产生新一代的种群
        population,fitness= crossover1(population, int(n / 2),temp_yuzhi,fitness, pc1,
                                       listdci1, listdci2, listdci3, listdci4, listdic5, listdci6,
                                       obn, fn, df2, view_index)  # 通过交叉产生新的个体
        population,fitness= mutation1(population,temp_yuzhi,fitness, n, pm,
                                      listdci1, listdci2, listdci3,
                                      listdci4, listdic5, listdci6,
                                      obn, fn, df2, view_index)  # 通过变异产生新个体
        if(i<t-1):
            population[fitness.argmin()] = copy.deepcopy(bestpop[i])
            fitness[fitness.argmin()] = copy.deepcopy(fitness_change[i])
            fitness_change[i + 1] = max(fitness)  # 找出每一代的适应度最大的染色体的适应度值
            bestpop[i + 1] = copy.deepcopy(population[fitness.argmax()])
        if(i==0):
            timeend=time.time()
            print(timeend-timestart)

    best_fitness = min(fitness_change)  # 记录下所有迭代中最大的适应度
    num = fitness_change.argmin()
    best_people = copy.deepcopy(bestpop[num])
    fitness3 = np.zeros(n)  # fitness为每一个个体的适应度值
    population2=copy.deepcopy(start2(bestpop,t,n, view_index))
    for ii in range(t1):
        if(ii==0):
            print(ii)
            timestart = time.time()
        if ii == 0:
            for j in range(n):
                fitness3[j] = Jd2(population2[j], view_index)
            fitness_change2[ii] = min(fitness3)  # 找出每一代中深度最小的
            c = fitness3.argmin()
            bestpop2[ii] = copy.deepcopy(population2[c])
        # fitnesssum = sum(fitness3)
        # fitnessavg = fitnesssum / n
        #print("第%d代种群平均适应度值为" % ii, fitnessavg)
        if (ii) < (t1)*(0.5):
            population2,fitness3= selection(population2, fitness3, n)  # 通过概率选择产生新一代的种群
        '''if (ii) < (t1) * (0.5):
            population2,fitness3= crossover(population2, int(n / 2),temp_yuzhi,fitness3,1)  # 通过交叉产生新的个体
        else:
            population2, fitness3 = crossover(population2, int(n / 2), temp_yuzhi, fitness3,2)  # 通过交叉产生新的个体'''

        population2, fitness3 = crossover1(population2, int(n / 2), temp_yuzhi, fitness3, pc1,
                                           listdci1, listdci2, listdci3,
                                           listdci4, listdic5, listdci6,
                                           obn, fn, df2, view_index)

        if (ii) < (t1) * (0.5):
            population2,fitness3= mutation(population2,temp_yuzhi,fitness3,0.7, n, pm,
                                           listdci1, listdci2, listdci3,
                                           listdci4, listdic5, listdci6,
                                           obn,fn ,df2, view_index)  # 通过变异产生新个体
        else:
            population2, fitness3 = mutation(population2, temp_yuzhi, fitness3, 0.5, n, pm,
                                             listdci1, listdci2, listdci3,
                                             listdci4, listdic5, listdci6,
                                             obn, fn, df2, view_index)  # 通过变异产生新个体
        #population2, fitness3 = mutation1(population2, temp_yuzhi, fitness3)
        if(ii<t1-1):
            population2[fitness3.argmin()] = copy.deepcopy(bestpop2[ii])
            fitness3[fitness3.argmin()] = copy.deepcopy(fitness_change2[ii])
            fitness_change2[ii + 1] = min(fitness3)  # 找出每一代的适应度最大的染色体的适应度值
            bestpop2[ii + 1] = copy.deepcopy(population2[fitness3.argmin()])
        if(ii==0):
            timeend = time.time()
            print(timeend - timestart)

    best_fitness2 = min(fitness_change2)  # 记录下所有迭代中最大的适应度
    num2 = fitness_change2.argmin()
    best_people2 = copy.deepcopy(bestpop2[num2])
    if best_fitness <= best_fitness2:
        best_people = copy.deepcopy(best_people2)
    best_pos=Jd1(best_people, obn, fn, df2, view_index,
                 listdci1, listdci2, listdci3,
                 listdci4, listdic5, listdci6)
    for i1 in range(viewnum):
        p1 = best_people[i1]
        p1 = p1.astype(int)
        satisfy = satisfy + p1
    return best_people, best_pos, satisfy

def start2(bestpop,t ,n, view_index):
    viewnum = bestpop[0].size
    population = np.zeros((n, viewnum))
    for i in range(t):
        population[i]=bestpop[i]
    for i in range(t,n):
        for j in range(viewnum):
            population[i, j]=random.randint(1,len(view_index[j]))
    return population


# 轮盘赌选择
def selection(population, fitness, n):
    viewnum = population[0].size
    fitness_sum = np.zeros(n)     #用来计算不同染色体适应度值所占比例
    fitness_new=np.zeros(n)
    for i in range(n):
        if i == 0:
            fitness_sum[i] = fitness[i]
        else:
            fitness_sum[i] = fitness[i] + fitness_sum[i - 1]
    for i in range(n):
        fitness_sum[i] = fitness_sum[i] / sum(fitness)

    # 选择新的种群
    population_new = np.zeros((n, viewnum))
    for i in range(n):
        rand = np.random.uniform(0, 1)
        for j in range(n):
            if j == 0:
                if rand <= fitness_sum[j]:
                    population_new[i] =copy.deepcopy(population[j])
                    fitness_new[i]=fitness[j]
                    break
            else:
                if fitness_sum[j - 1] < rand and rand <= fitness_sum[j]:
                    population_new[i] =copy.deepcopy(population[j])
                    fitness_new[i] = fitness[j]
                    break
    return population_new, fitness_new


# 交叉操作
def crossover1(population1, kk, temp_yuzhi, fitness, pc1,
               listdci1, listdci2, listdci3, listdci4, listdci5, listdci6,
               obn, fn, df2, view_index):
    viewnum = population1[0].size
    popu = copy.deepcopy(population1)
    np.random.shuffle(popu)
    father = copy.deepcopy(popu[0:kk, :])
    mother = copy.deepcopy(popu[kk:, :])
    fitfa = np.zeros(kk)
    fitmo = np.zeros(kk)
    for i in range(kk):
        c = np.random.uniform(0, 1)
        # for y in range (fn):
        father_1 = copy.deepcopy(father[i])
        mother_1 = copy.deepcopy(mother[i])
        if pc1>=c :
            # half_length = int(length / 2)  # half_length为交叉的位数
            # half_length=length
            for k in range(int(viewnum/2)):  # 进行交叉操作
                chucun=father_1[k]
                father_1[k]=mother_1[k]
                mother_1[k] =chucun
            jdfa1=Jd(father_1,temp_yuzhi,
                     listdci1, listdci2, listdci3,
                     listdci4, listdci5, listdci6,
                     obn, fn, df2, view_index)
            jdmo1=Jd(mother_1,temp_yuzhi,
                     listdci1, listdci2, listdci3,
                     listdci4, listdci5, listdci6,
                     obn, fn, df2, view_index)
            father[i] = copy.deepcopy(father_1)  # 将交叉后的个体替换原来的个体
            fitfa[i]=jdfa1
            mother[i] = copy.deepcopy(mother_1)
            fitmo[i]=jdmo1
    population1 = np.append(father, mother, axis=0)
    fitn=np.append(fitfa,fitmo,axis=0)
    return population1,fitn


def mutation1(population,temp_yuzhi,fitness,n ,pm,
              listdci1, listdci2, listdci3,
              listdci4, listdci5, listdci6,
              obn, fn, df2, view_index):
    viewnum = population[0].size
    fit=copy.deepcopy(fitness)
    for i in range(n):
        c = np.random.uniform(0, 1)
        if(pm>=c):
            mutation_s=copy.deepcopy(population[i])
            a =random.randint(0, viewnum-1)  # e是随机选择由0变为1的位置
            mutation_s[a] = random.randint(1,len(view_index[a]))
            a2 = Jd(mutation_s,temp_yuzhi,
                  listdci1, listdci2, listdci3,
                  listdci4, listdci5, listdci6,
                  obn, fn, df2, view_index)
            population[i] = copy.deepcopy(mutation_s)
            fit[i]=a2
    return population,fit
# 变异操作
def mutation(population,temp_yuzhi,fitness,a, n, pm,
             listdci1, listdci2, listdci3,
             listdci4, listdci5, listdci6,
             obn, fn, df2, view_index):
    viewnum = population[0].size
    fit=copy.deepcopy(fitness)
    fmax=max(fit)
    favg=sum(fit)/n
    if(a*fmax<favg):
        pmmax=0.4
    else:
        pmmax=pm
    for i in range(n):
        c = np.random.uniform(0, 1)
        if(pmmax>=c):
            mutation_s=copy.deepcopy(population[i])
            a =random.randint(0, viewnum-1)  # e是随机选择由0变为1的位置
            mutation_s[a] = random.randint(1, len(view_index[a]))
            a2=Jd(mutation_s,temp_yuzhi,
                  listdci1, listdci2, listdci3,
                  listdci4, listdci5, listdci6,
                  obn, fn, df2, view_index)
            if a2>=fit[i]:
                population[i] = copy.deepcopy(mutation_s)
                fit[i]=a2
    return population,fit

def Jd(x,temp_yuzhi,
       listdci1, listdci2, listdci3,
       listdci4, listdci5, listdci6,
       obn, fn ,df2, view_index):
    # pop=x
    # print('x', len(x))
    kcount=0
    for i1 in range(len(x)):
        p1=x[i1]
        p1=p1.astype(int)
        kcount=kcount+p1
    fx=Jd1(x,
           obn, fn, df2, view_index,
           listdci1, listdci2, listdci3,
           listdci4, listdci5, listdci6)
    '''if(fx>=temp_yuzhi):
        fit = fx*0.8+ ((1 - float(kcount / fn) )* 0.2)
    else:
        fit = fx * 0.8 + (float(1 - kcount / fn) * 0.02)'''
    fit = (0.907821-abs(fx-temp_yuzhi))* 0.99 + ((1 - float(kcount / fn)) * 0.01)
    return fit

#print(viewnum,viewlen,last_viewlen)
def start(viewnum, n, view_index):
    population = np.zeros((n, viewnum))
    for i in range(n):
        for j, view in enumerate(view_index):
            population[i, j] = random.randint(1, len(view))
            pass
    return population


# 用来计算最优粒度层
def Jd2(x, view_index):
    '''
    :param x: 选取视图数
    :param view_index: 视图组成
    :return: 返回计算后的粒度数
    '''
    count = 0
    for i, j in enumerate(x):
        count = count + len(view_index[i][int(j)-1])
        pass
    return count
    pass
