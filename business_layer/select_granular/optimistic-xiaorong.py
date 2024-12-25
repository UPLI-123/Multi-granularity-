import pandas as pd
import operator as op
import numpy as np
import random
import collections
import csv
import struct as st
from scipy.io import arff
import matplotlib.pyplot as plt
import time
import math
import copy

from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import neighbors
from sklearn import ensemble
from sklearn.model_selection import train_test_split

'''df=pd.read_csv('primary-tumor.csv',header=None, sep=',')
obn=363 #数据集中对象数量
fn=17 #条件属性数量'''
'''df=pd.read_csv('dataset1.csv',header=None, sep=',')
obn=189 #数据集中对象数量
fn=15 #条件属性数量'''
'''df=pd.read_csv('iris.csv',header=None, sep=',')
obn=150 #数据集中对象数量
fn=4 #条件属性数量'''
#df=pd.read_csv('sonar.all-data.csv',header=None, sep=',')
#df=pd.read_csv('wine.data.csv',header=None, sep=',')
#df=pd.read_csv('lymphography.data.csv',header=None, sep=',')
#df=pd.read_csv('test.csv',header=None, sep=',')
#df=pd.read_csv('divorce.csv',header=0, sep=';')
#df=pd.read_csv('vs2.csv',header=0, sep=',')
#df=pd.read_csv('vs2.csv',header=None, sep=',')
#df=pd.read_csv('sonar_test2.csv',header=None,sep=',')
df=pd.read_csv('../../../../伪装图片/pycharmproject/xuanzexiaorong/dermatology-change.csv', header=None, sep=',')
#df=pd.read_csv('CNAE-9.data.csv',header=None,sep=',')
#df=pd.read_csv('SCADI.csv',header=None,sep=',')
#df=pd.read_csv('SPECT.csv',header=None,sep=',')
#df=pd.read_csv('Urban.csv',header=None,sep=',')
#df=pd.read_csv('SPECT2.csv',header=None,sep=',')
#df=pd.read_csv('ionosphere.data.csv',header=0, sep=',')
#df=pd.read_csv('wdbc.data.csv',header=0, sep=',')
#df=pd.read_csv('diabetes_data_upload.csv',header=0,sep=',')
#df=pd.read_csv('Symptoms and COVID Presence (May 2020 data).csv',header=None, sep=',')
#df=pd.read_csv('plantCellSignaling.csv',header=None, sep=',')
#df=pd.read_csv('COVID, FLU, COLD Symptoms.csv',header=None, sep=',')
#df=pd.read_csv('COIL2000.csv',header=None, sep=',')
#df=pd.read_csv('emails.csv',header=None, sep=',')
#df=pd.read_csv('phishing websites.csv',header=None, sep=',')
#df=pd.read_csv('agaricus-lepiota.csv',header=None, sep=',')
#df=pd.read_csv('splice.csv',header=None, sep=',')
#df=pd.read_csv('splice01gai.csv',header=None, sep=',')
#df=pd.read_csv('Dry_Bean_Dataset.csv',header=None, sep=',')
#df=pd.read_csv('Video Games Rating By ESRB.csv',header=None, sep=',')
#df=pd.read_csv('Mushroom Classification.csv',header=None, sep=',')
#df=pd.read_csv('student-por.csv',header=None, sep=',')
#df=pd.read_csv('sonar_test2.csv',header=None, sep=',')
fn = len(df.columns) - 1  # 染色体长度
obn= len(df)  # 数据集样本个数
print(fn,obn)
df1=df.iloc[0:obn,0:fn]
df2=np.mat(df1)
dci1=df.iloc[0:obn,fn:fn+1]
dci2=np.mat(dci1)
dci=np.zeros(obn)
for b in range(obn):
    dci[b]=dci2[b,0]
turn=10
'''
listdci0=[]
listdci1=[]
for i in range(obn):
    if dci[i]==0:
        listdci0.append(i)
    if dci[i]==1:
        listdci1.append(i)
listdci1=set(listdci1)
listdci0=set(listdci0)
'''

'''listdci1=[]
listdci2=[]
for i in range(obn):
    if dci[i]==1:
        listdci1.append(i)
    if dci[i]==2:
        listdci2.append(i)
listdci1=set(listdci1)
listdci2=set(listdci2)
'''
'''
listdci0=[]
listdci1=[]
listdci2=[]
listdci3=[]
for i in range(obn):
    if dci[i]==0:
        listdci0.append(i)
    if dci[i]==1:
        listdci1.append(i)
    if dci[i]==2:
        listdci2.append(i)
    if dci[i]==3:
        listdci3.append(i)
listdci1=set(listdci1)
listdci0=set(listdci0)
listdci2=set(listdci2)
listdci3=set(listdci3)
'''
'''
listdci0=[]
listdci1=[]
listdci2=[]
listdci3=[]
listdci4=[]
listdci5=[]
for i in range(obn):
    if dci[i]==1:
        listdci0.append(i)
    if dci[i]==2:
        listdci1.append(i)
    if dci[i]==3:
        listdci2.append(i)
    if dci[i]==4:
        listdci3.append(i)
    if dci[i]==5:
        listdci4.append(i)
    if dci[i]==6:
        listdci5.append(i)
listdci1=set(listdci1)
listdci0=set(listdci0)
listdci2=set(listdci2)
listdci3=set(listdci3)
listdci4=set(listdci4)
listdci5=set(listdci5)
'''
'''
listdci0=[]
listdci1=[]
listdci2=[]
listdci3=[]
listdci4=[]
listdci5=[]
listdci6=[]
for i in range(obn):
    #if dci[i]==0:
        #listdci0.append(i)
        #continue
    if dci[i]==1:
        listdci1.append(i)
        continue
    if dci[i]==2:
        listdci2.append(i)
        continue
    if dci[i]==3:
        listdci3.append(i)
        continue
    if dci[i]==4:
        listdci4.append(i)
        continue
    if dci[i]==5:
        listdci5.append(i)
        continue
    if dci[i]==6:
        listdci6.append(i)
        continue
listdci1=set(listdci1)
listdci0=set(listdci0)
listdci2=set(listdci2)
listdci3=set(listdci3)
listdci4=set(listdci4)
listdci5=set(listdci5)
listdci6=set(listdci6)
'''

listdci1=[]
listdci2=[]
listdci3=[]
listdci4=[]
listdci5=[]
listdci6=[]
#listdci7=[]
for i in range(obn):
    if dci[i]==1:
        listdci1.append(i)
    if dci[i]==2:
        listdci2.append(i)
    if dci[i]==3:
        listdci3.append(i)
    if dci[i]==4:
        listdci4.append(i)
    if dci[i]==5:
        listdci5.append(i)
    if dci[i]==6:
        listdci6.append(i)
    #if dci[i]==7:
        #listdci7.append(i)
listdci1=set(listdci1)
listdci2=set(listdci2)
listdci3=set(listdci3)
listdci4=set(listdci4)
listdci5=set(listdci5)
listdci6=set(listdci6)
#listdci7=set(listdci7)

'''
listdci0=[]
listdci1=[]
listdci2=[]
listdci3=[]
listdci4=[]
listdci5=[]
listdci6=[]
listdci7=[]
listdci8=[]
for i in range(obn):
    if dci[i]==0:
        listdci0.append(i)
    if dci[i]==1:
        listdci1.append(i)
    if dci[i]==2:
        listdci2.append(i)
    if dci[i]==3:
        listdci3.append(i)
    if dci[i]==4:
        listdci4.append(i)
    if dci[i]==5:
        listdci5.append(i)
    if dci[i]==6:
        listdci6.append(i)
    if dci[i]==7:
        listdci7.append(i)
    if dci[i]==8:
        listdci8.append(i)

listdci1=set(listdci1)
listdci0=set(listdci0)
listdci2=set(listdci2)
listdci3=set(listdci3)
listdci4=set(listdci4)
listdci5=set(listdci5)
listdci6=set(listdci6)
listdci7=set(listdci7)
listdci8=set(listdci8)
'''
'''
listdci9=[]
listdci1=[]
listdci2=[]
listdci3=[]
listdci4=[]
listdci5=[]
listdci6=[]
listdci7=[]
listdci8=[]
for i in range(obn):
    if dci[i]==9:
        listdci9.append(i)
    if dci[i]==1:
        listdci1.append(i)
    if dci[i]==2:
        listdci2.append(i)
    if dci[i]==3:
        listdci3.append(i)
    if dci[i]==4:
        listdci4.append(i)
    if dci[i]==5:
        listdci5.append(i)
    if dci[i]==6:
        listdci6.append(i)
    if dci[i]==7:
        listdci7.append(i)
    if dci[i]==8:
        listdci8.append(i)

listdci1=set(listdci1)
listdci9=set(listdci9)
listdci2=set(listdci2)
listdci3=set(listdci3)
listdci4=set(listdci4)
listdci5=set(listdci5)
listdci6=set(listdci6)
listdci7=set(listdci7)
listdci8=set(listdci8)
'''
'''listdci0=[]
listdci1=[]
for i in range(obn):
    if dci[i]==1:
        listdci0.append(i)
    else:
        listdci1.append(i)
listdci1=set(listdci1)
listdci0=set(listdci0)'''


pc1=0.8
pc2=0.2
pm= 0.1  # pc为变异的概率

t = 5 # 遗传算法迭代的次数
t1=40
n = 60  # 种群的个体数,要求大于20以保证具有随机性



'''viewnum=math.ceil(pow(fn,0.5))
viewlen=int(fn/viewnum)
last_viewlen=fn-(viewnum-1)*viewlen'''
viewlen=5
viewnum=math.ceil(fn/viewlen)
last_viewlen=fn-(viewnum-1)*viewlen

#print(viewnum,viewlen,last_viewlen)
def start():
    population = np.zeros((n, viewnum))
    for i in range(n):
        for j in range(viewnum-1):
            population[i,j]=random.randint(1,viewlen)
        population[i,j+1]=random.randint(1,last_viewlen)
    return population

def start2(bestpop):
    population = np.zeros((n, viewnum))
    for i in range(t):
        population[i]=bestpop[i]
    for i in range(t,n):
        for j in range(viewnum-1):
            population[i,j]=random.randint(1,viewlen)
        population[i,j+1]=random.randint(1,last_viewlen)

    return population






# 轮盘赌选择
def selection(population, fitness):
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
    return population_new,fitness_new


# 交叉操作
def crossover1(population1,kk,temp_yuzhi,fitness):
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
            jdfa1=Jd(father_1,temp_yuzhi)
            jdmo1=Jd(mother_1,temp_yuzhi)
            father[i] = copy.deepcopy(father_1)  # 将交叉后的个体替换原来的个体
            fitfa[i]=jdfa1
            mother[i] = copy.deepcopy(mother_1)
            fitmo[i]=jdmo1
    population1 = np.append(father, mother, axis=0)
    fitn=np.append(fitfa,fitmo,axis=0)
    return population1,fitn


# 交叉操作
def crossover(population1,kk,temp_yuzhi,fitness,a):
    popu=copy.deepcopy(population1)
    fitn=copy.deepcopy(fitness)
    #np.random.shuffle(popu)
    move=random.randint(0, kk)#随机前挪move位
    movesave1=copy.deepcopy(popu[0:move, :])
    movesave2=copy.deepcopy(popu[move:, :])
    movefit1=copy.deepcopy(fitn[0:move])
    movefit2=copy.deepcopy(fitn[move: ])
    popu=np.append(movesave2, movesave1, axis=0)
    fitn = np.append(movefit2, movefit1)
    father = copy.deepcopy(popu[0:kk, :])
    mother = copy.deepcopy(popu[kk:, :])
    fitfa=copy.deepcopy(fitn[0:kk])
    fitmo=copy.deepcopy(fitn[kk: ])
    for i in range(kk):
        c = np.random.uniform(0, 1)
        #for y in range (fn):
        father_1=copy.deepcopy(father[i])
        mother_1=copy.deepcopy(mother[i])
        fa=copy.deepcopy(fitfa[i])
        mo=copy.deepcopy(fitmo[i])
        fit1=float((fa+mo)/2)
        if a == 1:
            pc=pc1+pc2*(1-fit1)
        else:
            pc=pc1-pc2*(fit1)
        if pc>=c :
            different=[]
            for j in range(viewnum):
                if father_1[j]!=mother_1[j]:
                    different.append(j)
            length = len(different)
            d = random.randint(1, length + 1)
            d = d - 1
            # half_length = int(length / 2)  # half_length为交叉的位数
            # half_length=length
            for k in range(d):  # 进行交叉操作
                if k < length:
                    p = different[k]
                    chucun=father_1[p]
                    father_1[p]=mother_1[p]
                    mother_1[p] =chucun
            jdfa1=Jd(father_1,temp_yuzhi)
            jdmo1=Jd(mother_1,temp_yuzhi)
            #if (jdfa1+jdmo1)>=(fa+mo):
            father[i] = copy.deepcopy(father_1)  # 将交叉后的个体替换原来的个体
            fitfa[i]=jdfa1
            mother[i] = copy.deepcopy(mother_1)
            fitmo[i]=jdmo1
            #else:
                #fitfa[i] = fa
                #fitmo[i] = mo
    population2 = np.append(father, mother, axis=0)
    fitn=np.append(fitfa,fitmo,axis=0)
    return population2,fitn


def mutation1(population,temp_yuzhi,fitness):
    fit=copy.deepcopy(fitness)
    for i in range(n):
        c = np.random.uniform(0, 1)
        if(pm>=c):
            mutation_s=copy.deepcopy(population[i])
            a =random.randint(0, viewnum-1)  # e是随机选择由0变为1的位置
            if a <viewnum-1:
                mutation_s[a] =random.randint(1,viewlen)
            else:
                mutation_s[a] = random.randint(1, last_viewlen)
            a2=Jd(mutation_s,temp_yuzhi)
            population[i] = copy.deepcopy(mutation_s)
            fit[i]=a2
    return population,fit
# 变异操作
def mutation(population,temp_yuzhi,fitness,a):
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
            if a <viewnum-1:
                mutation_s[a] =random.randint(1,viewlen)
            else:
                mutation_s[a] = random.randint(1, last_viewlen)
            a2=Jd(mutation_s,temp_yuzhi)
            if a2>=fit[i]:
                population[i] = copy.deepcopy(mutation_s)
                fit[i]=a2
    return population,fit

def Jd(x,temp_yuzhi):
    pop=copy.deepcopy(x)
    kcount=0
    for i1 in range(viewnum):
        p1=pop[i1]
        p1=p1.astype(int)
        kcount=kcount+p1
    fx=Jd1(x)
    '''if(fx>=temp_yuzhi):
        fit = fx*0.8+ ((1 - float(kcount / fn) )* 0.2)
    else:
        fit = fx * 0.8 + (float(1 - kcount / fn) * 0.02)'''
    fit = (0.907821-abs(fx-temp_yuzhi))* 0.99 + ((1 - float(kcount / fn)) * 0.01)
    return fit

def Jd1(x):
    pop=copy.deepcopy(x)
    # 从特征向量x中提取出相应的特征
    zhenyucheck=np.zeros(obn)
    kcount=0
    for i1 in range(viewnum):
        Feature = np.zeros(fn)  # 数组Feature用来存 x选择的是哪d个特征
        k = 0
        p1=pop[i1]
        p1=p1.astype(int)
        for i2 in range(p1):
            Feature[k]=i1*viewlen+i2
            k=k+1
        kcount=kcount+k
        df3 = np.zeros((obn, 1))
        for l in range(k):
            # if Feature[l]!=0:
            p = Feature[l]
            p = p.astype(int)  # 转化为整型数据
            q = df2[:, p]
            q = q.reshape(obn, 1)  # 在不改变数据内容的情况下，改变一个数组的格式
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
                for r in range(k):
                    if df3[i, r]==df3[j, r]:
                        judge = 1
                    else:
                        judge = 0
                        break
                if judge == 1:
                    #f = f + 1
                    #save2[f] = j
                    save3[j]=0
                    lists.append(j)
            if len(lists)!=0:
                listo.append(lists)
                listnum = listnum + 1
        for g in range(listnum):
            se=set(listo[g])
            le=len(listo[g])
            '''if se.issubset(listdci1) is True: #issubset判断集合的所有元素是否都包含在指定集合中
                for ll in range(le):
                    zhenyucheck[listo[g][ll]]=1
                continue
            if se.issubset(listdci0) is True:
                for ll in range(le):
                    zhenyucheck[listo[g][ll]]=1
                continue'''

            '''if se.issubset(listdci1) is True:  # issubset判断集合的所有元素是否都包含在指定集合中
                for ll in range(le):
                    zhenyucheck[listo[g][ll]] = 1
            if se.issubset(listdci2) is True:
                for ll in range(le):
                    zhenyucheck[listo[g][ll]] = 1'''
            '''
            if se.issubset(listdci1) is True:  # issubset判断集合的所有元素是否都包含在指定集合中
                for ll in range(le):
                    zhenyucheck[listo[g][ll]] = 1
                continue
            if se.issubset(listdci0) is True:
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
            '''

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
            '''
            if se.issubset(listdci7) is True:
                for ll in range(le):
                    zhenyucheck[listo[g][ll]] = 1
                continue
            '''
            '''
            if se.issubset(listdci1) is True: #issubset判断集合的所有元素是否都包含在指定集合中
                for ll in range(le):
                    zhenyucheck[listo[g][ll]]=1
                continue
            if se.issubset(listdci0) is True:
                for ll in range(le):
                    zhenyucheck[listo[g][ll]]=1
                continue
            if se.issubset(listdci2) is True:
                for ll in range(le):
                    zhenyucheck[listo[g][ll]]=1
                continue
            if se.issubset(listdci3) is True:
                for ll in range(le):
                    zhenyucheck[listo[g][ll]]=1
                continue
            if se.issubset(listdci4) is True:
                for ll in range(le):
                    zhenyucheck[listo[g][ll]]=1
                continue
            if se.issubset(listdci5) is True:
                for ll in range(le):
                    zhenyucheck[listo[g][ll]]=1
                continue
            if se.issubset(listdci6) is True:
                for ll in range(le):
                    zhenyucheck[listo[g][ll]]=1
                continue
            if se.issubset(listdci7) is True:
                for ll in range(le):
                    zhenyucheck[listo[g][ll]]=1
                continue
            if se.issubset(listdci8) is True:
                for ll in range(le):
                    zhenyucheck[listo[g][ll]]=1
                continue
            '''
            '''if se.issubset(listdci1) is True:  # issubset判断集合的所有元素是否都包含在指定集合中
                for ll in range(le):
                    zhenyucheck[listo[g][ll]] = 1
                continue
            if se.issubset(listdci0) is True:
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
                continue'''



    col=collections.Counter(zhenyucheck)
    lengthcount=col[1]      #lengthcount是zhenyucheck中1的个数
    fit =lengthcount/obn
    return fit


def GA(pop,temp_yuzhi,superfine_pos,superfine):
    satisfy=0
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
                fitness[j] = Jd(population[j],temp_yuzhi)  # 计算每一个体的适应度值
            fitness_change[i] = max(fitness)  # 找出每一代的适应度最大的染色体的适应度值
            c = fitness.argmax()
            bestpop[i] = copy.deepcopy(population[c])
        population,fitness= selection(population, fitness)  # 通过概率选择产生新一代的种群
        population,fitness= crossover1(population, int(n / 2),temp_yuzhi,fitness)  # 通过交叉产生新的个体
        population,fitness= mutation1(population,temp_yuzhi,fitness)  # 通过变异产生新个体
        if(i<t-1):
            population[fitness.argmin()] = copy.deepcopy(bestpop[i])
            fitness[fitness.argmin()] = copy.deepcopy(fitness_change[i])
            fitness_change[i + 1] = max(fitness)  # 找出每一代的适应度最大的染色体的适应度值
            bestpop[i + 1] = copy.deepcopy(population[fitness.argmax()])
        if(i==0):
            timeend=time.time()
            print(timeend-timestart)

    best_fitness = max(fitness_change)  # 记录下所有迭代中最大的适应度
    num = fitness_change.argmax()
    best_people = copy.deepcopy(bestpop[num])
    fitness3 = np.zeros(n)  # fitness为每一个个体的适应度值
    population2=copy.deepcopy(start2(bestpop))
    for ii in range(t1):
        if(ii==0):
            print(ii)
            timestart = time.time()
        if ii == 0:
            for j in range(n):
                fitness3[j] = Jd(population2[j],temp_yuzhi)
            fitness_change2[ii] = max(fitness3)  # 找出每一代的适应度最大的染色体的适应度值
            c = fitness3.argmax()
            bestpop2[ii] = copy.deepcopy(population2[c])
        fitnesssum = sum(fitness3)
        fitnessavg = fitnesssum / n
        #print("第%d代种群平均适应度值为" % ii, fitnessavg)
        #if (ii) < (t1)*(0.5):
        population2,fitness3= selection(population2, fitness3)  # 通过概率选择产生新一代的种群
        if (ii) < (t1) * (0.5):
            population2,fitness3= crossover(population2, int(n / 2),temp_yuzhi,fitness3,1)  # 通过交叉产生新的个体
        else:
            population2, fitness3 = crossover(population2, int(n / 2), temp_yuzhi, fitness3,2)  # 通过交叉产生新的个体
        if (ii) < (t1) * (0.5):
            population2,fitness3= mutation(population2,temp_yuzhi,fitness3,0.7)  # 通过变异产生新个体
        else:
            population2, fitness3 = mutation(population2, temp_yuzhi, fitness3, 0.5)  # 通过变异产生新个体
        #population2, fitness3 = mutation1(population2, temp_yuzhi, fitness3)
        if(ii<t1-1):
            population2[fitness3.argmin()] = copy.deepcopy(bestpop2[ii])
            fitness3[fitness3.argmin()] = copy.deepcopy(fitness_change2[ii])
            fitness_change2[ii + 1] = max(fitness3)  # 找出每一代的适应度最大的染色体的适应度值
            bestpop2[ii + 1] = copy.deepcopy(population2[fitness3.argmax()])
        if(ii==0):
            timeend = time.time()
            print(timeend - timestart)

    best_fitness2 = max(fitness_change2)  # 记录下所有迭代中最大的适应度
    num2 = fitness_change2.argmax()
    best_people2 = copy.deepcopy(bestpop2[num2])
    if best_fitness <= best_fitness2:
        best_people = copy.deepcopy(best_people2)
    best_pos=Jd1(best_people)
    '''if best_pos<temp_yuzhi:
        best_people=superfine
        best_pos=superfine_pos
        satisfy=fn
    else:
        for i1 in range(viewnum):
            p1=best_people[i1]
            p1=p1.astype(int)
            satisfy=satisfy+p1'''
    for i1 in range(viewnum):
        p1 = best_people[i1]
        p1 = p1.astype(int)
        satisfy = satisfy + p1
    return best_people,best_pos,satisfy

if __name__ == '__main__':
    temp_yuzhi = np.zeros(5)
    locality_yuzhi_fuben = 0.4
    wulidu = np.zeros(5)
    wuweizhi = np.zeros(5)
    satisfy=np.zeros(5)
    satisfycount = np.zeros(5)
    best_poscount=np.zeros(5)
    for num_1 in range(5):
        locality_yuzhi=0
        locality_yuzhi=locality_yuzhi_fuben+0.2*num_1
        print(locality_yuzhi)
        best_fitness=np.zeros(turn)
        best_pos=np.zeros(turn)
        pospercent=np.zeros(turn)
        superfine=np.zeros(viewnum)
        for i in range(viewnum-1):
            superfine[i]=viewlen
        superfine[i+1]=last_viewlen
        superfine_pos=Jd1(superfine)
        print("最细粒度的正域划分率为%f"%superfine_pos)
        cengcicount = 0
        choice = np.zeros(viewnum)
        pop = copy.deepcopy(superfine)
        print(pop)
        # 从特征向量x中提取出相应的特征
        zhenyucheck = np.zeros(obn)
        for i1 in range(viewnum):
            count = 0
            # obndci=np.zeros(obn)
            dcicount = np.zeros(obn)
            p1 = pop[i1]
            p1 = p1.astype(int)
            if p1 <= 1:
                choice[i1] = 1          #如果当前视角下最细粒层为1层，该视角不进行筛选，直接将最终结果对应的视角层数置为1
            else:
                while p1 >= 1:
                    obndci = np.zeros(obn)    #obndci用来记录个体所在等价类的类标签数量
                    Feature = np.zeros(p1)    #数组Feature用来存选择的是哪些个特征
                    k = 0
                    temporarycount = []
                    for i2 in range(p1):
                        Feature[k] = i1 * viewlen + i2
                        k = k + 1
                    df3 = np.zeros((obn, 1))
                    for l in range(k):
                        # if Feature[l]!=0:
                        p = Feature[l]
                        p = p.astype(int)  # 转化为整型数据
                        q = df2[:, p]
                        q = q.reshape(obn, 1)  # 在不改变数据内容的情况下，改变一个数组的格式
                        df3 = np.append(df3, q, axis=1)
                    df3 = np.delete(df3, 0, axis=1)

                    listo = []                    #存储所有等价类
                    listnum = 0
                    save3 = np.ones(obn)
                    for i in range(obn):
                        lists = []                 #存储某一个等价类
                        judge = 1
                        judge2 = 1
                        # f = 0
                        if save3[i] == 0:          #save3[i]=0就说明对应的第i个对象已经被划分到某个等价类中去了
                            continue
                        # save2[f] = i
                        lists.append(i)            #如果没被跳过就说明save3[i]！=0，对应的第i个对象还未被划到等价类中，用其作为开头划分一个新的等价类
                        for j in range(i + 1, obn, 1):      #从第i个对象往后面遍历， 遇到save3=0的就直接跳过
                            if save3[j] == 0:
                                continue
                            for r in range(k):
                                if df3[i, r] == df3[j, r]:
                                    judge = 1
                                else:
                                    judge = 0                #如果比较的两个对象的某个属性值不同，则将judge置为0
                                    break
                            if judge == 1:                   #如果judge=1，证明比较的两个对象在所有属性上的值都相同，将他们划分到一个等价类中
                                save3[j] = 0
                                lists.append(j)
                        listo.append(lists)                  #将新的等价类添加到listo列表中
                        listnum = listnum + 1                #等价类个数加一
                    for g in range(listnum):                 #分别遍历每一个等价类
                        dcinum = 0
                        dcilist = []
                        se = set(listo[g])
                        le = len(listo[g])                   #le记录当前等价类中的对象个数
                        for u in range(le):
                            if u == 0:
                                dcilist.append(dci2[listo[g][u]])    #如果当前对象是等价类中的第一个对象，就将它的决策属性值加入到dcilist中去
                                dcinum = dcinum + 1
                            else:
                                check = True
                                for v in range(dcinum):              #将当前对象的决策属性值与之前已经保存到dcilist中的决策属性值相比较
                                    if dci2[listo[g][u]] == dcilist[v]:
                                        check = False
                                        break
                                if check == True:                  #如果当前对象的决策属性值与之前已经保存到dcilist中的决策属性值都不相同就将当前对象的决策属性值加入到dcilist中去
                                    dcilist.append(dci2[listo[g][u]])
                                    dcinum = dcinum + 1
                        for v in range(le):
                            obndci[listo[g][v]] = dcinum
                        # temporarycount.append(dcinum)
                        # print(temporarycount)
                    check = True
                    checkobn=np.ones(obn)
                    locality_yuzhi_count=obn
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
        for g in range(viewnum):
            cengcicount = cengcicount + choice[g]
        wu_pos=Jd1(choice)
        print("Wu方法正域划分率为：%.4f" %wu_pos)
        print(choice)
        temp_yuzhi[num_1]=wu_pos/superfine_pos
        print("与最细粒层的正域划分率之比为：%.4f"%temp_yuzhi[num_1])
        '''wuweizhi=np.zeros(turn)
        for g in range(turn):
            wuweizhi[g]=cengcicount/fn'''
        wuweizhi[num_1] = cengcicount / fn
        wulidu[num_1] = wuweizhi[num_1] * fn
        print("粒度为：%.4f" % wulidu[num_1])

        for i in range(turn):
            population = copy.deepcopy(start())
            best_people, best_pos[i], satisfy = GA(population, wu_pos, superfine_pos, superfine)
            satisfycount[num_1] = satisfycount[num_1] + satisfy
            print("满足阈值的最粗粒层为：%.4f" % (satisfy))
            print("第%d次选择最好粒层组合为：" % i, best_people)
            best_poscount[num_1] = best_poscount[num_1] + best_pos[i]
            print("第%d次选出的问题求解层正域划分率为：" % i, best_pos[i])
            print("误差为%.4f", (best_pos[i] - wu_pos) / wu_pos)
        satisfycount[num_1] = satisfycount[num_1] / turn
        best_poscount[num_1] = best_poscount[num_1] / turn
        print("问题求解层平均粒度：", satisfycount[num_1])
        print("问题求解层粒度平均正域划分率：", best_poscount[num_1])

    x = np.arange(0.2, 1.2, 0.2)  # 函数返回一个有终点和起点的固定步长的排列，三个参数时，第一个参数为起点，第二个参数为终点，第三个参数为步长
    plt.xlabel('threshold')
    plt.ylabel('The ratio of the number of layers')
    plt.ylim((0.0, fn))  # y坐标的范围
    '''plt.plot(x,pospercent,color='red')
    plt.plot(x, lenpercent, color='black')
    plt.plot(x, wuweizhi, color='yellow')'''
    plt.plot(x, satisfycount, color='red')
    plt.plot(x, wulidu, color='black')
    plt.show()
