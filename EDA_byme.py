# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:23:28 2019

@author: dell
"""

#import featexp
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import geohash

df_transaction=pd.read_csv('C:/Users/Administrator/Desktop/ticen/transaction_train_new.csv')
df_operation=pd.read_csv('C:/Users/Administrator/Desktop/ticen/operation_train_new.csv')
df_tag=pd.read_csv('C:/Users/Administrator/Desktop/ticen/tag_train_new.csv')

print(df_operation.columns)#训练集操作详情表单
print(df_transaction.columns)#训练集交易详情表单
print(df_tag.columns)#黑白样本标签

df1 = df_operation.head(100)
df2 = df_transaction.head(100)

l1 = df_transaction.columns.tolist()
l2 = df_operation.columns.tolist()
l_bin = [i for i in l1 if i in l2]
'''
操作详情页：UID-用户编号,day-操作日期,mode-操作类型,success-操作状态,time-操作时间点,os-操作系统
version-客户端版本号,decice1-操作设备参数1,device2-操作设备参数2,device_code1-操作设备唯一标识
device_code2-操作设备唯一标识2,mac1-MAC地址,ip1-IP地址,icoin_count-IP地址,device_code3-设备唯一标识3,
mac2-MAC地址，wifi-WIFI名称,geo_code-地理位置,ip1_sub-IP地址,icoin_count_sub-IP地址
交易详情页：UID-用户编号，channel-平台,day-交易日期,time-交易时间点,trans_amt-脱敏后交易金额,
amt_src1-资金类型,merchant-商虎标识,code1-商户标识,code2-商户终端设备标识,trans_type1-交易类型1,
acc_id1-账户相关,device_code1-操作设备唯一标识,device_code2-操作设备唯一标识2,device_code3-操作设备唯一标识3,
device1-操作设备参数1,device2-操作设备参数2,mac1-MAC地址,ip1-IP地址,bal-脱敏后账户余额,
amt_src2-资金相关,acc_id2-账户相关,acc_id3-账户相关,geocode-地理位置,trans_type2-交易类型,
market_code-营销活动号编码,market_type-营销活动标识,ip1_sub-IP地址
'''

df_transaction.info()
df_tag.info()
Counter(df_tag.Tag)
#Counter({0: 26894, 1: 4285}),26894: 4285

df_operation_t=pd.merge(df_operation,df_tag,on=['UID'],how='left')
df_transaction_t = pd.merge(df_transaction,df_tag,on=['UID'],how='left')

#观察缺失和类型
df_operation_t.info()
df_transaction_t.info()

df_operation_t.time = df_operation_t.time.str.split(':').str[0].astype(int)
df_transaction_t.time = df_transaction_t.time.str.split(':').str[0].astype(int)
#查看数据特征
from featexp import get_univariate_plots
# Plots drawn for all features if nothing is passed in feature_list parameter.
get_univariate_plots(data=df_operation_t, target_col='Tag', 
                     features_list=['time'], bins=24)
get_univariate_plots(data=df_transaction_t, target_col='Tag', 
                     features_list=['time'], bins=24)


df_operation_t[df_operation_t.Tag==1].groupby("day").size()
'''
##基于业务理解的分析和深入的分析方向
#包含“羊毛党”在内的欺诈用户一般特征，欺诈行为通常是牟利行为，因此需要成本尽量低，利润尽量高
操作类型的解释为，查询余额,修改密码，所以根据一个思路，即羊毛党是通过利用促销,
缺乏日常使用，即少量操作和交易来做
'''
pd.DataFrame({"羊毛党操作":df_operation_t[df_operation_t.Tag==1].groupby("UID").size().describe(),"用户操作":df_operation_t[df_operation_t.Tag==0].groupby("UID").size().describe()
             ,"羊毛党交易":df_transaction_t[df_transaction_t.Tag==1].groupby("UID").size().describe(),"用户交易":df_transaction_t[df_transaction_t.Tag==0].groupby("UID").size().describe()})
#通过数据分析证实，即羊毛党的操作确实显著的低于正常用户，而交易行为并不显著低于正常用户，可以从此处着手去进一步分析问题，区分用户，固此处的特征只需要用操作数据集即可

'''
##多账号的羊毛党
如果你想通过一场营销活动中收益，并且这个收益要能超出你专门从事此行业所付出的劳动，那么多账号获取收益是必然的。因此，从这个角度说，所有羊毛党必然是多账号用户。
如果一个用户使用多账号的话，那么此用户的手机，银行账号，各种设备识别号，必然会出现多账号共用情况。数据字段里大量的识别号也是默认了这些维度最容易识别羊毛党
所以大量的设备的操作参数和标识地址很重要，识别多账户共存
'''
def c1(x):
    return set(x.UID.values)
coin=df_transaction_t.groupby("acc_id2").apply(c1)#转出账号这个字段的重合
coin_count=Counter(coin.apply(len).values)
plt.loglog(coin_count.keys(),coin_count.values(),"o")
#幂律分布，是越多越好的偏好，证明有利益驱动，对应越多收益越高，对于正常用户账户越多月麻烦，鼓励的机制才会让账号越多的分布趋近于尾巴上翘
set_thre=set([])
for i in coin.values:
    if len(i)>3:
        set_thre=set_thre|i
set_all=set(df_transaction_t[df_transaction_t.Tag==1].UID.values)
print(len(set_thre),len(set_all),len(set_thre&set_all))


#将此分析过程封装为函数
def u1(x0,x1):
    def c1(x):
        return set(x.UID.values)
    coin=df_transaction_t.groupby(x0).apply(c1)
    coin_count=Counter(coin.apply(len).values)
    plt.loglog(coin_count.keys(),coin_count.values(),"o")
    set_thre=set([])
    for i in coin.values:
        if len(i)>x1:
            set_thre=set_thre|i
    return [set_thre,(len(set_thre),len(set_all),len(set_thre&set_all),format(len(set_thre&set_all)/len(set_thre),'.2f'))]

#调整参数至准确率越大，但尽量保证泛化能力，因为阈值越大样本越小，得不偿失，在确定的欺诈用户数量变化小的情况选择样本变化小的，准确率不变选样本大的
y1=u1("acc_id1",1)
y1[1]
y2=u1("acc_id2",3)#3，4拿不定，结果显示区别不大
y2[1]
y3=u1("acc_id3",1)
y3[1]
de1=u1("device_code1",3)
de1[1]
de2=u1("device_code2",3)
de2[1]
de3=u1("device2",4)#1，2都没有区分度
de3[1]
de4=u1("device_code3",1)#到2是13%的提升，但总体是降低，不知道与数据集分布是否有关
de4[1]
#手机型号基本没有分辨能力
ip1=u1("ip1",2)
ip1[1]
ip2=u1("ip2_sub",2)
ip2[1]
co1=u1("code1",2)
co1[1]
co2=u1("code2",2)#这个呈现线性关系，样本量也少，
co2[1]
co3=u1("time",10)#时间与欺诈行为呈现的是反抛物线，账号越多，对应时间点的数量少，与欺诈无关
co3[1]
#ip是比较弱的特征，但是覆盖面大，结合其他规则能提高召回率

#用更保守的方式构造特征，精度上升,如改变参数为1，4，1，4，5，5，但精度下降，所以用类似网格调参的方法寻找最大精度的参数
'''
def u1(x0,x1):
    def c1(x):
        return set(x.UID.values)
    coin=df_transaction_t.groupby(x0).apply(c1)
    set_thre=set([])
    for i in coin.values:
        if len(i)>x1:
            set_thre=set_thre|i
    return [set_thre,(len(set_thre),len(set_all),len(set_thre&set_all),format(len(set_thre&set_all)/len(set_thre),'.2f'))]

f2=0
for i in range(1,3):
    for o in range(3,5):
        for h in range(1,4):
            for j in range(3,6):
                for k in range(3,6):
                    y1=u1("acc_id1",i)
                    y2=u1("acc_id2",o)#3，4拿不定，结果显示区别不大
                    y3=u1("acc_id3",h)
                    de1=u1("device_code1",j)
                    de2=u1("device_code2",k)
                    w=y1[0]|y2[0]|y3[0]|de1[0]|de2[0]
                    f1=len(w&set_all)/len(set_all)
                    if f1>f2:
                        f2=f1
                        a=[f1,[i,o,h,j,k]]
'''
#结果是[0.5569747057350363, [1, 3, 1, 3, 3]],而[1, 3, 1, 4, 4]是0.5529676934635612,但后两个参数其实提升了7%的精度

w=y1[0]|y2[0]|y3[0]|de1[0]|de2[0]|co1[0]|de4[0]#code1的加入影响特别大，从0.55到0.66，这是出乎我意料的，我以为对方选择的变量已经是最好了
print(len(w),len(set_all),len(w&set_all))
f0=len(w&set_all)/len(w)
f1=len(w&set_all)/len(set_all)
f2=f0*f1*2/(f0+f1)
print(f0,f1,f2)#仅仅用简单的条件就能达到0.5以上的结果

#我们用最好的参数构造一列新的特征，即基于规则的识别，在这些条件中的为欺诈用户标记为1，非这个条件的是非欺诈用户标记为0
#用到了acc_id1，acc_id2，acc_id3，device_code1，device_code2，device_code3，code1
#可以通过以上特征的规则构建多列，最后两列是一列交集，一列并集，都是0，1的识别

#操作集上的识别多账户共存
def u2(x0,x1):
    def c2(x):
        return set(x.UID.values)
    coin=df_operation_t.groupby(x0).apply(c1)
    coin_count=Counter(coin.apply(len).values)
    plt.loglog(coin_count.keys(),coin_count.values(),"o")
    set_thre=set([])
    for i in coin.values:
        if len(i)>x1:
            set_thre=set_thre|i
    return [set_thre,(len(set_thre),len(set_all),len(set_thre&set_all),format(len(set_thre&set_all)/len(set_thre),'.2f'))]

#调整参数至准确率越大，但尽量保证泛化能力，因为阈值越大样本越小，得不偿失，在确定的欺诈用户数量变化小的情况选择样本变化小的，准确率不变选样本大的
n1=u2("device1",1)#无分辨能力
n1[1]
n2=u2("device2",3)
n2[1]
n3=u2("device_code1",4)#好特征，0.5
n3[1]
n4=u2("device_code2",3)#好特征
n4[1]
n5=u2("device_code3",3)#无分辨能力
n5[1]
n6=u2("mac2",3)#mac地址无分辨能力
n6[1]
n7=u2("wifi",10)#wifi无分辨能力
n7[1]
n8=u2("geo_code",10)#无分辨能力
n8[1]
#手机型号基本没有分辨能力,ip1,2都无分辨能力
n9=u2("ip1",10)
n9[1]
n10=u2("ip1_sub",2)
n10[1]

w_=n3[0]|n4[0]
print(len(w_),len(set_all),len(w_&set_all))
f0=len(w_&set_all)/len(w_)
f1=len(w_&set_all)/len(set_all)
f2=f0*f1*2/(f0+f1)
print(f0,f1,f2)#结果分数很差，通过操作的设备账号对应来确定没有太大的区分，同样的字段，但是在交易和操作数据集的区分不同，可以认为，欺诈用户的操作行为一定与正常用户有区别，类别的count,每种类别count的比率，或者金额与操作类与数量的比率
#上面有一个分析是羊毛党的操作显著低于正常用户但是交易并不会


#能否时间因素对欺诈用户产生区分的挖掘
plt.plot(df_transaction_t[df_transaction_t.Tag==1].groupby("day").size()/(df_transaction_t[df_transaction_t.Tag==1].groupby("day").size().max()-df_transaction_t[df_transaction_t.Tag==1].groupby("day").size().min()))
plt.plot(df_transaction_t[df_transaction_t.Tag==0].groupby("day").size()/(df_transaction_t[df_transaction_t.Tag==0].groupby("day").size().max()-df_transaction_t[df_transaction_t.Tag==0].groupby("day").size().min()))

plt.plot(df_operation_t[df_operation_t.Tag==1].groupby("day").size()/(df_operation_t[df_operation_t.Tag==1].groupby("day").size().max()-df_operation_t[df_operation_t.Tag==1].groupby("day").size().min()))
plt.plot(df_operation_t[df_operation_t.Tag==0].groupby("day").size()/(df_operation_t[df_operation_t.Tag==0].groupby("day").size().max()-df_operation_t[df_operation_t.Tag==0].groupby("day").size().min()))


plt.plot(df_transaction_t[df_transaction_t.Tag==1].groupby("time").size()/(df_transaction_t[df_transaction_t.Tag==1].groupby("time").size().max()-df_transaction_t[df_transaction_t.Tag==1].groupby("time").size().min()))
plt.plot(df_transaction_t[df_transaction_t.Tag==0].groupby("time").size()/(df_transaction_t[df_transaction_t.Tag==0].groupby("time").size().max()-df_transaction_t[df_transaction_t.Tag==0].groupby("time").size().min()))

plt.plot(df_operation_t[df_operation_t.Tag==1].groupby("time").size()/(df_operation_t[df_operation_t.Tag==1].groupby("time").size().max()-df_operation_t[df_operation_t.Tag==1].groupby("time").size().min()))
plt.plot(df_operation_t[df_operation_t.Tag==0].groupby("time").size()/(df_operation_t[df_operation_t.Tag==0].groupby("time").size().max()-df_operation_t[df_operation_t.Tag==0].groupby("time").size().min()))
#时间和时刻缺乏对欺诈用户的区分度，曲线较为一致，没有显著差异

#由于操作量与正常用户有显著的差异，我们来探索一下操作类别是否有偏好，是否有显著差异
a1 = df_operation_t[df_operation_t.Tag==1]['mode'].value_counts()/df_operation_t[df_operation_t.Tag==1]['mode'].value_counts().sum()
a2 = df_operation_t[df_operation_t.Tag==0]['mode'].value_counts()/df_operation_t[df_operation_t.Tag==0]['mode'].value_counts().sum()
a3 = (a1/a2).sort_values(ascending = False)



