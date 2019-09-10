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

df_round1_transaction=pd.read_csv('C:/Users/Administrator/Desktop/ticen/transaction_round1_new.csv')
df_round1_operation=pd.read_csv('C:/Users/Administrator/Desktop/ticen/operation_round1_new.csv')




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

#观察缺失和类型
df_transaction.info()
df_operation.info()
df_tag.info()
Counter(df_tag.Tag)
#Counter({0: 26894, 1: 4285}),26894: 4285
L1 = df_transaction.count()/len(df_transaction)
#acc_id1,amt_src2,device_code2,mac1,device_code1,geo_code,device1,device2,ip1,ip1_sub,trans_type2
L2 = df_operation.count()/len(df_operation)
#wifi,mac2,geo_code,device_code2,device2,device_code1,ip1,ip1_sub,version,device1,success

df_operation_t=pd.merge(df_operation,df_tag,on=['UID'],how='left')
df_transaction_t = pd.merge(df_transaction,df_tag,on=['UID'],how='left')


print(len(np.intersect1d(df_transaction_t['UID'],df_operation_t['UID'])),df_transaction_t['UID'].nunique(),df_operation_t['UID'].nunique())
print(len(np.intersect1d(df_round1_operation['UID'],df_round1_transaction['UID'])),df_round1_operation['UID'].nunique(),df_round1_transaction['UID'].nunique())
df_tag['UID'].nunique()#31179


df_operation_t.time = df_operation_t.time.str.split(':').str[0].astype(int)
df_transaction_t.time = df_transaction_t.time.str.split(':').str[0].astype(int)
#查看数据特征

df_transaction_t = df_transaction_t.drop(columns=['ip2','ip2_sub','mac1','device_code3'])
df_operation_t = df_operation_t.drop(columns=['code1','code2','acc_id2','acc_id3','mac1','market_type','market_code','device_code3'])


print(df_operation_t.columns)#训练集操作详情表单
print(df_transaction_t.columns)#训练集交易详情表单
print(df_tag.columns)#黑白样本标签



df_operation_t[df_operation_t.Tag==1].groupby("day").size()
'''
##基于业务理解的分析和深入的分析方向
#包含“羊毛党”在内的欺诈用户一般特征，欺诈行为通常是牟利行为，因此需要成本尽量低，利润尽量高
操作类型的解释为，查询余额,修改密码，所以根据一个思路，即羊毛党是通过利用促销,
缺乏日常使用，即少量操作和交易来做
'''
#通过数据分析证实，即羊毛党的操作确实显著的低于正常用户，而交易行为并不显著低于正常用户，可以从此处着手去进一步分析问题，区分用户，固此处的特征只需要用操作数据集即可
pd.DataFrame({"羊毛党操作":df_operation_t[df_operation_t.Tag==1].groupby("UID").size().describe(),"用户操作":df_operation_t[df_operation_t.Tag==0].groupby("UID").size().describe()})


df_transaction_t[df_operation_t.Tag==1].groupby("UID").size().describe()

df_operation_t['UID'].nunique()
df_operation_t.groupby('UID').size()

df_operation_t['day'].nunique()
df_operation_t.groupby('day').size()
plt.plot(df_operation_t[df_operation_t.Tag==1].groupby("day")['UID'].nunique())
plt.plot(df_operation_t[df_operation_t.Tag==0].groupby("day")['UID'].nunique())

df_operation_t['mode'].nunique()#89个
df_operation_t[df_operation_t.Tag==1].groupby("UID")['mode'].nunique()
df_operation_t[df_operation_t.Tag==0].groupby("UID")['mode'].nunique()

df_operation_t[df_operation_t.Tag==1].groupby('mode')['UID'].count()
df_operation_t[df_operation_t.Tag==0].groupby('mode')['UID'].count()

df_operation_t['success'].unique()#2个
df_operation_t[df_operation_t.Tag==1]['success'].sum()/df_operation_t[df_operation_t.Tag==1]['success'].count()#95.9%
df_operation_t[df_operation_t.Tag==0]['success'].sum()/df_operation_t[df_operation_t.Tag==0]['success'].count()#94.6


df_operation_t['time'].nunique()#80670个
df_operation_t['time'].unique()
df_operation_t['hour'] = df_operation_t['time'].str.split(':').str[0]
plt.plot(df_operation_t[df_operation_t.Tag==1].groupby("hour").size())
plt.plot(df_operation_t[df_operation_t.Tag==0].groupby("hour").size())
plt.legend(['1',"0"])

#提取小时成为一列新特征

df_operation_t['os'].nunique()#7个

df_operation_t['version'].nunique()#38个

df_operation_t.groupby('UID')['device1'].nunique()#2421个





df_transaction_t['UID'].nunique()
df_transaction_t.groupby('UID').size()
pd.DataFrame({"羊毛党交易":df_transaction_t[df_transaction_t.Tag==1].groupby("UID").size().describe(),"用户交易":df_transaction_t[df_transaction_t.Tag==0].groupby("UID").size().describe()})


df_transaction_t['channel'].nunique()
pd.DataFrame({"羊毛党频道":df_transaction_t[df_transaction_t.Tag==1].groupby("UID")['channel'].nunique().describe(),
             "用户频道":df_transaction_t[df_transaction_t.Tag==0].groupby("UID")['channel'].nunique().describe()})
#特征交叉，一个频道对应黑产和正常用户的数量
L3 = df_transaction_t[df_transaction_t.Tag==1].groupby('channel')['UID'].count()
L4 = df_transaction_t[df_transaction_t.Tag==0].groupby('channel')['UID'].count()
#频道对应黑产与非黑产用户的数量
channel_UID_count = pd.DataFrame({'黑产':L3,'非黑产':L4}).fillna(0)
#频道对应黑产与非黑产用户的占比
channel_UID_percent = channel_UID_count/channel_UID_count.sum()

plt.plot(pd.DataFrame(channel_UID_count.reset_index(drop=True)))
plt.xticks(pd.DataFrame(channel_UID_count.reset_index(drop=True).index))

plt.plot(pd.DataFrame(channel_UID_percent.reset_index(drop=True)))
plt.xticks(pd.DataFrame(channel_UID_percent.reset_index(drop=True).index))
plt.legend(['1',"0"])


df_transaction_t['day'].nunique()#30天
df_transaction_t.groupby('day').size()#1-30
plt.plot(df_transaction_t[df_transaction_t.Tag==1].groupby("day").size())
plt.plot(df_transaction_t[df_transaction_t.Tag==0].groupby("day").size())




df_transaction_t['UID'].nunique()



df_transaction_t['hour'] = df_transaction_t['time'].str.split(':').str[0]



df_transaction_t['time'].nunique()#60075个
df_transaction_t['time'].unique()
df_transaction_t['hour'] = df_transaction_t['time'].str.split(':').str[0]
plt.plot(df_transaction_t[df_transaction_t.Tag==1].groupby("hour").size())
plt.plot(df_transaction_t[df_transaction_t.Tag==0].groupby("hour").size())
plt.legend(['1',"0"])



df_transaction_t['trans_amt'].nunique()#11225个
#黑产用户金额分布
pd.DataFrame({"羊毛党额度种类":df_transaction_t[df_transaction_t.Tag==1].groupby("UID")['trans_amt'].nunique().describe(),
             "用户额度种类":df_transaction_t[df_transaction_t.Tag==0].groupby("UID")['trans_amt'].nunique().describe()})

pd.DataFrame({"羊毛党金额":df_transaction_t[df_transaction_t.Tag==1]['trans_amt'].describe(),
             "用户金额":df_transaction_t[df_transaction_t.Tag==0]['trans_amt'].describe()}).astype('int64')
plt.boxplot(df_transaction_t[df_transaction_t.Tag==1]['trans_amt'])

plt.scatter(df_transaction_t[df_transaction_t.Tag==1].groupby('trans_amt')['UID'].nunique().index,df_transaction_t[df_transaction_t.Tag==1].groupby('trans_amt')['UID'].nunique())
plt.scatter(df_transaction_t[df_transaction_t.Tag==0].groupby('trans_amt')['UID'].nunique().index,df_transaction_t[df_transaction_t.Tag==0].groupby('trans_amt')['UID'].nunique())
plt.scatter(df_transaction_t.groupby('trans_amt')['UID'].nunique().index,df_transaction_t.groupby('trans_amt')['UID'].nunique())


df_transaction_t['amt_src1'].nunique()#28个
df_transaction_t['amt_src1'].unique()

pd.DataFrame({"羊毛党":df_transaction_t[df_transaction_t.Tag==1].groupby("UID")['amt_src1'].nunique().describe(),
             "用户":df_transaction_t[df_transaction_t.Tag==0].groupby("UID")['amt_src1'].nunique().describe()})
df_transaction_t[df_transaction_t.Tag==1].groupby("amt_src1")['UID'].count()/df_transaction_t[df_transaction_t.Tag==1].groupby("amt_src1")['UID'].count().sum()
df_transaction_t[df_transaction_t.Tag==0].groupby("amt_src1")['UID'].count()/df_transaction_t[df_transaction_t.Tag==0].groupby("amt_src1")['UID'].count().sum()

df_transaction_t['bal'].nunique()#12307个
#黑产用户金额分布
pd.DataFrame({"羊毛党":df_transaction_t[df_transaction_t.Tag==1].groupby("UID")['bal'].nunique().describe(),
             "用户":df_transaction_t[df_transaction_t.Tag==0].groupby("UID")['bal'].nunique().describe()})

pd.DataFrame({"羊毛党余额":df_transaction_t[df_transaction_t.Tag==1]['bal'].describe(),
             "用户余额":df_transaction_t[df_transaction_t.Tag==0]['bal'].describe()}).astype('int64')



df_transaction_t['merchant'].nunique()#19766个
df_transaction_t['trans_type1'].nunique()#15个
df_transaction_t['trans_type2'].nunique()#4个

df_transaction_t['acc_id1'].nunique()#27630个
df_transaction_t['amt_src2'].nunique()#115个
