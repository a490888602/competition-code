# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:40:06 2019

@author: Administrator
"""
import pandas as pd
import numpy as np
import os
data_path='C:/Users/Administrator/Desktop/ticen/'

# 获取设备数
def mode_nunique(x):
    try:
        return x.value_counts().iloc[0]
    except:
        return np.nan

def data_group(data, col):
    # 某特征与其他重要区分特征的交叉组合，如设备，交易金额
    df = pd.DataFrame(data.groupby(col)['UID'].nunique())#获取col里变量对应的UID唯一值的统计次数
    df.columns = ['cnt_uid_' + col]#特征重命名,index是col
    df['cou_uid_' + col] = data[data[col].notnull()].groupby(['UID'])[col].count()#一个UID对应特征记录数
    #账户余额与交易金额的比值与其他特征的交叉组合
    df['least_ratio_' + col] = data.groupby(col)['ratio'].min()
    df['most_ratio_' + col] = data.groupby(col)['ratio'].max()
    df['range_ratio_' + col] = df['most_ratio_' + col] - df['least_ratio_' + col]
    
    #某特征变量对应的相同同设备最大重复数
    if col not in ['device_code1', 'device_code2', 'device_code3']:
        df['dc1_' + col] = data.groupby(col)['device_code1'].apply(mode_nunique)
        df['dc2_' + col] = data.groupby(col)['device_code2'].apply(mode_nunique)
        df['dc3_' + col] = data.groupby(col)['device_code3'].apply(mode_nunique)

    # UID关联的自身属性
    nunique_var = ['acc_id1', 'acc_id2', 'acc_id3', 'amt_src1', 'amt_src2', 'bal', 'channel', 'code1', 'code2', 'day',
                   'device1', 'device2', 'device_code1', 'device_code2', 'device_code3', 'geo_code', 'ip1', 'ip1_sub',
                   'ip2', 'ip2_sub', 'mac1', 'mac2', 'market_code', 'market_type', 'merchant', 'mode',
                   'os', 'success', 'trans_amt', 'trans_type1', 'trans_type2', 'version', 'wifi']

    for nv in nunique_var:
        df[f'uid_{nv}_nunique'] = data.groupby('UID')[nv].nunique()#统计不同值数量
        df[f'uid_{nv}_notnull'] = data[data[nv].notnull()].groupby('UID')['UID'].count()#统计非空记录数量

    df = df.reset_index()
    return df

def extract_feat(data_frame):
    data_tmp = pd.DataFrame(data_frame.groupby('UID')['trans_amt'].min())
    data_tmp.columns = ['uid_min_amt']
    data_tmp['uid_max_amt'] = data_frame.groupby('UID')['trans_amt'].max()
    data_tmp['uid_range_amt'] = data_tmp['uid_max_amt'] - data_tmp['uid_min_amt']

    data_tmp['uid_min_bal'] = data_frame.groupby('UID')['bal'].min()
    data_tmp['uid_max_bal'] = data_frame.groupby('UID')['bal'].max()
    data_tmp['uid_range_bal'] = data_tmp['uid_max_bal'] - data_tmp['uid_min_bal']

    data_tmp['uid_min_rat'] = data_frame.groupby('UID')['ratio'].min()
    data_tmp['uid_max_rat'] = data_frame.groupby('UID')['ratio'].max()
    data_tmp['uid_range_rat'] = data_tmp['uid_max_rat'] - data_tmp['uid_min_rat']


    # UID关联到的设备等的信息，对欺诈影响大的特征
    relate_var = ['acc_id1', 'acc_id2', 'acc_id3', 'amt_src1', 'amt_src2', 'version', 'code1', 'code2',
                  'device_code1', 'device_code2', 'device_code3', 'market_code', 'merchant', 'mode']
    for rv in relate_var:
        print(f'waiting for generating feature of {rv} ...')
        sample_data = data_frame[['UID', rv]].drop_duplicates()
        group_data = data_group(data_frame, rv)#该特征与其他特征的交叉
        sample_data = sample_data.merge(group_data, on=rv, how='left')#col对应的各项交叉特征加到col变量对应的UID上去
        data_tmp['relate_cnt_uid_' + rv + '_max'] = sample_data.groupby('UID')['cnt_uid_' + rv].max()
        data_tmp['relate_cnt_uid_' + rv + '_min'] = sample_data.groupby('UID')['cnt_uid_' + rv].min()
        data_tmp['relate_cnt_uid_' + rv + '_mean'] = sample_data.groupby('UID')['cnt_uid_' + rv].mean()
        data_tmp['relate_cnt_uid_' + rv + '_skew'] = sample_data.groupby('UID')['cnt_uid_' + rv].skew()

        data_tmp['relate_cnt_uid_' + rv + '_max'] = sample_data.groupby('UID')['cou_uid_' + rv].max()
        data_tmp['relate_cnt_uid_' + rv + '_min'] = sample_data.groupby('UID')['cou_uid_' + rv].min()
        data_tmp['relate_cnt_uid_' + rv + '_mean'] = sample_data.groupby('UID')['cou_uid_' + rv].mean()
        data_tmp['relate_cnt_uid_' + rv + '_skew'] = sample_data.groupby('UID')['cou_uid_' + rv].skew()
        
        if rv not in ['device_code1', 'device_code2', 'device_code3']:
            data_tmp['relate_dc1_' + rv + '_max'] = sample_data.groupby('UID')['dc1_' + rv].max()
            data_tmp['relate_dc1_' + rv + '_min'] = sample_data.groupby('UID')['dc1_' + rv].min()
            data_tmp['relate_dc1_' + rv + '_mean'] = sample_data.groupby('UID')['dc1_' + rv].mean()
            data_tmp['relate_dc1_' + rv + '_skew'] = sample_data.groupby('UID')['dc1_' + rv].skew()

            data_tmp['relate_dc2_' + rv + '_max'] = sample_data.groupby('UID')['dc2_' + rv].max()
            data_tmp['relate_dc2_' + rv + '_min'] = sample_data.groupby('UID')['dc2_' + rv].min()
            data_tmp['relate_dc2_' + rv + '_mean'] = sample_data.groupby('UID')['dc2_' + rv].mean()
            data_tmp['relate_dc2_' + rv + '_skew'] = sample_data.groupby('UID')['dc2_' + rv].skew()

            data_tmp['relate_dc3_' + rv + '_max'] = sample_data.groupby('UID')['dc3_' + rv].max()
            data_tmp['relate_dc3_' + rv + '_min'] = sample_data.groupby('UID')['dc3_' + rv].min()
            data_tmp['relate_dc3_' + rv + '_mean'] = sample_data.groupby('UID')['dc3_' + rv].mean()
            data_tmp['relate_dc3_' + rv + '_skew'] = sample_data.groupby('UID')['dc3_' + rv].skew()


        data_tmp['relate_least_ratio_' + rv + '_max'] = sample_data.groupby('UID')['least_ratio_' + rv].max()
        data_tmp['relate_least_ratio_' + rv + '_min'] = sample_data.groupby('UID')['least_ratio_' + rv].min()
        data_tmp['relate_least_ratio_' + rv + '_mean'] = sample_data.groupby('UID')['least_ratio_' + rv].mean()
        data_tmp['relate_least_ratio_' + rv + '_skew'] = sample_data.groupby('UID')['least_ratio_' + rv].skew()

        data_tmp['relate_most_ratio_' + rv + '_max'] = sample_data.groupby('UID')['most_ratio_' + rv].max()
        data_tmp['relate_most_ratio_' + rv + '_min'] = sample_data.groupby('UID')['most_ratio_' + rv].min()
        data_tmp['relate_most_ratio_' + rv + '_mean'] = sample_data.groupby('UID')['most_ratio_' + rv].mean()
        data_tmp['relate_most_ratio_' + rv + '_skew'] = sample_data.groupby('UID')['most_ratio_' + rv].skew()

        data_tmp['relate_range_ratio_' + rv + '_max'] = sample_data.groupby('UID')['range_ratio_' + rv].max()
        data_tmp['relate_range_ratio_' + rv + '_min'] = sample_data.groupby('UID')['range_ratio_' + rv].min()
        data_tmp['relate_range_ratio_' + rv + '_mean'] = sample_data.groupby('UID')['range_ratio_' + rv].mean()
        data_tmp['relate_range_ratio_' + rv + '_skew'] = sample_data.groupby('UID')['range_ratio_' + rv].skew()

    return data_tmp


operation_train = pd.read_csv(open(data_path+'operation_train_new.csv', encoding='utf8'))
transaction_train = pd.read_csv(open(data_path+'transaction_train_new.csv', encoding='utf8'))
tag_train = pd.read_csv(open(data_path+'tag_train_new.csv', encoding='utf8'))

operation_round1 = pd.read_csv(open(data_path+'operation_round1_new.csv', encoding='utf8'))
transaction_round1 = pd.read_csv(open(data_path+'transaction_round1_new.csv', encoding='utf8'))

action_train = operation_train.append(transaction_train).reset_index(drop=True)
action_train = action_train.sort_values(by=['UID', 'day', 'time'], ascending=[True, True, True])
action_train = action_train.merge(tag_train, on='UID')

action_round1 = operation_round1.append(transaction_round1).reset_index(drop=True)
action_round1 = action_round1.sort_values(by=['UID', 'day', 'time'], ascending=[True, True, True])

tag_valid = pd.read_csv(open(data_path+'test_tag_r11.csv', encoding='utf8'))[['UID']]
print('Successed load in train and test data.')

all_data = action_train.append(action_round1).reset_index(drop=True)
all_data['version'] = all_data.version.fillna('0.0.0')
all_data['ratio'] = all_data['trans_amt'] / all_data['bal']


data_var = extract_feat(all_data)
data_var = data_var.reset_index()#把UID放出来

train = tag_train.merge(data_var, on='UID')
valid = tag_valid.merge(data_var, on='UID')
print(f'Gen train shape: {train.shape}, test shape: {valid.shape}')

drop_train = train.T.drop_duplicates().T
drop_valid = valid.T.drop_duplicates().T

features = [i for i in drop_train.columns if i in drop_valid.columns]
print('features num: ', len(features)-1)

train[features + ['Tag']].to_csv(data_path+'/gen_data/juz_train_data.csv', index=False)
valid[features].to_csv(data_path+'/gen_data/juz_test_data.csv', index=False)


    

    
    
    
    