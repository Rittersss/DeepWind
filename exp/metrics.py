import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix

spd_bins = [0, 5.5, 8, 10.8, 13.9, 17.2, 9999]
spd_labels = [1, 2, 3, 4, 5, 6]
max_bins = [0, 8, 10.8, 13.9, 17.2, 20.8, 9999]
max_labels = [1, 2, 3, 4, 5, 6]
weight = [0.05, 0.05, 0.15, 0.15, 0.25, 0.35]

def TS_score(true, pred, bins, labels, weight):
    true = pd.cut(x=true, bins=bins, labels=labels, include_lowest=True, right=False)
    pred = pd.cut(x=pred, bins=bins, labels=labels, include_lowest=True, right=False)
    TS = []
    for label in spd_labels:
        item_confusion = pd.DataFrame(confusion_matrix(true==label, pred==label))
        NA = item_confusion.loc[1,1]
        NB = item_confusion.loc[0,1]
        NC = item_confusion.loc[1,0]
        if NA+NB+NC == 0:
            item_TS = 0
        else:
            item_TS = NA/(NA+NB+NC)
        TS.append(item_TS)
    return np.multiply(np.array(weight), np.array(TS)).sum()
def AC_score(train_oof):
    bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 9999]
    labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
    true = pd.cut(x=train_oof['wdir_2min_true'], bins=bins, labels=labels, include_lowest=True, right=True, ordered=False).astype('object')
    pred = pd.cut(x=train_oof['wdir_2min_pred'], bins=bins, labels=labels, include_lowest=True, right=True, ordered=False).astype('object')
    idx = train_oof['spd_2min_true']<=0.2
    true[idx] = 'C'
    idx = train_oof['spd_2min_pred']<=0.2
    pred[idx] = 'C'
    results = true==pred
    results[true=='C'] = True
    return results.mean()*100
def B_score(true, pred, thres):
    true = true >= thres
    pred = pred >= thres
    confusion = pd.DataFrame(confusion_matrix(true, pred))
    NA = confusion.loc[1,1]
    NB = confusion.loc[0,1]
    NC = confusion.loc[1,0]
    if NA+NB+NC == 0:
        B=1
    elif (NA+NB == 0) & (NC!=0):
        B=0
    elif (NA+NC == 0) & (NB!=0):
        B=0
    else:
        B = np.exp(-np.abs(np.log((NA+NB)/(NA+NC))))
    return B

def score(train_oof, test_pred):
    # train_oof columns: ID, wdir_2min_true, wdir_2min_pred, spd_2min_true, spd_2min_pred, spd_inst_max_true, spd_inst_max_pred
    # test_pred columns: ID, wdir_2min, spd_2min, spd_inst_max

    train_oof['group'] = train_oof['ID'].apply(lambda x: ''.join(x.split('_')[:-1]))
    train_oof['time'] = train_oof['ID'].apply(lambda x: int(x.split('_')[-1]))
    train_oof = train_oof.sort_values(['group', 'time']).reset_index(drop=True)
    
    train_oof_max = train_oof.copy(deep=True)
    train_oof_max['time'] = (train_oof_max.time-1)//24
    train_oof_max = train_oof.groupby(['group', 'time']).max()
    train_oof_max.reset_index(drop=True, inplace=True)

    TS1h_mean = TS_score(train_oof['spd_2min_true'], train_oof['spd_2min_pred'], spd_bins, spd_labels, weight)
    TS1h_max = TS_score(train_oof['spd_inst_max_true'], train_oof['spd_inst_max_pred'], max_bins, max_labels, weight)
    TS24h_mean = TS_score(train_oof_max['spd_2min_true'], train_oof_max['spd_2min_pred'], spd_bins, spd_labels, weight)
    TS24h_max = TS_score(train_oof_max['spd_inst_max_true'], train_oof_max['spd_inst_max_pred'], max_bins, max_labels, weight)
    AC1h = AC_score(train_oof)
    B_mean = B_score(train_oof['spd_2min_true'], train_oof['spd_2min_pred'], 13.9)
    B_max = B_score(train_oof['spd_inst_max_true'], train_oof['spd_inst_max_pred'], 20.8)


    score = (TS24h_mean-0.12)/0.04*0.24 + (TS24h_max-0.19)/0.1*0.24 \
        + (TS1h_mean-0.11)/0.03*0.135 + (TS1h_max-0.17)/0.07*0.135 \
        + (AC1h-50.18)/5.21*0.05 \
        + (B_mean-0.2)/0.27*0.1 + (B_max-0.22)/0.3*0.1
    print('score: {:.4f}'.format(score))
    print('TS24h_mean: {:.4f}'.format(TS24h_mean))
    print('TS24h_max: {:.4f}'.format(TS24h_max))
    print('TS1h_mean: {:.4f}'.format(TS1h_mean))
    print('TS1h_max: {:.4f}'.format(TS1h_max))
    print('AC1h: {:.4f}'.format(AC1h))
    print('B_mean: {:.4f}'.format(B_mean))
    print('B_max: {:.4f}'.format(B_max))
    print('mean: {:.4f}'.format((TS1h_mean-0.11)/0.03*0.135 + (B_mean-0.2)/0.27*0.1))
    print('max: {:.4f}'.format((TS1h_max-0.17)/0.07*0.135 + (B_max-0.22)/0.3*0.1))
    return score

def get_MSE_Score(train_oof):
    print(1)