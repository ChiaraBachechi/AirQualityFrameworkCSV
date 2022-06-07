import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from sklearn.metrics import accuracy_score
font = {'family': 'Verdana',
        'color':  'black',
        'weight': 'heavy',
        'size': 30,
        }
EEA_soglie_NO = [50,100,200,400]
EEA_soglie_O3 = [80,100,120,140]
sensors_list = ['4003','4004','4005','4006','4007','4008','4009','4010','4011','4012','4013','4014']

def range_EEA_O3(val):
        clas = np.zeros(len(val), dtype=int)
        for i in np.arange(0, len(val)):
            if val.iloc[i] > 0 and val.iloc[i] <= 80:
                clas[i] = 1     #dark_green
            else:
                if val.iloc[i] > 80 and val.iloc[i] <= 100:
                    clas[i] = 2     #light_green
                else:
                    if val.iloc[i] > 100 and val.iloc[i] <= 120:
                        clas[i] = 3     #yellow
                    else:
                        if val.iloc[i] > 120 and val.iloc[i] <= 140:
                            clas[i] = 4     #orange
                        else:
                            clas[i] = 5         #red
        return clas
def range_EEA_NOx(val):
    clas = np.zeros(len(val), dtype=int)
    for i in np.arange(0, len(val)):
        if val.iloc[i] > 0 and val.iloc[i] <= 50:
            clas[i] = 1     #dark_green
        else:
            if val.iloc[i] > 50 and val.iloc[i] <= 100:
                clas[i] = 2     #light_green
            else:
                if val.iloc[i] > 100 and val.iloc[i] <= 200:
                    clas[i] = 3     #yellow
                else:
                    if val.iloc[i] > 200 and val.iloc[i] <= 400:
                        clas[i] = 4     #orange
                    else:
                        clas[i] = 5         #red
    return clas

class CalibrationEvaluationFramework():
    
    def evaluate(self,calibrated,real,p):
        mae_dic = {}
        mre_dic = {}
        rmse_dic = {}
        accuracy_dic = {}
        #mre_lstm = pd.Series()
        #mre_vrsvr = pd.Series()
        calibrated.phenomenon_time = pd.to_datetime(calibrated.phenomenon_time)
        calibrated.set_index('phenomenon_time',inplace=True)
        real.phenomenon_time = pd.to_datetime(real.phenomenon_time)
        real.set_index('phenomenon_time',inplace=True)
        union = calibrated.join(real, lsuffix = '_predicted', rsuffix = '_real', how='inner')
        df_mae = union[['label_' + p + '_predicted','label_' + p + '_real']].copy()
        df_mae.dropna(inplace = True)
        mae = abs(df_mae['label_' + p + '_predicted'] - df_mae['label_' + p + '_real'])
        mae_dic[p] = [mae.mean(),mae.min(),mae.max()]
        mre = abs(mae/df_mae['label_' + p  + '_real'])
        mre_perc = 0
        if (mre.shape[0] > 0):
            mre_perc = mre[mre <= 0.20].shape[0] / mre.shape[0]
        rmse = np.sqrt(((df_mae['label_' + p + '_predicted'] - df_mae['label_' + p + '_real']) ** 2).mean())
        #mre_dic[p] = [mre.replace([np.inf, -np.inf], np.nan).dropna().mean(),mre.replace([np.inf, -np.inf], np.nan).dropna().min(),mre.replace([np.inf, -np.inf], np.nan).dropna().max(), mre_perc]
        mre_dic[p] = mre_perc
        if p == 'o3':
            class_pred = range_EEA_O3(df_mae['label_' + p + '_predicted'])
            class_test = range_EEA_O3(df_mae['label_' + p + '_real'])
        else:
            class_pred = range_EEA_NOx(df_mae['label_' + p + '_predicted'])
            class_test = range_EEA_NOx(df_mae['label_' + p + '_real'])
        acc_EEA = accuracy_score(class_pred, class_test)
        #mre_lstm = mre_lstm.append(mre)
        #mre_vrsvr = mre_vrsvr.append(mre)
        rmse_dic[p] = rmse
        accuracy_dic[p] = acc_EEA
        result = {'MAE':mae_dic,'MRE':mre_dic,'RMSE':rmse_dic, 'ACCURACY': accuracy_dic}
        return result