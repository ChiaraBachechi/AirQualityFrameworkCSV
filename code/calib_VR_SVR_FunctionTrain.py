import os

import pandas as pd
import dill
import numpy as np
import psycopg2
import sklearn
import matplotlib.pyplot as plt
import json
import datetime  # this is to pring the current time during the run

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from calibrationAlgorithmTrainer_interfaces import *
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor


class Calib_VR_SVR_FunctionTrainer_001(CalibartionAlgorithmTrainer_interface):

    def init(self, info_dictionary):
        self.info_dictionary = info_dictionary
        self.calibrator = None

    def getCalibrator(self):
        return self.calibrator

    def doTrain(self, df_train_features, df_train_labels_full):
        #df_train_labels_full=df_train_labels_full.rename(columns={df_train_labels_full.columns[0]:self.info_dictionary['pollutant_label']})

        updateInfoData(self.info_dictionary, df_train_features, df_train_labels_full)
        df_train_labels = df_train_labels_full[self.info_dictionary['target_label']]
        #
        #
        print('conversion_factor_for :'+self.info_dictionary['pollutant_label'])
        print(" --- conversion_factor_for:\n",
           json.dumps(self.info_dictionary['units_of_measure'][self.info_dictionary['pollutant_label']]))
        factor = self.info_dictionary['units_of_measure'][self.info_dictionary['pollutant_label']]['conversions'][0][
           'factor']
        print(" --- factor: " + str(factor))
        #
        # n_jobs means the CPU. -1 is all :)
        pesi_vr=tuning_pesi_Voting(df_train_features,
                                   df_train_labels_full)
        param_svr=tuning_parametri_SVR(df_train_features,df_train_labels,'../results/')

        vr_no_number_of_previous_observations,vr_number_of_previous_observations,svr,scaler_feat,scaler_pollutant, min_max=calibration_VR_SVR_all(self.info_dictionary,df_train_features,df_train_labels_full,param_svr,
                                                    pesi_vr)

        calibrator={"VR_number_of_previous_observations":vr_number_of_previous_observations,
                    "VR_no_number_of_previous_observations":vr_no_number_of_previous_observations,
                    "SVR":svr}
        scaler={"scaler_feat":scaler_feat,
                "scaler_pollutant":scaler_pollutant}
        self.info_dictionary["hyper_parameters"]={"VR":pesi_vr,"SVR":param_svr}

        self.calibrator = Calib_VR_SVR_Function()
        self.calibrator.init(self.info_dictionary, calibrator,scaler,min_max)
        return self.calibrator


class Calib_VR_SVR_Function(CalibartionAlgorithm_interface):
    """
    the calibrator produced by the trainer - example
    """

    def init(self, info_dictionary, calibrator,scaler,min_max):
        super().init(info_dictionary)
        info = self.info_dictionary
        #
        self.calibrator = calibrator
        self.scaler=scaler
        self.min_max=min_max

    def apply_df(self, data_frame_in,interval,path_dill):

        scaler_feat = self.scaler["scaler_feat"]
        scaler_pollutant = self.scaler["scaler_pollutant"]

        min_train=list(self.min_max['min_train'].values())
        max_train=list(self.min_max['max_train'].values())

        # print(" ---- data_frame_in: " + str(data_frame_in))
        dataset_feature = data_frame_in[self.info_dictionary['feat_order']]
        dataset_na = dataset_feature.dropna()
        dataset_feature['phenomenon_time'] = pd.to_datetime(data_frame_in['phenomenon_time'])

        # need to add a test for outside of the bounds. they should be percent
        if dataset_na.empty:
            return np.nan
        else:
            pred = pd.DataFrame()
            pred['phenomenon_time'] = data_frame_in.iloc[self.info_dictionary["number_of_previous_observations"]:]['phenomenon_time']
            yhat=[]
            
            prev_data=dataset_feature.index[0]
            dataset_feature.drop('phenomenon_time',axis=1,inplace=True)
            for i in np.arange(self.info_dictionary['number_of_previous_observations'], dataset_feature.shape[0]):
                range = np.all((dataset_feature.iloc[i] > min_train) & (dataset_feature.iloc[i] < max_train))
                if range == False:
                    tmp = np.array(dataset_feature.iloc[i].copy(), dtype=float)
                    tmp = scaler_feat.transform(tmp.reshape(1, -1))
                    p = np.array(self.calibrator['SVR'].predict(tmp))
                    p = scaler_pollutant.inverse_transform(p.reshape(-1, 1))
                    yhat.append(np.abs(p[0]))
                else:
                    cur = dataset_feature.index[i]
                    diff_number_of_previous_observations = cur - prev_data
                    diff_number_of_previous_observations= diff_number_of_previous_observations.total_seconds() / 60
                    if (diff_number_of_previous_observations == (interval * self.info_dictionary['number_of_previous_observations'])):
                        tmp = dataset_feature.iloc[i].copy()
                        for k in np.arange(1, self.info_dictionary['number_of_previous_observations'] + 1):
                            if self.info_dictionary['pollutant_label'] == "o3":
                                tmp['ox_aux_{}'.format(k)] = dataset_feature.iloc[i - k]['ox_aux']
                                tmp['ox_we_{}'.format(k)] = dataset_feature.iloc[i - k]['ox_we']
                            else:
                                tmp[self.info_dictionary['pollutant_label'] + '_aux_{}'.format(k)] = \
                                    dataset_feature.iloc[i - k][self.info_dictionary['pollutant_label'] + '_aux']
                                tmp[self.info_dictionary['pollutant_label'] + '_we_{}'.format(k)] = \
                                    dataset_feature.iloc[i - k][self.info_dictionary['pollutant_label'] + '_we']
                        tmp = np.array(tmp, dtype=float).reshape(1, -1)
                        p = self.calibrator['VR_number_of_previous_observations'].predict(tmp)
                        yhat.append(p)
                    else:
                        tmp = np.array(dataset_feature.iloc[i].copy(), dtype=float).reshape(1, -1)
                        p = self.calibrator['VR_no_number_of_previous_observations'].predict(tmp)
                        yhat.append(p)
                prev_data = dataset_feature.index[i]
        pred[self.info_dictionary["target_label"]]=yhat
        pred[self.info_dictionary["target_label"]] = pred[self.info_dictionary["target_label"]].astype(float)

        return pred

    #
    #
    @abstractfunc
    def get_info(self):
        return (self.info_dictionary)



def updateInfoData(info_dictionary, df_train_features, df_train_labels_full):

    #df_train_labels = df_train_labels_full[info_dictionary['pollutant_label']]
    df_train_labels = df_train_labels_full[info_dictionary['target_label']]
    print("Hello! updateInfoData " \
          + info_dictionary['trainer_class_name'] \
          + "." + info_dictionary['trainer_module_name'] \
          + " ...")
    # print(str(df_train_features))
    # print("--- labels ---")
    # print(str(df_train_labels))
    #
    trees = 1500  # roughly the optimum need to recheck for all sensors
    leaves = 10  # the minimum number of leaves per a split
    #if ('hyper_parameters' in info_dictionary['algorithm_parameters']):
    #    d = info_dictionary['algorithm_parameters']['hyper_parameters']
    #    trees = d['trees']
    #    leaves = d['leaves']
    #info_dictionary['algorithm_parameters']['hyper_parameters'] = \
    #    {'trees': trees, 'leaves': leaves}
    info_dictionary["name"]="Voting Regressor + SVR"
    info_dictionary["python_env"] = {'sklearn': sklearn.__version__,
                                     'pandas': pd.__version__,
                                     'dill': dill.__version__,
                                     'numpy': np.__version__,
                                     'psycopg2': psycopg2.__version__}
    info_dictionary["features"] = {}
    info_dictionary["number_of_previous_observations"]=info_dictionary['number_of_previous_observations']
    #
    ACmin_values = np.percentile(df_train_features, 0, axis=0)
    ACmax_values = np.percentile(df_train_features, 100, axis=0)
    for idx, feat in enumerate(info_dictionary['feat_order']):
        info_dictionary['features'][feat] = {}
        # add the max and min values
        info_dictionary['features'][feat]['range'] = ['%.2f' % float(ACmin_values[idx]),
                                                      '%.2f' % float(ACmax_values[idx])]
        info_dictionary['features'][feat]['unit_of_measure'] = 'mV'
    #
    for label_name in info_dictionary['label_list']:
        l = label_name
        info_dictionary[l] = {}
        info_dictionary[l]['range'] = ['%.2f' % float(df_train_labels.min()), '%.2f' % float(df_train_labels.max())]
        [min(df_train_labels), max(df_train_labels)]
        info_dictionary[l]['unit_of_measure'] = info_dictionary['units_of_measure'][label_name]['unit_of_measure']
    #
    #l = "label_" + info_dictionary['pollutant_label']
    #info_dictionary[l]
    info_dictionary['pollutant_unit_of_measure'] = \
    info_dictionary['units_of_measure'][info_dictionary['pollutant_label']]['unit_of_measure']





"""
Determino i pesi dei tre regressori.

"""
def tuning_pesi_Voting(data, pollutant):
    print("\nTuning pesi per il Voting")

    y = pollutant
    X = data.copy()

    random = RandomForestRegressor()
    extra = ExtraTreesRegressor()
    gradient = GradientBoostingRegressor()

    pred_rand = cross_val_predict(random, X, np.ravel(y))
    pred_extra = cross_val_predict(extra, X, np.ravel(y))
    pred_gradient = cross_val_predict(gradient, X, np.ravel(y))
    err_rand = mean_squared_error(pred_rand, y)
    err_extra = mean_squared_error(pred_extra, y)
    err_gradient = mean_squared_error(pred_gradient, y)
    somma = err_extra + err_rand + err_gradient
    w_rand = err_rand / somma
    w_extra = err_extra / somma
    w_gradient = err_gradient / somma

    pesi = {"rand": w_rand, "extra": w_extra, "gradient": w_gradient}
    return pesi

"""
La funzione più importante, in quanto l'SVR è più sensibile ai parametri.
Per dimostrare che la scelta dei valori è ottimale, ogni volta che calibro questi parametri,
grafico nella cartella del sensore la curva dell'RMSE al variare dei parametri gamma e epsilon.
Si nota come la curva abbia effettivamente un minimo.
"""
def tuning_parametri_SVR(data, pollutant, dir):
    print("\nTuning pesi per SVR")

    scaler = StandardScaler()

    y = pollutant
    X = data.copy()

    X = scaler.fit_transform(X)
    y = scaler.fit_transform(np.array(y).reshape(-1, 1))
    m = np.mean(y)
    s = np.std(y)
    c_1 = np.abs(m + 3 * s)
    c_2 = np.abs(m - 3 * s)
    if c_1 > c_2:
        c = c_1
    else:
        c = c_2
    gamma = np.array([0.000030518, 0.00012207, 0.000488281, 0.001953125, 0.0078125, 0.03125, 0.125, 0.5, 2, 4],
                     dtype=float)
    MSE = []
    for i in np.arange(gamma.size):
        svr = SVR(C=c, gamma=gamma[i], cache_size=2000)
        predict = cross_val_predict(svr, X, np.ravel(y))
        #print("Param\n")
        #print("Gamma:", gamma[i])
        #print(svr.get_params)
        MSE.append(mean_squared_error(scaler.inverse_transform(np.array(predict).reshape(-1, 1)),
                                      scaler.inverse_transform(np.array(y).reshape(-1, 1))))
        #print("RMSE:", np.sqrt(MSE[i]))
        #print("\n")
    MSE = pd.Series(MSE)

    plt.figure(figsize=(10, 5))
    plt.xticks(np.arange(0, gamma.size), gamma, fontsize='small')
    plt.xlabel("Valore parametro")
    plt.ylabel("MSE")
    plt.plot(MSE.values, "g", label="prediction", linewidth=2.0)
    plt.title("Parametro Gamma")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(dir, str(pollutant.name) + '_gamma.png'), dpi=96)

    g = gamma[MSE.idxmin()]
    #print(g)
    #print("------------------------------")

    epsilon = np.array([0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    MSE = []
    for i in np.arange(epsilon.size):
        svr = SVR(C=c, gamma=g, epsilon=epsilon[i], cache_size=2000)
        predict = cross_val_predict(svr, X, np.ravel(y))
        #print("Param\n")
        #print("Epsilon:", epsilon[i])
        #print(svr.get_params)
        MSE.append(mean_squared_error(scaler.inverse_transform(np.array(predict).reshape(-1, 1)),
                                      scaler.inverse_transform(np.array(y).reshape(-1, 1))))
        #print("RMSE:", np.sqrt(MSE[i]))
        #print("\n")
    MSE = pd.Series(MSE)

    plt.figure(figsize=(10, 5))
    plt.xticks(np.arange(0, epsilon.size), epsilon, fontsize='small')
    plt.xlabel("Valore parametro")
    plt.ylabel("MSE")
    plt.plot(MSE.values, "g", label="prediction", linewidth=2.0)
    plt.title("Parametro Epsilon")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(dir, str(pollutant.name) + '_epsilon.png'), dpi=96)

    epsilon = epsilon[MSE.idxmin()]
    #print(epsilon)
    #print("------------------------------")

    best_param = {"C": c, "gamma": g, "epsilon": epsilon}
    return best_param
"""
In questa funzione creo due diversi dataset per il training: uno contiene l'intero dataset, mentre il
secondo contiene soltanto gli elementi che hanno NUMBER_OF_PREVIOUS_OBSERVATIONS osservazioni precedenti, ovvero non ci sono buchi temporali.
Empiricamente si è notato come la NUMBER_OF_PREVIOUS_OBSERVATION migliore per questi regressori sia generalmente 1, e come dia migliori risultati
in termini di feature importance e di errore. 
"""

def calibration_VR_SVR_all(info_dictionary,X,y,param_svr,pesi_vr):

    max_train = X.max()
    min_train = X.min()
    X['phenomenon_time']=X.index
    y['phenomenon_time']=y.index
    number_of_previous_observations=1
    pollutant=info_dictionary['pollutant_label']
    freq=int(info_dictionary['interval'][:-1])

    id_sensor=info_dictionary['id_sensor']

    X_train_no_number_of_previous_observations= X.sort_index()
    X_train_number_of_previous_observations = X_train_no_number_of_previous_observations.copy()
    y_train = y.sort_index()

    min_max={"max_train":max_train.to_dict(),
              "min_train":min_train.to_dict()}



    X_train_number_of_previous_observations['number_of_previous_observations'] = X_train_no_number_of_previous_observations.phenomenon_time - X_train_no_number_of_previous_observations.phenomenon_time.shift(number_of_previous_observations)
    for i in np.arange(1, number_of_previous_observations + 1):
        if pollutant == "o3":
            X_train_number_of_previous_observations['ox_aux_{}'.format(i)] = X_train_no_number_of_previous_observations['ox_aux'].shift(i)
            X_train_number_of_previous_observations['ox_we_{}'.format(i)] = X_train_no_number_of_previous_observations['ox_we'].shift(i)
        else:
            X_train_number_of_previous_observations[pollutant + '_aux_{}'.format(i)] = X_train_no_number_of_previous_observations[pollutant + '_aux'].shift(i)
            X_train_number_of_previous_observations[pollutant + '_we_{}'.format(i)] = X_train_no_number_of_previous_observations[pollutant + '_we'].shift(i)
    X_train_number_of_previous_observations['number_of_previous_observations'] = X_train_number_of_previous_observations['number_of_previous_observations'].dt.total_seconds() / 60
    X_train_number_of_previous_observations = X_train_number_of_previous_observations[X_train_number_of_previous_observations['number_of_previous_observations'] == (freq * number_of_previous_observations)]

    X_train_no_number_of_previous_observations = X_train_no_number_of_previous_observations.drop(['phenomenon_time'], axis=1)
    X_train_number_of_previous_observations = X_train_number_of_previous_observations.drop(['phenomenon_time','number_of_previous_observations'], axis=1)


    y_train_number_of_previous_observations = y_train.copy()

    y_train_number_of_previous_observations['number_of_previous_observations'] = y_train_number_of_previous_observations.phenomenon_time - y_train_number_of_previous_observations.phenomenon_time.shift(number_of_previous_observations)
    y_train_number_of_previous_observations['number_of_previous_observations'] = y_train_number_of_previous_observations['number_of_previous_observations'].dt.total_seconds() / 60
    y_train_number_of_previous_observations = y_train_number_of_previous_observations[y_train_number_of_previous_observations['number_of_previous_observations'] == (freq * number_of_previous_observations)]
    y_train = y_train.drop(['phenomenon_time'], axis=1)
    y_train_number_of_previous_observations = y_train_number_of_previous_observations.drop(['phenomenon_time','number_of_previous_observations'], axis=1)

    scaler_1 = StandardScaler()
    scaler_2 = StandardScaler()
    svr = SVR(cache_size=2000)
    svr.set_params(**param_svr)

    X_train_scaled = scaler_1.fit_transform(X_train_no_number_of_previous_observations)
    y_train_scaled = scaler_2.fit_transform(y_train)
    svr.fit(X_train_scaled, np.ravel(y_train_scaled))

    r_no_number_of_previous_observations = RandomForestRegressor()
    r_number_of_previous_observations = RandomForestRegressor()
    e_no_number_of_previous_observations = ExtraTreesRegressor()
    e_number_of_previous_observations = ExtraTreesRegressor()
    g_no_number_of_previous_observations = GradientBoostingRegressor()
    g_number_of_previous_observations = GradientBoostingRegressor()

    model_no_number_of_previous_observations = VotingRegressor([('Random', r_no_number_of_previous_observations), ('Gradient', g_no_number_of_previous_observations), ('Extra', e_no_number_of_previous_observations)],
                              weights=(pesi_vr['rand'], pesi_vr['gradient'], pesi_vr['extra']))
    model_no_number_of_previous_observations.fit(X_train_no_number_of_previous_observations, np.ravel(y_train))

    model_number_of_previous_observations = VotingRegressor([('Random', r_number_of_previous_observations), ('Gradient', g_number_of_previous_observations),
                               ('Extra', e_number_of_previous_observations)],
                              weights=(pesi_vr['rand'], pesi_vr['gradient'], pesi_vr['extra']))
    model_number_of_previous_observations.fit(X_train_number_of_previous_observations, np.ravel(y_train_number_of_previous_observations))

    return model_no_number_of_previous_observations,model_number_of_previous_observations,svr, scaler_1,scaler_2, min_max