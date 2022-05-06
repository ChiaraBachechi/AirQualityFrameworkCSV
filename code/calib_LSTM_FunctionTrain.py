import os
import tensorflow 
from tensorflow.random import set_seed
set_seed(2)
import datetime
import pandas as pd
import dill
import numpy as np
from numpy.random import seed
seed(1)
import psycopg2
import sklearn
import matplotlib.pyplot as plt
import json
import datetime  # this is to pring the current time during the run
import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import *
from calibrationAlgorithmTrainer_interfaces import *


class Calib_LSTM_FunctionTrainer_001(CalibartionAlgorithmTrainer_interface):

    def init(self, info_dictionary):
        self.info_dictionary = info_dictionary
        self.calibrator = None
    def getCalibrator(self):
        return self.calibrator

    def doTrain(self, df_train_features, df_train_labels_full):
        #df_train_labels_full = df_train_labels_full.rename(
        #    columns={df_train_labels_full.columns[0]: self.info_dictionary['pollutant_label']})
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
        calibrator,scaler_feat,scaler_pollutant=calibration_lstm_all(self.info_dictionary,df_train_features,df_train_labels)
        scaler={"scaler_feat":scaler_feat,
                "scaler_pollutant":scaler_pollutant}
        self.calibrator = Calib_LSTM_Function()
        self.calibrator.init(self.info_dictionary,scaler)
        calibrator.save("tmp")
        return self.calibrator


class Calib_LSTM_Function(CalibartionAlgorithm_interface):
    """
    the calibrator produced by the trainer - example
    """

    def init(self, info_dictionary,scaler):
        super().init(info_dictionary)
        info = self.info_dictionary
        self.scaler=scaler
        #

    def apply_df(self, data_frame_in,interval,path_dill):

        # print(" ---- data_frame_in: " + str(data_frame_in))
        dataset_feature = data_frame_in[self.info_dictionary['feat_order']]

        n_feature=dataset_feature.shape[1]
        dataset_na = dataset_feature.dropna()
        model=tensorflow.keras.models.load_model(
            path_dill)


        # need to add a test for outside of the bounds. they should be percent
        if dataset_na.empty:
            return np.nan
        else:
            # this apply doesn't work for o3 if it doesn't contain NO2 raw features in it's calibration
            # for NO, NO2, and CO the RF results are in ug/m3 directly.
            dataset_feature['phenomenon_time'] = pd.to_datetime(data_frame_in['phenomenon_time'])
            dataset_feature.index=dataset_feature['phenomenon_time']
            X=split_calib_data(dataset_feature,self.info_dictionary["number_of_previous_observations"], interval, self.info_dictionary['feat_order'])
            pred = pd.DataFrame()
            pred['phenomenon_time']=X.index
            X=self.scaler["scaler_feat"].transform(X)
            X = X.reshape(X.shape[0], self.info_dictionary["number_of_previous_observations"] + 1, n_feature)
            yhat = model.predict(X, verbose=0)
            yhat = np.abs(self.scaler["scaler_pollutant"].inverse_transform(yhat))
            pred[self.info_dictionary["target_label"]]=yhat
        return pred

    #
    #
    @abstractfunc
    def get_info(self):
        return (self.info_dictionary)



def updateInfoData(info_dictionary, df_train_features, df_train_labels_full):
    df_train_labels = df_train_labels_full[info_dictionary['target_label']]
    #
    # print(str(df_train_features))
    # print("--- labels ---")
    # print(str(df_train_labels))
    #
    #if ('hyper_parameters' in info_dictionary['algorithm_parameters']):
    #    d = info_dictionary['algorithm_parameters']['hyper_parameters']
    #    trees = d['trees']
    #    leaves = d['leaves']
    #info_dictionary['algorithm_parameters']['hyper_parameters'] = \
    #    {'trees': trees, 'leaves': leaves}
    info_dictionary['name']="LSTM"
    info_dictionary["number_of_previous_observations"]=info_dictionary['number_of_previous_observations']
    info_dictionary["python_env"] = {'sklearn': sklearn.__version__,
                                     'pandas': pd.__version__,
                                     'dill': dill.__version__,
                                     'numpy': np.__version__,
                                     'psycopg2': psycopg2.__version__,
                                     'tensorflow': tensorflow.__version__,
                                     'keras': keras.__version__}
    info_dictionary["features"] = {}
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
    
    info_dictionary[info_dictionary['pollutant_label']] = {}
    info_dictionary[info_dictionary['pollutant_label']]['range'] = ['%.2f' % float(df_train_labels.min()), '%.2f' % float(df_train_labels.max())]
    [min(df_train_labels), max(df_train_labels)]
    info_dictionary[info_dictionary['pollutant_label']]['unit_of_measure'] = info_dictionary['units_of_measure'][info_dictionary['pollutant_label']]['unit_of_measure']


"""
Funzioni di prova
"""

"""
Funzione necessaria per rendere compatibili i dati con la struttura richiesta da keras
"""
def split_data(X,y,n_steps,freq_sampling, feat_list):
    y['phenomenon_time']=X['phenomenon_time'].copy()
    X=X.sort_index()
    y=y.sort_index()
    for j in np.arange(1,n_steps+1):
        for feat in feat_list:
            X[feat+"_{}".format(j)]=X[feat].shift(-j)

    X['number_of_previous_observations'] = X.phenomenon_time.shift(-n_steps)-X.phenomenon_time
    X['number_of_previous_observations'] = X['number_of_previous_observations'].dt.total_seconds() / 60
    X['number_of_previous_observations']=X['number_of_previous_observations'].fillna(0)
    X.index=X.index.shift(n_steps,freq=freq_sampling)

    X = X[X['number_of_previous_observations'].astype(int) == (int(freq_sampling[:-1]) * n_steps)]
    if X.shape[0]==0:
        print("Samples not found in this sliding window")
        quit()
    X= X.drop(['phenomenon_time','number_of_previous_observations'], axis=1)

    y['number_of_previous_observations'] = y.phenomenon_time - y.phenomenon_time.shift(n_steps)
    y['number_of_previous_observations'] = y['number_of_previous_observations'].dt.total_seconds() / 60
    y['number_of_previous_observations']=y['number_of_previous_observations'].fillna(0)

    y = y[y['number_of_previous_observations'].astype(int) == (int(freq_sampling[:-1]) * n_steps)]
    y = y.drop(['phenomenon_time','number_of_previous_observations'], axis=1)
    return X,y
def split_calib_data(X,n_steps,freq_sampling, feat_list):
    #dataset_feature,self.info_dictionary["number_of_previous_observations"], interval, self.info_dictionary['feat_order']
    X=X.sort_index()
    for j in np.arange(1,n_steps+1):
        for feat in feat_list:
            X[feat+"_{}".format(j)]=X[feat].shift(-j)


    X['number_of_previous_observations'] = X.phenomenon_time.shift(-n_steps)-X.phenomenon_time
    X['number_of_previous_observations'] = X['number_of_previous_observations'].dt.total_seconds() / 60
    X['number_of_previous_observations'] = X['number_of_previous_observations'].fillna(0)


    X.index=X.index.shift(n_steps,freq=freq_sampling)
    X = X[X['number_of_previous_observations'].astype(int) == (int(freq_sampling[:-1]) * n_steps)]

    if X.shape[0]==0:
        print("Samples not found in this sliding window")
    X= X.drop(['phenomenon_time','number_of_previous_observations'], axis=1)
    return X

def calibration_lstm_all(info_dictionary,X,y,validation = False):
    number_of_previous_observations=info_dictionary['number_of_previous_observations']
    algorithm_parameters  = info_dictionary['algorithm_parameters']
    if 'learning_rate' not in algorithm_parameters.keys():
        learning_rate = 0.005
        info_dictionary['algorithm_parameters']['learning_rate']=0.005
    else:
        learning_rate = algorithm_parameters['learning_rate']
    if 'epochs' not in algorithm_parameters.keys():
        epochs = 200
        info_dictionary['algorithm_parameters']['epochs']=200
    else:
        epochs = algorithm_parameters['epochs']
    if 'batch' not in algorithm_parameters.keys():
        batch = 32
        info_dictionary['algorithm_parameters']['batch']=32
    else:
        batch = algorithm_parameters['batch']
    freq=info_dictionary['interval']
    #id_sensor=info_dictionary['id_sensor']
    feat_list=info_dictionary['feat_order']
    X['phenomenon_time']=X.index
    n_features=X.shape[1]-1
    y=pd.DataFrame(y)
    X,y=split_data(X,y,number_of_previous_observations,freq,feat_list)

    scaler_feat=MinMaxScaler()
    scaler_pollutant = MinMaxScaler()
    X=scaler_feat.fit_transform(X)
    X=X.reshape(X.shape[0],number_of_previous_observations+1,n_features)
    y=scaler_pollutant.fit_transform(np.array(y).reshape(-1,1))
    X_train = X
    y_train = y
    n_neurons = np.floor(X_train.shape[0] / (2 * (X_train.shape[1] + 1)))

    model = Sequential()
    """
    model.add(LSTM(int(n_neurons*0.9), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    """


    model.add(LSTM(int(n_neurons*0.8), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(int(n_neurons*0.5), activation='relu',return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(int(n_neurons * 0.2), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate= learning_rate),loss='mean_squared_error')

    # Fitting to the training set
    #history = model.fit(X_train, y_train, epochs=200, validation_data=(X_val,y_val),batch_size=32)
    history = model.fit(X_train, y_train, epochs = epochs,batch_size= batch,shuffle=False)
    if(validation):
        history = model.fit(X_train, y_train, epochs = epochs, validation_split=0.1,batch_size= batch)
        print(history.history.keys())
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig('../results/' + info_dictionary["dill_file_name"] + "_loss.png")
       #prediction=model.predict(X_val)
       #rmse=np.sqrt(mean_squared_error(y_val,prediction))

       #if info_dictionary["pollutant_label"] == "o3":
       #    class_pred = range_EEA_O3(prediction)
       #    class_test = range_EEA_O3(y_val)
       #elif info_dictionary["pollutant_label"] == "co":
       #    class_pred = range_EEA_CO(prediction)
       #    class_test = range_EEA_CO(y_val)
       #else:
       #    class_pred = range_EEA_NOx(prediction)
       #    class_test = range_EEA_NOx(y_val)

       #acc_EEA = accuracy_score(class_pred, class_test)
       #info_dictionary["accuracy_EEA"]=acc_EEA
       #info_dictionary["RMSE"] = rmse

       #if info_dictionary["pollutant_label"] == "o3":
       #    class_pred = range_ARPA_O3(prediction)
       #    class_test = range_ARPA_O3(y_val)
       #elif info_dictionary["pollutant_label"] == "co":
       #    class_pred = range_ARPA_CO(prediction)
       #    class_test = range_ARPA_CO(y_val)
       #else:
       #    class_pred = range_ARPA_NOx(prediction)
       #    class_test = range_ARPA_NOx(y_val)

       #acc_ARPA = accuracy_score(class_pred, class_test)
       #info_dictionary["accuracy_ARPA"]=acc_ARPA

    info_dictionary["hyper_parameters"]={"number_of_neurons":n_neurons}
    return model, scaler_feat,scaler_pollutant
