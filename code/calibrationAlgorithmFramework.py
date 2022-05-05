import ast
import os
from glob import glob
import numpy as np
import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
import sys
import importlib
import json
import sklearn

import calibrationAlgorithmTrainer_interfaces

sys.path.insert(0, '../..')
from trafair_db_config import trafair_db_getConnection

# sys.path.insert(0, 'calibrators')


"""
 A tool library for managing calibration
 requirements
   pip3 install numpy pandas
"""

class CalibrationAlgorithmFramework():
    #
    # class variables
    #
    status = None;
    #
    # constructor
    #
    def __init__(self):
        self.conn = None
        self.df_testAndTraining={}
        self.calibrator=None
        #

    def __str__(self):
        status = self.getStatus()
        str_to_print = json.dumps(status, sort_keys=True, indent=2)
        return str_to_print

    def initFromInfo(self, info):
      self.begin_time = info["dates"]["start"]
      self.end_time = info["dates"]["end"]
      self.id_sensor = info["id_sensor"]
      self.features = info["feat_order"]
      self.target_label = info['target_label']
      self.trainer_module_name = info["trainer_module_name"]
      self.trainer_class_name = info["trainer_class_name"]
      self.test_size          = info['test_size']
      self.pollutant_label = info['pollutant_label']
      self.interval = info['interval']
      self.number_of_previous_observations=info['number_of_previous_observations']
      self.algorithm_parameters = (info['algorithm_parameters'] if 'algorithm_parameters' in info else {})
      self.dill_file_name=info["dill_file_name"]
      self.units_of_measure = info['units_of_measure']
      self.csv_feature_data = info['csv_feature_data']
      self.csv_target_data = info['csv_target_data']
      self.label_list = info['label_list']
      
    def getStatus(self):
        status={}
        status["dates"]={ 'start': self.begin_time, 'end': self.end_time }
        status['id_sensor']=self.id_sensor
        status['feat_order']=self.features
        status['label_list']=self.label_list
        status['target_label']=self.target_label
        status['csv_feature_data']=self.csv_feature_data
        status['csv_target_data'] = self.csv_target_data
        status['trainer_module_name']=self.trainer_module_name
        status['trainer_class_name']=self.trainer_class_name
        status['interval']=self.interval
        status['test_size']=self.test_size
        status['pollutant_label']=self.pollutant_label
        status['interval']=self.interval
        status['number_of_previous_observations']=self.number_of_previous_observations
        status['algorithm_parameters']=self.algorithm_parameters
        status['units_of_measure']=self.units_of_measure
        status['dill_file_name']=self.dill_file_name
        return status

    def getConnection(self):
      if (self.conn == None):
        self.conn = trafair_db_getConnection()
      return(self.conn)
      
    def get_df_train_features(self):
        train_features = self.df_testAndTraining['train_features']
        return (train_features)
    def get_df_train_labels(self):
        train_labels = self.df_testAndTraining['train_labels']
        return (train_labels)
    def get_df_test_features(self):
        rv = None
        if 'test_features' in self.df_testAndTraining:
            rv = self.df_testAndTraining['test_features']
        return (rv)
    def get_df_test_labels(self):
        test_labels = self.df_testAndTraining['test_labels'];
        return (test_labels)
        
    def instantiateTrainingClass(self):
      module = importlib.import_module(self.trainer_module_name)
      class_ = getattr(module, self.trainer_class_name)
      instance = class_()
      return(instance)
 
    def trainCalibrator(self):
        trainer = self.instantiateTrainingClass()
        info = self.getStatus()
        trainer.init(info)
        self.calibrator = trainer.doTrain(self.get_df_train_features(),self.get_df_train_labels())
        
    def getIntervalInMinutesFromString(self, interval_string):
        interval=None
        if interval_string.find('T')>0:
            interval=interval_string.split('T')[0]
        else:#there is an H
            interval=str(int(interval_string.split('H')[0])*60)
        return(interval)

    def getCalibrator(self):
        return(self.calibrator)
        
    def loadDatasetFromDb(self):
        """
        query the DB and return a dataframes with all data available in the db
        """
        strFeature=""
        comma=""
        for f in self.features:
            strFeature=strFeature+comma+'avg('+str(f)+') as '+str(f)
            comma=', '
        strLabelAvg=""
        strLabelList=""
        comma=""
        for f in self.label_list:
            strLabelAvg=strLabelAvg+comma+'avg('+str(f)+') as label_'+str(f)
            strLabelList=strLabelList+comma+'station.label_'+str(f)
            comma=", "
        # print(" -- strLabelAvg: " + strLabelAvg)
        interval=self.getIntervalInMinutesFromString(self.interval)
        q="""
          select sensor.*
               , station.coverage station_coverage
               , %s
           from
             (
             select status.id_sensor_low_cost
                  , (to_timestamp(ceil(extract(epoch 
                                               from phenomenon_time::timestamp with time zone
                                              ) / (60 * %s )
                                       ) * (60 * %s)
                                  )
                     )::timestamp as phenomenon_time_rounded
                  , count(id_sensor_low_cost_status) as coverage
                  , status.id status_id
                  , status.id_sensor_low_cost_feature
                  , %s
                  from sensor_raw_observation as raw
                     ,(select ss1.id
                           , ss1.id_sensor_low_cost
                           , ss1.status
                           , ss1.id_sensor_low_cost_feature
                           , ss1.operator
                           , ss1.datetime
                           , (select ss2.datetime
                                from sensor_low_cost_status ss2
                               where ss1.id_sensor_low_cost = ss2.id_sensor_low_cost
                                 and ss1.datetime < ss2.datetime
                                order by ss2.datetime
                                limit 1
                             ) as datetime_end
                        from sensor_low_cost_status ss1
                         order by ss1.id_sensor_low_cost, ss1.datetime
                       ) as status
                  where status.id_sensor_low_cost = %s
                    and raw.id_sensor_low_cost_status = status.id
                    and phenomenon_time >= '%s'
                    and phenomenon_time  < '%s'
                    and (status.status = 'calibration')
                    and (phenomenon_time >= status.datetime
                         and (phenomenon_time < status.datetime_end or status.datetime_end is null ))
                  group by(status.id_sensor_low_cost,status.id_sensor_low_cost_feature
                           , phenomenon_time_rounded, status.id)
                  order by phenomenon_time_rounded
              ) as sensor
              , (
                 select (to_timestamp(ceil(extract(epoch
                                            from phenomenon_time::timestamp with time zone
                                        ) / (60 * %s )) * (60 * %s))
                        )::timestamp as phenomenon_time_rounded
                    , id_aq_legal_station
                    , count(raw.phenomenon_time) as coverage
                    , %s
                  from 
                    aq_legal_station_observation_one_minute_not_validated as raw
                  where phenomenon_time >= '%s'
                    and phenomenon_time  < '%s'
                  group by(phenomenon_time_rounded, id_aq_legal_station)
                  order by phenomenon_time_rounded
               ) station
              , sensor_low_cost_feature feature
           where sensor.phenomenon_time_rounded = station.phenomenon_time_rounded
             and sensor.id_sensor_low_cost_feature =  feature.id
             and feature.id_aq_legal_station = station.id_aq_legal_station
           order by phenomenon_time_rounded
        """%(strLabelList
             , interval
             , interval
             , strFeature
             , self.id_sensor
             , self.begin_time
             , self.end_time
             , interval
             , interval
             , strLabelAvg
             , self.begin_time
             , self.end_time)
        #
        # print( " ---the query---\n" + q)
        conn = self.getConnection()
        df_station_and_raw_resampled=sqlio.read_sql_query(q, conn)
        #
        df_station_and_raw_resampled.set_index('phenomenon_time_rounded',inplace=True)
        #
        #
        cutoff_Value=0.1
        df_trainig_and_testing=df_station_and_raw_resampled.astype('float')
        for pollutant_label in self.label_list:
            label_name = 'label_'+str(pollutant_label)
            a=0
            check=df_trainig_and_testing[label_name]
            #cur = conn.cursor()
            for current, next in zip(check, check[1:]): # this loop removes constant values.
                if current==next:
                    check[a]=np.nan
                a=a+1    
            check=[float('nan') if x<cutoff_Value else x for x in check]
            df_trainig_and_testing[label_name]=check.copy()
            df_trainig_and_testing.dropna(inplace=True)
            #cur.close()
        return(df_trainig_and_testing)
        
    def createTrainingAndTestingDBRepairing(self):
        """
        query the DB and creates dataframes without anomalies for training and testing 
        """
        df_trainig_and_testing = self.loadDatasetFromDbRepairing()
        df_training_X=df_trainig_and_testing[self.features]
        #
        tmp_label_list=[]
        for l in self.label_list:
            tmp_label_list.append('label_'+str(l))
        #print(" -- tmp_label_list: " + str(tmp_label_list));
        df_training_Y=df_trainig_and_testing[tmp_label_list]
        df_training_Y.columns = self.label_list;
        # print(" -- df_training_Y: " + str(df_training_Y));
        #from sklearn.model_selection import train_test_split
        #train_features, test_features, train_labels, test_labels = \
         #   train_test_split(df_training_X, df_training_Y,test_size = self.test_size)
        #print (" test_labels columns: " + str(test_labels.columns))
        #print(" --- test_labels type: ---------- "+type(test_labels).__name__+"")
        #print(" --- test_features type: ---------- "+type(test_labels).__name__+"")
        #print (" test_labels: " + str(test_labels))
        #print (" test_labels columns: " + str(test_labels.columns))
        self.df_testAndTraining['train_features']=df_training_X
        self.df_testAndTraining['train_labels']=df_training_Y
        #self.df_testAndTraining['test_features']=test_features
        #self.df_testAndTraining['test_labels']=test_labels
        return self.df_testAndTraining

    def loadDatasetFromDbRepairing(self):
            """
            query the DB and return a dataframes with all data available in the db
            """
            strFeature=""
            comma=""
            for f in self.features:
                rstr = str(f)
                if str(f) == 'ox_we' or str(f) == 'ox_aux':
                    rstr = 'o3_' + rstr.split('_')[1]
                strFeature = strFeature + comma + 'coalesce(a.'+str(f)+', r.' + rstr + ')' + ' as ' + str(f)
                comma=', '
            strLabelAvg=""
            strLabelList=""
            comma=""
            for f in self.label_list:
                strLabelAvg=strLabelAvg+comma+'avg('+str(f)+') as label_'+str(f)
                strLabelList=strLabelList+comma+'station.label_'+str(f)
                comma=", "
            # print(" -- strLabelAvg: " + strLabelAvg)
            interval=self.getIntervalInMinutesFromString(self.interval)
            q="""
              select ss.id_sensor_low_cost, a.phenomenon_time_sensor_raw_observation_10min as phenomenon_time_rounded,
              ss.id as status_id, ss.id_sensor_low_cost_feature, %s
                   , station.coverage station_coverage
                   , %s
               from (aggregated_raw_observation_without_anomaly as a left join
               sensor_low_cost_status as ss on ss.id = a.id_sensor_low_cost_status) join
    (
                     select (to_timestamp(ceil(extract(epoch
                                                from phenomenon_time::timestamp with time zone
                                            ) / (60 * %s )) * (60 * %s))
                            )::timestamp as phenomenon_time_rounded
                        , id_aq_legal_station
                        , count(raw.phenomenon_time) as coverage
                        , %s
                      from 
                        aq_legal_station_observation_one_minute_not_validated as raw
                      where phenomenon_time >= '%s'
                        and phenomenon_time  < '%s'
                      group by(phenomenon_time_rounded, id_aq_legal_station)
                      order by phenomenon_time_rounded
                   ) as station on station.phenomenon_time_rounded = a.phenomenon_time_sensor_raw_observation_10min left 
                   join repaired_anomaly as r on r.id_sensor_low_cost_status = a.id_sensor_low_cost_status 
                   and r.phenomenon_time_sensor_raw_observation_10min = a.phenomenon_time_sensor_raw_observation_10min
                   where ss.id_sensor_low_cost = %s
    order by phenomenon_time_rounded
            """%(strFeature
                 , strLabelList
                 , interval
                 , interval 
                 , strLabelAvg
                 , self.begin_time
                 , self.end_time
                 , self.id_sensor)
            #
            print( " ---the query---\n" + q)
            conn = self.getConnection()
            df_station_and_raw_resampled=sqlio.read_sql_query(q, conn)
            #
            df_station_and_raw_resampled.set_index('phenomenon_time_rounded',inplace=True)
            print(df_station_and_raw_resampled.columns)
            #
            cutoff_Value=0.1
            df_trainig_and_testing=df_station_and_raw_resampled.astype('float')
            for pollutant_label in self.label_list:
                label_name = 'label_'+str(pollutant_label)
                a=0
                check=df_trainig_and_testing[label_name]
                cur = conn.cursor()
                for current, next in zip(check, check[1:]): # this loop removes constant values.
                    if current==next:
                        check[a]=np.nan
                    a=a+1    
                check=[float('nan') if x<cutoff_Value else x for x in check]
                df_trainig_and_testing[label_name]=check.copy()
                df_trainig_and_testing.dropna(inplace=True)
                cur.close()
            return(df_trainig_and_testing)

    def createTrainingDB(self, anomaly = False):
        df_trainig_and_testing = self.loadDatasetFromDb()
        if (anomaly):
            df_trainig_and_testing = self.loadDatasetFromDbAnomalyDetection()
        self.df_testAndTraining['train_features'] = df_trainig_and_testing[self.features]
        tmp_label_list = []
        for l in self.label_list:
            tmp_label_list.append('label_'+str(l))
        #print(" -- tmp_label_list: " + str(tmp_label_list));
        self.df_testAndTraining['train_labels'] = df_trainig_and_testing[tmp_label_list]
        #self.df_testAndTraining['train_labels'] = self.label_list;
        return self.df_testAndTraining
    def createAndSaveDataCleaningDB(self, df_csv_file_prefix):
        strFeature=""
        comma=""
        # print(" -- strLabelAvg: " + strLabelAvg)
        q="""
          select sensor_raw_observation.*, feature.id_aq_legal_station
           from sensor_raw_observation,
           (
           select ss1.id
                           , ss1.id_sensor_low_cost
                           , ss1.status
                           , ss1.id_sensor_low_cost_feature
                           , ss1.operator
                           , ss1.datetime
                           , (select ss2.datetime
                                from sensor_low_cost_status ss2
                               where ss1.id_sensor_low_cost = ss2.id_sensor_low_cost
                                 and ss1.datetime < ss2.datetime
                                order by ss2.datetime
                                limit 1
                             ) as datetime_end
                        from sensor_low_cost_status ss1
                         order by ss1.id_sensor_low_cost, ss1.datetime
                       ) as status
                       , sensor_low_cost_feature feature
           where id_sensor_low_cost_feature =  feature.id
           and status.id_sensor_low_cost = %s
                    and id_sensor_low_cost_status = status.id
                    and phenomenon_time >= '%s'
                    and phenomenon_time  < '%s'
                    and (status.status = 'calibration')
                    and (phenomenon_time >= status.datetime
                    and (phenomenon_time < status.datetime_end or status.datetime_end is null ))
                    and feature.id_aq_legal_station is not NULL
           order by phenomenon_time
        """%( self.id_sensor
             , self.begin_time
             , self.end_time)
        #
        # print( " ---the query---\n" + q)
        conn = self.getConnection()
        df_raw_observations=sqlio.read_sql_query(q, conn)
        df_raw_observations.to_csv(df_csv_file_prefix+'_not-filtered_raw.csv')
        strLabel=''
        for f in self.label_list:
            strLabel=strLabel+','+str(f)
        
        q="""
        select phenomenon_time,id_aq_legal_station%s from aq_legal_station_observation_one_minute_not_validated 
        where phenomenon_time >= '%s'
        and phenomenon_time  < '%s'
        """%(strLabel,self.begin_time
             , self.end_time)
        df_aq_ls_data=sqlio.read_sql_query(q, conn)
        df_aq_ls_data.to_csv(df_csv_file_prefix+'_not-filtered_legalStation.csv')
        return
    def createTrainingAndTestingDB(self, anomaly = False):
        """
        query the DB and creates dataframes for training and testing
        """
        df_trainig_and_testing = self.loadDatasetFromDb()
        if(anomaly):
            df_trainig_and_testing = self.loadDatasetFromDbAnomalyDetection()
        df_training_X=df_trainig_and_testing[self.features]
        #
        tmp_label_list=[]
        for l in self.label_list:
            tmp_label_list.append('label_'+str(l))
        #print(" -- tmp_label_list: " + str(tmp_label_list));
        df_training_Y=df_trainig_and_testing[tmp_label_list]
        df_training_Y.columns = self.label_list;
        # print(" -- df_training_Y: " + str(df_training_Y));
        #from sklearn.model_selection import train_test_split
        #train_features, test_features, train_labels, test_labels = \
        #    train_test_split(df_training_X, df_training_Y,test_size = self.test_size)
        #print (" test_labels columns: " + str(test_labels.columns))
        #print(" --- test_labels type: ---------- "+type(test_labels).__name__+"")
        #print(" --- test_features type: ---------- "+type(test_labels).__name__+"")
        #print (" test_labels: " + str(test_labels))
        #print (" test_labels columns: " + str(test_labels.columns))
        self.df_testAndTraining['train_features']=df_training_X
        self.df_testAndTraining['train_labels']=df_training_Y
        #self.df_testAndTraining['test_features']=test_features
        #self.df_testAndTraining['test_labels']=test_labels
        return self.df_testAndTraining

    # def getRange(self):
    #     ranges={}
    #     for c in self.df_testAndTraining['train_features'].columns:
    #         ranges[c]=(self.df_testAndTraining['train_features'][c].min(axis=0),self.df_testAndTraining['train_features'][c].max(axis=0))
    #     for c in self.df_testAndTraining['train_labels'].columns:
    #         ranges[c]=(self.df_testAndTraining['train_labels'][c].min(axis=0),self.df_testAndTraining['train_labels'][c].max(axis=0))
    #     return ranges

    def saveTrainingAndTestingDataToCsv(self, df_csv_file_prefix):
        fn_train_features = df_csv_file_prefix + "_train_features.csv"
        fn_train_labels   = df_csv_file_prefix + "_train_labels.csv"
        self.get_df_train_features().to_csv(fn_train_features,header=True)
        self.get_df_train_labels().to_csv(fn_train_labels,header=True)
        saved="data saved to "+fn_train_features+", " + fn_train_labels
        if(not (self.get_df_test_features() is None)):
            fn_test_features = df_csv_file_prefix + "_test_features.csv"
            fn_test_labels   = df_csv_file_prefix + "_test_labels.csv"
            self.get_df_test_features().to_csv(fn_test_features,header=True)
            self.get_df_test_labels().to_csv(fn_test_labels,header=True)
            saved=saved + ", "+fn_test_features+", " + fn_test_labels
        #
        print("\n "+ saved + "\n")



    def loadTrainingAndTestingDataFromCsv(self, df_csv_file_prefix):
        fn_train_features = df_csv_file_prefix + "_train_features.csv"
        fn_train_labels   = df_csv_file_prefix + "_train_labels.csv"
        fn_test_features = df_csv_file_prefix + "_test_features.csv"
        fn_test_labels   = df_csv_file_prefix + "_test_labels.csv"
        #
        print("\n loading data from "+fn_train_features+", " + fn_train_labels \
              + ", "+fn_test_features+", " + fn_test_labels
              + "\n")
        #
        self.df_testAndTraining['train_features']=pd.read_csv(fn_train_features,       index_col='phenomenon_time_rounded') 
        self.df_testAndTraining['train_labels']=pd.read_csv(fn_train_labels, header=0, index_col='phenomenon_time_rounded')
        self.df_testAndTraining['test_features']=pd.read_csv(fn_test_features,         index_col='phenomenon_time_rounded')
        self.df_testAndTraining['test_labels']=pd.read_csv(fn_test_labels, header=0,   index_col='phenomenon_time_rounded')
        #
        #for df in self.df_testAndTraining:
        #    print (' ------ ' + str(df))
        #    df_testAndTraining[df].drop(0,1)
        #

    def insertDillToDB(self,calibrator):
            conn = self.getConnection()
            cur = conn.cursor()
            sql = """INSERT INTO sensor_calibration_algorithm_test(id,model_name,hyper_parameters,training_start,training_end,
            regression_variables,note,python_library,id_aq_legal_station,info)
                     VALUES(DEFAULT ,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id;"""
            data=(
                           str(calibrator.info_dictionary["name"]),
                            str(calibrator.info_dictionary["hyper_parameters"]),
                            self.begin_time,
                            self.end_time,
                            self.features,
                            os.getcwd()+"/dill",
                            str(calibrator.info_dictionary["python_env"]),
                            None,
                            json.dumps(calibrator.info_dictionary)
                        )
            cur.execute(sql,data)
            dill_id=cur.fetchone()[0]
            conn.commit()
            cur.close()
            conn.close()
            return dill_id

    """
    Funzioni aggiuntive
    """
    """
    Questa funzione è sicuramente poco elegante, creo due file temporanei per poi eliminarli.
    Tuttavia, facendo il groupby dava dei problemi per quanto riguarda le colonne, questa è stata
    la soluzione più veloce a cui abbia pensato.
    """
    def loadDatasetFromCSV(self):
        with open('../data/' + self.csv_feature_data, 'r') as csv_file:
            df1 = pd.read_csv(csv_file)
        df1 = df1[self.features + ['phenomenon_time','id_aq_legal_station']]
        with open('../data/' + self.csv_target_data, 'r') as csv_file:
            df2 = pd.read_csv(csv_file)
        df2 = df2[['phenomenon_time', 'id_aq_legal_station', str(self.target_label)]]
        # converto per fare il resample
        df1['phenomenon_time'] = pd.to_datetime(df1['phenomenon_time'])
        df2['phenomenon_time'] = pd.to_datetime(df2['phenomenon_time'])
        # elimino valori nulli

        df1 = df1.dropna()
        df2 = df2.dropna()
        df2 = df2[df2[self.target_label] > 0.1]
        df1= df1.groupby('id_aq_legal_station').resample(str(self.interval), on='phenomenon_time', label='right').mean()
        df2 = df2.groupby('id_aq_legal_station').resample(str(self.interval), on='phenomenon_time',
                                                          label='right').mean()
        df1.to_csv('tmp_raw_' + self.pollutant_label + '.csv')
        df2.to_csv('tmp_legal_' + self.pollutant_label + '.csv')
        df_raw = pd.read_csv('tmp_raw_' + self.pollutant_label + '.csv')
        df_legal = pd.read_csv('tmp_legal_' + self.pollutant_label + '.csv')

        #df_legal = df_legal.drop(df_legal.columns[2], axis=1)
        
        df_raw['phenomenon_time'] = pd.to_datetime(df_raw['phenomenon_time'])
        df_legal['phenomenon_time'] = pd.to_datetime(df_legal['phenomenon_time'])
        df_raw = df_raw.dropna()
        df_legal = df_legal.dropna()
        cutoff_Value=0.1
        a=0
        check=df_legal[self.target_label]
        for current, next in zip(check, check[1:]): # this loop removes constant values.
            if current==next:
                check[a]=np.nan
            a=a+1    
        check=[float('nan') if x < cutoff_Value else x for x in check]
        df_legal[self.target_label]=check.copy()
        df_legal.dropna(inplace=True)
        data = pd.merge(df_raw, df_legal, on=('phenomenon_time', 'id_aq_legal_station'))
        data = data.sort_values(by=['phenomenon_time'])
        data = data.dropna()
        data.set_index('phenomenon_time',inplace=True)
        data = data.astype('float')
        for file in glob('tmp*.*'):
           os.remove(file)
        return data

    def createTrainingAndTestingFromCSV(self):
        df_trainig_and_testing = self.loadDatasetFromCSV()

        self.df_testAndTraining['train_features'] = df_trainig_and_testing[self.features]
        #tmp_label_list = []
        #for l in self.label_list:
        #    tmp_label_list.append('label_' + str(l))
        # print(" -- tmp_label_list: " + str(tmp_label_list));
        self.df_testAndTraining['train_labels'] = df_trainig_and_testing[[self.target_label]]
        # self.df_testAndTraining['train_labels'] = self.label_list;
        return self.df_testAndTraining
    def createTrainingAndTestingDBAnomalyDetection(self):
        """
        query the DB and creates dataframes without anomalies for training and testing 
        """
        df_trainig_and_testing = self.loadDatasetFromDbAnomalyDetection()
        df_training_X=df_trainig_and_testing[self.features]
        #
        tmp_label_list=[]
        for l in self.label_list:
            tmp_label_list.append('label_'+str(l))
        #print(" -- tmp_label_list: " + str(tmp_label_list));
        df_training_Y=df_trainig_and_testing[tmp_label_list]
        df_training_Y.columns = self.label_list;
        # print(" -- df_training_Y: " + str(df_training_Y));
        #from sklearn.model_selection import train_test_split
        #train_features, test_features, train_labels, test_labels = \
        #    train_test_split(df_training_X, df_training_Y,test_size = self.test_size)
        #print (" test_labels columns: " + str(test_labels.columns))
        #print(" --- test_labels type: ---------- "+type(test_labels).__name__+"")
        #print(" --- test_features type: ---------- "+type(test_labels).__name__+"")
        #print (" test_labels: " + str(test_labels))
        #print (" test_labels columns: " + str(test_labels.columns))
        self.df_testAndTraining['train_features']=df_training_X
        self.df_testAndTraining['train_labels']=df_training_Y
        #self.df_testAndTraining['test_features']=test_features
        #self.df_testAndTraining['test_labels']=test_labels
        return self.df_testAndTraining

    def loadDatasetFromDbAnomalyDetection(self):
        """
        query the DB and return a dataframes with all data available in the db without anomalies
        """
        strFeature=""
        comma=""
        for f in self.features:
            pollutantName = str(f).split('_')[0]
            strFeature=strFeature+comma+'sum(CASE WHEN a.' + pollutantName + \
            '= True THEN 0 ELSE raw.' + str(f) + ' END)/(CASE when sum(CASE WHEN a.' + pollutantName +\
            '= True THEN 0 ELSE 1 END) = 0 then 1 else sum(CASE WHEN a.' + pollutantName + '= True THEN 0 ELSE 1 END) end ) as '+str(f)
            comma=', '
        strLabelAvg=""
        strLabelList=""
        comma=""
        for f in self.label_list:
            strLabelAvg=strLabelAvg+comma+'avg('+str(f)+') as label_'+str(f)
            strLabelList=strLabelList+comma+'station.label_'+str(f)
            comma=", "
        # print(" -- strLabelAvg: " + strLabelAvg)
        interval=self.getIntervalInMinutesFromString(self.interval)
        q="""
          select sensor.*, %s
               , station.coverage station_coverage
                          from
             (
             select status.id_sensor_low_cost
                  , (to_timestamp(ceil(extract(epoch 
                                               from phenomenon_time::timestamp with time zone
                                              ) / (60 * %s )
                                       ) * (60 * %s)
                                  )
                     )::timestamp as phenomenon_time_rounded
                  , count(raw.id_sensor_low_cost_status) as coverage
                  , status.id status_id
                  , status.id_sensor_low_cost_feature
                  , %s
                  from sensor_raw_observation as raw LEFT JOIN (select *
                from sensor_raw_observation_anomaly
                where id_anomaly_detection_algorithm=13) AS a ON (a.phenomenon_time_sensor_raw_observation=raw.phenomenon_time and a.id_sensor_low_cost_status=raw.id_sensor_low_cost_status)
                     ,(select ss1.id
                           , ss1.id_sensor_low_cost
                           , ss1.status
                           , ss1.id_sensor_low_cost_feature
                           , ss1.operator
                           , ss1.datetime
                           , (select ss2.datetime
                                from sensor_low_cost_status ss2
                               where ss1.id_sensor_low_cost = ss2.id_sensor_low_cost
                                 and ss1.datetime < ss2.datetime
                                order by ss2.datetime
                                limit 1
                             ) as datetime_end
                        from sensor_low_cost_status ss1
                         order by ss1.id_sensor_low_cost, ss1.datetime
                       ) as status
                  where status.id_sensor_low_cost = %s
                    and raw.id_sensor_low_cost_status = status.id
                    and phenomenon_time >= '%s'
                    and phenomenon_time  < '%s'
                    and (status.status = 'calibration')
                    and (phenomenon_time >= status.datetime
                         and (phenomenon_time < status.datetime_end or status.datetime_end is null ))
                  group by(status.id_sensor_low_cost,status.id_sensor_low_cost_feature
                           , phenomenon_time_rounded, status.id)
                  order by phenomenon_time_rounded
              ) as sensor
              , (
                 select (to_timestamp(ceil(extract(epoch
                                            from phenomenon_time::timestamp with time zone
                                        ) / (60 * %s )) * (60 * %s))
                        )::timestamp as phenomenon_time_rounded
                    , id_aq_legal_station
                    , count(raw.phenomenon_time) as coverage
                    , %s
                  from 
                    aq_legal_station_observation_one_minute_not_validated as raw
                  where phenomenon_time >= '%s'
                    and phenomenon_time  < '%s'
                  group by(phenomenon_time_rounded, id_aq_legal_station)
                  order by phenomenon_time_rounded
               ) station
              , sensor_low_cost_feature feature
           where sensor.phenomenon_time_rounded = station.phenomenon_time_rounded
             and sensor.id_sensor_low_cost_feature =  feature.id
             and feature.id_aq_legal_station = station.id_aq_legal_station
           order by phenomenon_time_rounded
        """%(strLabelList
             , interval
             , interval
             , strFeature
             , self.id_sensor
             , self.begin_time
             , self.end_time
             , interval
             , interval
             , strLabelAvg
             , self.begin_time
             , self.end_time)
        #
        # print( " ---the query---\n" + q)
        conn = self.getConnection()
        df_station_and_raw_resampled=sqlio.read_sql_query(q, conn)
        #
        df_station_and_raw_resampled.set_index('phenomenon_time_rounded',inplace=True)
        #
        #
        cutoff_Value=0.1
        df_trainig_and_testing=df_station_and_raw_resampled.astype('float')
        for pollutant_label in self.label_list:
            label_name = 'label_'+str(pollutant_label)
            a=0
            check=df_trainig_and_testing[label_name]
            cur = conn.cursor()
            for current, next in zip(check, check[1:]): # this loop removes constant values.
                if current==next:
                    check[a]=np.nan
                a=a+1    
            check=[float('nan') if x<cutoff_Value else x for x in check]
            df_trainig_and_testing[label_name]=check.copy()
            df_trainig_and_testing.dropna(inplace=True)
            cur.close()
        return(df_trainig_and_testing)




