import sys
import os
import json

from psycopg2 import extras

import dill
import psycopg2
import numpy as np
import pandas as pd
import pandas.io.sql as sqlio
import datetime

sys.path.insert(0, '../..')
from trafair_db_config import trafair_db_getConnection

"""
 A tool library for applying calibration
 requirements
   pip3 install numpy pandas
"""

class CalibrationApplyFramework():
    def applyCalibrationSensorPollutantDillDf(self
                                              , calibrator
                                              , begin_time
                                              , end_time
                                              , id_sensor
                                              , interval_in_minutes
                                              , pollutant_label
                                              , do_persist_data
                                              , path_dill
                                              , anomaly = False
    ):
        """
        This function never write the DB,
        because lacks data: id_sensor_calibration and
        the other pollutants.
        """
        #
        #
        # getting data to calibrate
        #

        records = self.getDataToApplyAsDataFrame(begin_time
                                                 , end_time
                                                 , id_sensor
                                                 , interval_in_minutes
                                                 , anomaly
        )

        records = records.rename(columns = {'phenomenon_time_rounded':'phenomenon_time'})
        output = records.copy()
        output = output.loc[calibrator.info_dictionary["number_of_previous_observations"]:]

        output['result_time'] = datetime.datetime.now()
        #
        prediction = calibrator.apply_df(records,str(interval_in_minutes)+"T",path_dill)
        prediction['coverage'] = records['coverage']
        return prediction
        #
        """
        output=output.merge(prediction, how='left', on='phenomenon_time')
        print('--input size--')
        print(records.shape)
        print('--output size--')
        print(records.shape)
        print('----output----')
        print(output)
        print(output['phenomenon_time'])
        """
        #sqlio.to_sql(output,table_name,sql_engine,if_exists='append',index=False)
        #
        # if (do_persist_data):
        #     conn = trafair_db_getConnection()
        #     print("writing data to db..")
        #     # conn.commit()
    
    def applyCalibrationSensorPollutantDillCSV(self
                                              , calibrator
                                              , begin_time
                                              , end_time
                                              , interval_in_minutes
                                              , csv_file_name_with_features
    ):
        """
        This function produces calibrated data using the given calibrator on the csv feature data
        """
        #
        #
        # getting data to calibrate
        #

        with open('../data/' + csv_file_name_with_features, 'r') as csv_file:
            records = pd.read_csv(csv_file)

        for f in calibrator.info_dictionary["feat_order"]:
            if f not in records.columns:
                #ERROR
                print('one of the feature needed to run the model is missing in the csv input:' + str(f))

        output=records.copy()
        output=output.loc[calibrator.info_dictionary["number_of_previous_observations"]:]

        output['result_time'] = datetime.datetime.now()
        records['phenomenon_time'] = pd.to_datetime(records['phenomenon_time'])
        records = records.resample(str(interval_in_minutes) + 'T', on='phenomenon_time', label='right').mean()
        records.reset_index(level=0,inplace=True)
        prediction = calibrator.apply_df(records,str(interval_in_minutes),'../data/' + calibrator.info_dictionary["dill_file_name"])
        #prediction['coverage'] = records['coverage']
        return prediction
        #
        """
        output=output.merge(prediction, how='left', on='phenomenon_time')
        print('--input size--')
        print(records.shape)
        print('--output size--')
        print(records.shape)
        print('----output----')
        print(output)
        print(output['phenomenon_time'])
        """
        #sqlio.to_sql(output,table_name,sql_engine,if_exists='append',index=False)
        #
        # if (do_persist_data):
        #     conn = trafair_db_getConnection()
        #     print("writing data to db..")
        #     # conn.commit()


    def applyCalibrationDillDfRepaired(self
                                              , calibrator
                                              , begin_time
                                              , end_time
                                              , id_sensor
                                              , interval_in_minutes
                                              , pollutant_label
                                              , do_persist_data
                                              , path_dill
                                              , anomaly = False
    ):
        """
        This function never write the DB,
        because lacks data: id_sensor_calibration and
        the other pollutants.
        """
        #
        #
        # getting data to calibrate
        #

        records = self.getDataToApplyAsDataFrameRepaired(begin_time
                                                 , end_time
                                                 , id_sensor
                                                 , interval_in_minutes
        )

        records = records.rename(columns = {'phenomenon_time_rounded':'phenomenon_time'})

        output=records.copy()
        output=output.loc[calibrator.info_dictionary["number_of_previous_observations"]:]

        output['result_time'] = datetime.datetime.now()
        #
        prediction = calibrator.apply_df(records,str(interval_in_minutes)+"T",path_dill)
        prediction['coverage'] = records['coverage']
        return prediction
        #
        """
        output=output.merge(prediction, how='left', on='phenomenon_time')
        print('--input size--')
        print(records.shape)
        print('--output size--')
        print(records.shape)
        print('----output----')
        print(output)
        print(output['phenomenon_time'])
        """
        #sqlio.to_sql(output,table_name,sql_engine,if_exists='append',index=False)
        #
        # if (do_persist_data):
        #     conn = trafair_db_getConnection()
        #     print("writing data to db..")
        #     # conn.commit()
    def getDataToApply_theSqlQuery(self,anomaly = False):
        rv ="""
         select status.id_sensor_low_cost 
           , (to_timestamp(ceil(extract(epoch from phenomenon_time::timestamp with time zone) / (60 * %s )) * (60 * %s)))::timestamp as phenomenon_time_rounded
           ,  count(id_sensor_low_cost_status) as coverage
           , status.id_sensor_low_cost_feature,
            avg(no_aux) as no_aux,
            avg(no_we) as no_we,
            avg(no2_aux) as no2_aux,
            avg(no2_we) as no2_we,
            avg(ox_aux) as ox_aux,
            avg(ox_we) as ox_we,
            avg(co_aux) as co_aux,
            avg(co_we) as co_we,
            avg(humidity) as humidity ,
            avg(temperature) as temperature
           from sensor_raw_observation as raw, sensor_low_cost_status as status
           where status.id_sensor_low_cost = %s
             and raw.id_sensor_low_cost_status = status.id
             and phenomenon_time < %s
             and phenomenon_time >= %s
             and (status.status = 'running' or status.status = 'calibration')
           group by(status.id_sensor_low_cost,status.id_sensor_low_cost_feature
                            , phenomenon_time_rounded)
           order by phenomenon_time_rounded
           ;
        """
        rv_anomaly = """
        select status.id_sensor_low_cost 
           , (to_timestamp(ceil(extract(epoch from phenomenon_time::timestamp with time zone) / (60 * %s )) * (60 * %s)))::timestamp as phenomenon_time_rounded
           ,  count(raw.id_sensor_low_cost_status) as coverage
           , status.id_sensor_low_cost_feature,
           sum(CASE WHEN a.no = True THEN 0 ELSE no_we END)/(CASE when sum(CASE WHEN a.no = True THEN 0 ELSE 1 END) = 0 then null else sum(CASE WHEN a.no = True THEN 0 ELSE 1 END) end ) as no_we, 
               sum(CASE WHEN a.no2 = True THEN 0 ELSE no2_we END)/(CASE when sum(CASE WHEN a.no2 = True THEN 0 ELSE 1 END) = 0 then null else sum(CASE WHEN a.no2 = True THEN 0 ELSE 1 END) end ) as no2_we, 
                sum(CASE WHEN a.no = True THEN 0 ELSE no_aux END)/(CASE when sum(CASE WHEN a.no = True THEN 0 ELSE 1 END) = 0 then null else sum(CASE WHEN a.no = True THEN 0 ELSE 1 END) end ) as no_aux, 
                sum(CASE WHEN a.no2 = True THEN 0 ELSE no2_aux END)/(CASE when sum(CASE WHEN a.no2 = True THEN 0 ELSE 1 END) = 0 then null else sum(CASE WHEN a.no2 = True THEN 0 ELSE 1 END) end ) as no2_aux, 
               sum(CASE WHEN a.ox = True THEN 0 ELSE ox_aux END)/(CASE when sum(CASE WHEN a.ox = True THEN 0 ELSE 1 END) = 0 then null else sum(CASE WHEN a.ox = True THEN 0 ELSE 1 END) end ) as ox_aux,
               sum(CASE WHEN a.ox = True THEN 0 ELSE ox_we END)/(CASE when sum(CASE WHEN a.ox = True THEN 0 ELSE 1 END) = 0 then null else sum(CASE WHEN a.ox = True THEN 0 ELSE 1 END) end ) as ox_we,
               sum(CASE WHEN a.co = True THEN 0 ELSE co_we END)/(CASE when sum(CASE WHEN a.co = True THEN 0 ELSE 1 END) = 0 then null else sum(CASE WHEN a.co = True THEN 0 ELSE 1 END) end ) as co_we,
                 sum(CASE WHEN a.co = True THEN 0 ELSE co_aux END)/(CASE when sum(CASE WHEN a.co = True THEN 0 ELSE 1 END) = 0 then null else sum(CASE WHEN a.co = True THEN 0 ELSE 1 END) end ) as co_aux,
                 sum(CASE WHEN a.temperature = True THEN 0 ELSE raw.temperature END)/(CASE when sum(CASE WHEN a.temperature = True THEN 0 ELSE 1 END) = 0 then null else sum(CASE WHEN a.temperature = True THEN 0 ELSE 1 END) end ) as temperature,
                 sum(CASE WHEN a.humidity = True THEN 0 ELSE raw.humidity END)/(CASE when sum(CASE WHEN a.humidity = True THEN 0 ELSE 1 END) = 0 then null else sum(CASE WHEN a.humidity = True THEN 0 ELSE 1 END) end ) as humidity
           from sensor_raw_observation as raw LEFT JOIN (select *
                from sensor_raw_observation_anomaly
                where id_anomaly_detection_algorithm=13) AS a ON (
                a.phenomenon_time_sensor_raw_observation=raw.phenomenon_time 
                and a.id_sensor_low_cost_status=raw.id_sensor_low_cost_status),
                sensor_low_cost_status as status
           where status.id_sensor_low_cost = %s
             and raw.id_sensor_low_cost_status = status.id
             and phenomenon_time < %s
             and phenomenon_time >= %s
             and (status.status = 'running' or status.status = 'calibration')
           group by(status.id_sensor_low_cost,status.id_sensor_low_cost_feature
                            , phenomenon_time_rounded)
           order by phenomenon_time_rounded
           ;
           """
        if (anomaly):
            return rv_anomaly
        else:
            return rv
    def getDataToApply_theSqlQueryRepaired(self,anomaly = False):
        q = """
        select  ss.id_sensor_low_cost, a.phenomenon_time_sensor_raw_observation_10min as phenomenon_time_rounded,
        5 as coverage, ss.id_sensor_low_cost_feature,
        coalesce( a.no_we, r.no_we ) as no_we,
        coalesce( a.no_aux, r.no_aux ) as no_aux,
        coalesce( a.no2_we, r.no2_we ) as no2_we,
        coalesce( a.no2_aux, r.no2_aux ) as no2_aux,
        coalesce( a.ox_we, r.o3_we ) as ox_we,
        coalesce( a.ox_aux, r.o3_aux ) as ox_aux,
        coalesce( a.co_we, r.co_we ) as co_we,
        coalesce( a.co_aux, r.co_aux ) as co_aux,
        coalesce( a.temperature, r.temperature ) as temperature,
        coalesce( a.humidity, r.humidity ) as humidity
          from (aggregated_raw_observation_without_anomaly as a left join sensor_low_cost_status as ss on ss.id = a.id_sensor_low_cost_status )
         left join repaired_anomaly as r on r.id_sensor_low_cost_status = a.id_sensor_low_cost_status 
                       and r.phenomenon_time_sensor_raw_observation_10min = a.phenomenon_time_sensor_raw_observation_10min
        where a.phenomenon_time_sensor_raw_observation_10min < %s
        and a.phenomenon_time_sensor_raw_observation_10min >= %s
		and ss.id_sensor_low_cost = %s
        order by ss.id_sensor_low_cost,a.phenomenon_time_sensor_raw_observation_10min
        """
        return q
    def getDataToApplyAsCursor(self, begin_time
                               , end_time
                               , id_sensor
                               , interval_in_minutes
        ):
        conn = trafair_db_getConnection()
        cur = conn.cursor()
        sqlLowCost2=self.getDataToApply_theSqlQuery()
        cur.execute(sqlLowCost2,
                    (
                        str(interval_in_minutes)
                        , str(interval_in_minutes)
                        , id_sensor
                        , end_time
                        , begin_time
                    ))
        return(cur)

    def getDataToApplyAsDataFrame(self, begin_time
                                  , end_time
                                  , id_sensor
                                  , interval_in_minutes
                                  , anomaly = False
        ):

        conn = trafair_db_getConnection()
        sqlLowCost2=self.getDataToApply_theSqlQuery(anomaly)
        records=sqlio.read_sql_query(sqlLowCost2,conn
                                     , params=(
                                         str(interval_in_minutes)
                                         , str(interval_in_minutes)
                                         , id_sensor
                                         , end_time
                                         , begin_time
                                     ))

        return(records)
    def getDataToApplyAsDataFrameRepaired(self, begin_time
                                  , end_time
                                  , id_sensor
                                  , interval_in_minutes
        ):

        conn = trafair_db_getConnection()
        sqlLowCost2=self.getDataToApply_theSqlQueryRepaired()
        records=sqlio.read_sql_query(sqlLowCost2,conn
                                     , params=(end_time
                                         , begin_time
										 , id_sensor
                                     ))

        return(records)
    def getSensorCalibration(self,id_row):
        try:
            conn = trafair_db_getConnection()
            cur = conn.cursor()
            query="""SELECT * FROM sensor_calibration WHERE id=%s;"""
            cur.execute(query,(id_row,))
            row=cur.fetchone()
            res={'id': id_row,
                 'co':row[1],
                 'no':row[2],
                 'no2':row[3],
                 'o3':row[4],
                 'sensor':row[5],
                 }
            return res
        except(Exception,psycopg2.ProgrammingError) as error:
            print("Insert a valid id row for sensor_calibration table:", error)
            quit()

    def openDill(self, id_dill,name_dill):
        try:
            conn = trafair_db_getConnection()
            cur = conn.cursor()
            query = """SELECT * FROM sensor_calibration_algorithm_test WHERE id=%s;"""
            cur.execute(query, (id_dill,))
            row = cur.fetchone()
        except(Exception, psycopg2.ProgrammingError) as error:
            print("Insert a valid id row for sensor_calibration_algorithm table:", error)
            return None
        path=row[6]+'/'
        try:
            with open(os.path.join(path,name_dill),'rb') as dill_file:
                calibrator = dill.load(dill_file)
            return calibrator,path
        except:
            print("Can't open "+name_dill)
            return None

    def getSensorFeat(self,id_sensor,end_time):
        try:
            conn = trafair_db_getConnection()
            query = """
            SELECT id_sensor_low_cost_feature,datetime
            FROM sensor_low_cost_status
            WHERE id_sensor_low_cost=%s AND datetime<%s
            ORDER BY datetime
            """
            sensor_feat=sqlio.read_sql_query(query,conn,params=(id_sensor,end_time))
            return (sensor_feat)
        except(Exception, psycopg2.ProgrammingError) as error:
            print("Error\n")
            print(error)
            return


    def insertPredictionToDB(self,prediction):
        pd.set_option('display.max_columns', None)
        print(prediction)
        tuples=[tuple(x) for x in prediction.to_numpy()]
        try:
            conn = trafair_db_getConnection()
            cur = conn.cursor()
            query = """
            INSERT INTO sensor_calibrated_observation_one_hour(
            id_sensor_calibration,phenomenon_time,result_time,no,no2,co,o3,
            co_out_of_range,no_out_of_range,no2_out_of_range,o3_out_of_range,
            id_sensor_low_cost_feature,coverage)
            VALUES %s
            ON CONFLICT(id_sensor_calibration,phenomenon_time) DO UPDATE SET 
            (id_sensor_calibration,phenomenon_time,result_time,no,no2,co,o3,co_out_of_range,no_out_of_range,
            no2_out_of_range,o3_out_of_range,id_sensor_low_cost_feature,coverage)=
            (EXCLUDED.id_sensor_calibration,EXCLUDED.phenomenon_time,EXCLUDED.result_time,EXCLUDED.no,EXCLUDED.no2,EXCLUDED.co,EXCLUDED.o3,
            EXCLUDED.co_out_of_range,EXCLUDED.no_out_of_range,EXCLUDED.no2_out_of_range,EXCLUDED.o3_out_of_range,
            EXCLUDED.id_sensor_low_cost_feature,EXCLUDED.coverage);
            """
            #cur.executemany(query, prediction.values)
            extras.execute_values(cur,query,tuples)
            conn.commit()
            cur.close()
            conn.close()
            print("Calibration's results saved")
            return
        except(Exception, psycopg2.ProgrammingError) as error:
            print("Error while writing calibration's results in the DB\n")
            print(error)
            return


