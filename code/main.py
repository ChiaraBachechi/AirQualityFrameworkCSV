import argparse
import ast
import os

import numpy as np
import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
import sys
import importlib

import tensorflow

import dill
import json
import time
import datetime

# ---
import calibrationAlgorithmTrainer_interfaces
import calibrationAlgorithmFramework
import calibrationApplyFramework


def addOptions():
    parser = argparse.ArgumentParser(description='Generate dill file for calibration, training, testing and tuning')
    parser.add_argument('--id_sensor', '-s', dest='id_sensor', type=str \
                        ,
                        help='The id of the sensors which is willing to be calibrated separeted by a - or all for all the available sensorss.' \
                        , default="all")
    parser.add_argument('--begin_time', '-b', dest='begin_time', type=str, \
                        help='Insert the date and time to start the calibration from. Formatted as YYYY-MM-DD HH:MM:SS')
    parser.add_argument('--end_time', '-e', dest='end_time', type=str, \
                        help='Insert the date and time to end the calibration. Formatted as YYYY-MM-DD HH:MM:SS')
    parser.add_argument('--feature_list', '-l', dest='feature_list', type=str \
                        , help='Insert the name of the pollutants separated by a -.' \
                        , default='')
    parser.add_argument('--label_list', dest='label_list', type=str \
                        , help='Insert the name of the pollutants separated by a -.' \
                        , default='')
    parser.add_argument('--pollutant_label', '-p', dest='pollutant_label', type=str \
                        , help='Insert the name of the pollutant to calibrate.' \
                        )
    parser.add_argument('--trainer_class_name', dest='trainer_class_name', type=str,
                        help='Insert the name of the class that contains the definition of the trainer.')
    parser.add_argument('--trainer_module_name', dest='trainer_module_name', type=str,
                        help='Insert the name of the module (the file) that contains the definition of the trainer class.')
    parser.add_argument('--csv_feature_data', dest='csv_feature_data', type=str,
                        help='The name of the input csv file with the feature data, set --from_csv True', default='')
    parser.add_argument('--csv_target_data', dest='csv_target_data', type=str,
                        help='The name of the csv file with the target labels, set --from_csv True', default='')
    parser.add_argument('--interval_of_aggregation', '-t', dest='interval', type=str,
                        help='The number of minutes to aggregate the raw data and station data.', default="10T")
    parser.add_argument('--test_size', dest='test_size', type=float,
                        help='A number between 0 and 1 indicating the percentage of data to use to test the algorithm.',
                        default=0.20)
    parser.add_argument('--action', dest='action', type=str, help='The framework action to perform.', default="")
    parser.add_argument('--dill_file_name', dest='dill_file_name', type=str,
                        help='The file name of a trained calibrator.', default="tmp_calibrator.dill")
    parser.add_argument('--info_file_name', dest='info_file_name', type=str,
                        help='The file name of a trained calibrator.', default="tmp_calibrator.info")
    parser.add_argument('--algorithm_parameters', dest='algorithm_parameters', type=str,
                        help='A Json string with specific algorithm information.', default="")
    parser.add_argument('--do_persist_data', dest='do_persist_data', type=str,
                        help='Makes the db values persistent - it is not a dry run.', default="false")
    parser.add_argument('--number_of_previous_observations',dest='number_of_previous_observations',type=int,
                        help='The temporal window to consider with LSTM',default=1)
    parser.add_argument('--id_calibration',dest='id_calibration',type=int,
                        help="Insert sensor_calibration's row id to calibrate",default=0)
    parser.add_argument('--anomaly',dest='anomaly',type=ast.literal_eval,
                        help="True if you want to exclude anomalies",default=False)
    return parser

def optionsToInfo(options):
  status={}
  status["dates"] = {'start': options.begin_time, 'end': options.end_time}
  status['id_sensor'] = options.id_sensor
  status['feat_order'] = options.feature_list.split('-')
  status['label_list'] = options.label_list.split('-')
  status['trainer_module_name'] = options.trainer_module_name
  status['trainer_class_name'] = options.trainer_class_name
  status['interval'] = options.interval
  status['test_size'] = options.test_size
  status['pollutant_label'] = options.pollutant_label
  status['dill_file_name']=options.dill_file_name
  status['csv_target_data']=options.csv_target_data
  status['csv_feature_data']=options.csv_feature_data
  status['number_of_previous_observations'] =options.number_of_previous_observations
  if (options.algorithm_parameters == ""):
      status['algorithm_parameters'] = {}
  else:
      status['algorithm_parameters'] = json.load(options.algorithm_parameters)
  status['units_of_measure'] = {'no': {'unit_of_measure': 'ug/m^3', 'conversions': [{'from': 'ppb', 'factor': 1.25}]}, 'no2': {'unit_of_measure': 'ug/m^3', 'conversions': [{'from': 'ppb', 'factor': 1.912}]}, 'o3': {'unit_of_measure': 'ug/m^3', 'conversions': [{'from': 'ppb', 'factor': 2.0}]}, 'co': {'unit_of_measure': 'ug/m^3', 'conversions': [{'from': 'mg/m^3', 'factor': 1000}, {'from': 'ppm', 'factor': 1160}, {'from': 'ppb', 'factor': 1.16}]}}

  print(" --- optionsToInfo:\n", json.dumps(status, sort_keys=True, indent=2))
  return status


def main(args=None):
    argParser = addOptions()
    options = argParser.parse_args(args=args)

    all_feature_available = ['no_we', 'no_aux', 'no2_we', 'no2_aux', 'ox_we', 'ox_aux', 'co_we', 'co_aux',
                             'temperature', 'humidity']
    if not options.feature_list:  # check if empty -  if so, get the entire list
        features = all_feature_available
    else:  # gets the input and convert to int list
        features = options.feature_list.split('-')  # split the str to list
        for p in features:  # check the numbers of the sensors match
            if p not in all_feature_available:
                raise ValueError(p + ' is not a valid pollutant')
    if options.interval.find('T') < 0 and options.interval.find('H') < 0:
        raise ValueError(
            options.interval + ' is not a valid interval. It should contain T (for minutes) or H (for hours).')
    #
    #
    #
    ##
    action = options.action
    #
    #
    if (action == "trainAndSaveDillToFile"):
        #
        #
        """ example
         python3 main.py \
           --id_sensor 4003 \
           --begin_time "2019-08-01 00:00:00" \
           --end_time   "2019-08-20 00:00:00" \
           --feature_list "no_we-no_aux" \
           --label_list     "no" \
           --pollutant_label "no" \
           --trainer_module_name calibrationAlgorithmTrainer_dummy \
           --trainer_class_name  CalibrationAlgorithmTrainer_dummy \
           --dill_file_name      tmp_calibrator.dill \
           --df_csv_file_prefix "data/tmp_calibrator" \
           --action trainAndSaveDillToFile
         python3 main.py \
           --dill_file_name tmp_calibrator.dill \
           --action getInfoFromDillFile
           --anomaly True
           python3 main.py --id_sensor 4003\
           --begin_time "2019-08-01 00:00:00\
           --end_time   "2019-08-20 00:00:00"\
           --feature_list "no_we-no_aux"\
           --label_list     "no"\
           --pollutant_label "no"\
           --trainer_module_name calib_LSTM_FunctionTrain\
           --trainer_class_name  Calib_LSTM_FunctionTrainer_001\
           --dill_file_name provaAnomaly.dill\
           --action trainAndSaveDillToFile\
           --anomaly True
        """
        framework = calibrationAlgorithmFramework.CalibrationAlgorithmFramework()
        framework.initFromInfo(optionsToInfo(options))
        framework.createTrainingAndTestingFromCSV()
        framework.trainCalibrator()
        calibrator = framework.getCalibrator()
        # print(" --- calibrator: " + str(calibrator))
        with open('../results/' + options.dill_file_name, 'wb') as dill_file:
            dill.dump(calibrator, dill_file)

    #
    #
    elif (action == "trainAndSaveDillToFileFromInfo"):
        #
        #
        """ example
     cat <<EOF > Calib_RF_FunctionTrainer_001.json
  {
    "dates": {
      "end": "2019-08-20 00:00:00",
      "start": "2019-08-01 00:00:00"
    },
    "feat_order": [
      "no2_we",
      "no2_aux"
    ],
    "label_list": [
      "no2"
    ],
    "id_sensor": "4003",
    "interval": "10T",
    "pollutant_label": "no2",
    "test_size": 0.2,
    "trainer_class_name": "Calib_RF_FunctionTrainer_001",
    "trainer_module_name": "calib_RF_FunctionTrain",
    "units_of_measure": {
      "no2": {
        "unit_of_measure": "ug/m^3",
        "conversions" : [
          { "from": "ppb", "factor": 1.912 }
        ]
      }
    }
  }
  EOF
         python3 main.py \
           --action trainAndSaveDillToFileFromInfo \
           --info_file_name      Calib_RF_FunctionTrainer_001.json \
           --dill_file_name      calibrator001.dill \
           --df_csv_file_prefix "data/calibrator001"
        """
        framework = calibrationAlgorithmFramework.CalibrationAlgorithmFramework()
        with open(options.info_file_name, 'r') as f:
            info = json.load(f)
        framework.initFromInfo(info)
        framework.createTrainingAndTestingDB()
        framework.trainCalibrator()
        calibrator = framework.getCalibrator()
        #
        print(" --- calibrations info:\n",
              json.dumps(calibrator.get_info(), sort_keys=True, indent=2))
        #
        with open(options.dill_file_name, 'wb') as dill_file:
            dill.dump(calibrator, dill_file)
        print("\n dill calibrator saved as " + options.dill_file_name + "\n")
        if (options.df_csv_file_prefix != ""):
            framework.saveTrainingAndTestingDataToCsv(options.df_csv_file_prefix)
        #
        #
    elif (action == "trainAndSaveDillToFileFromInfoAndDf"):
        #
        #
        framework = calibrationAlgorithmFramework.CalibrationAlgorithmFramework()
        with open(options.info_file_name, 'r') as f:
            info = json.load(f)
        framework.initFromInfo(info)
        framework.createTrainingAndTestingDB()
        framework.trainCalibrator()
        calibrator = framework.getCalibrator()
        #
        print(" --- calibrations info:\n",
              json.dumps(calibrator.get_info(), sort_keys=True, indent=2))
        #
        with open(options.dill_file_name, 'wb') as dill_file:
            dill.dump(calibrator, dill_file)
        print("\n dill calibrator saved as " + options.dill_file_name + "\n")
        if (options.df_csv_file_prefix != ""):
            framework.saveTrainingAndTestingDataToCsv(options.df_csv_file_prefix)
        #
        print(" ----- testing features ")
        print(str(framework.get_df_test_features()))
        print(" ----- expected labels ")
        print(str(framework.get_df_test_labels()))
        print(" ----- output ")
        t = time.time()
        print(str(calibrator.apply_df(framework.get_df_test_features())))
        print(" --- time: " + str(time.time() - t))
        #
    #
    #
    elif (action == "getInfoFromDillFile"):
        #
        #
        """ example
         python3 main.py \
           --dill_file_name tmp_calibrator.dill \
           --action getInfoFromDillFile
        """
        with open(options.dill_file_name, 'rb') as dill_file:
            calibrator = dill.load(dill_file)
        # framework.initFromInfo(calibrator.get_info())
        print(json.dumps(calibrator.get_info(), sort_keys=True, indent=2))
        # trash
        #  print(json.dumps(calibrator.get_json, sort_keys=True, indent=2))
        #  print(calibrator.get_json)
        #  print(json.dumps(json.loads(calibrator.get_json), sort_keys=True, indent=2))
    #
    #
    #
    elif (action == "saveTrainingAndTestingDataToCsv"):
        #
        #
        """ example
         python3 main.py \
           --id_sensor 4003 \
           --begin_time "2019-08-01 00:00:00" \
           --end_time   "2019-08-20 00:00:00" \
           --feature_list "no_we-no_aux" \
           --label_list "no-o3" \
           --pollutant_label "no" \
           --df_csv_file_prefix "data/df_csv" \
           --action saveTrainingAndTestingDataToCsv
        """
        # variable check
        framework = calibrationAlgorithmFramework.CalibrationAlgorithmFramework()
        framework.initFromInfo(optionsToInfo(options))
        framework.createTrainingAndTestingDB()
        framework.saveTrainingAndTestingDataToCsv(options.df_csv_file_prefix)
        #
        # save to file of the dataset
    #
    #
    #
    # elif(action == "testDill_fromFileAndCsv_simple"):
    # #
    # #
    #   """ example
    #    python3 main.py \
    #      --dill_file_name tmp_calibrator.dill \
    #      --df_csv_file_prefix "data/tmp_calibrator" \
    #      --action testDill_fromFileAndCsv_simple
    #   """
    #   with open(options.dill_file_name, 'rb') as dill_file:
    #     calibrator = dill.load(dill_file)
    #   info=json.loads(calibrator.get_json)
    #   info["pollutant_label"]="no"
    #   info["interval"]="10T"
    #   info["test_size"]=20
    #   #
    #   framework = calibrationAlgorithmFramework.CalibrationAlgorithmFramework()
    #   framework.initFromInfo(info)
    #   framework.loadTrainingAndTestingDataFromCsv(options.df_csv_file_prefix)
    #   #
    #   # save to file of the dataset
    #   # framework.saveTrainingAndTestingDataToCsv("trash_")
    #   df_features=framework.get_df_test_features()
    #   df_labels=framework.get_df_test_labels()
    #   #print(str(df_features)
    #   #print(str(df_labels)
    #   #
    #   rv = None
    #   full_features_list=[ 'no_aux', 'no_we','no2_aux', 'no2_we','ox_aux', 'ox_we', 'co_aux', 'co_we','humidity','temperature']
    #   #to exclude values that are not in the range
    #   for i in range (0, df_features.shape[0]):
    #     f=df_features.iloc[i]
    #     l=df_labels.iloc[i]
    #     #print("f: " + str(f))
    #     #print("l:" + str(l))
    #     input_np=np.zeros([1,10])
    #     for c in df_features.columns:
    #       input_np[0,full_features_list.index(c)] = f[c]
    #       #print("c: ",c," f[c]:, ",f[c],"")
    #     rv = calibrator.apply_np(input_np)[0]
    #     #
    #     print("label: [",l[0],"] rv: [",rv,"]")
    #     #
    #
    #
    elif action == "saveTrainingDataToCsv":
        """ example
         python3 main.py \
           --id_sensor 4003 \
           --begin_time "2019-08-01 00:00:00" \
           --end_time   "2019-08-20 00:00:00" \
           --feature_list "no_we-no_aux" \
           --label_list "no-o3" \
           --pollutant_label "no" \
           --df_csv_file_prefix "" \
           --action saveTrainingDataToCsv
        """
        framework = calibrationAlgorithmFramework.CalibrationAlgorithmFramework()
        framework.initFromInfo(optionsToInfo(options))
        framework.createTrainingDB()
        framework.saveTrainingAndTestingDataToCsv(options.df_csv_file_prefix)
    elif action == "saveDatasetToCsv":
        """ example
         python3 main.py \
           --id_sensor 4003 \
           --begin_time "2019-08-01 00:00:00" \
           --end_time   "2019-08-20 00:00:00" \
           --feature_list "no_we-no_aux" \
           --label_list "no-o3" \
           --pollutant_label "no" \
           --df_csv_file_prefix "" \
           --action saveDatasetToCsv
        """
        framework = calibrationAlgorithmFramework.CalibrationAlgorithmFramework()
        framework.initFromInfo(optionsToInfo(options))
        framework.createAndSaveDataCleaningDB(options.df_csv_file_prefix)
    elif (action == "applyCalibrationSensorPollutantDillDf"):
        #
        #
        """ note
         this is a function for developers,
         usualy YOU MUST NOT RUN this method with do_persist_data==true
         it is intended for Dill testing only.
        example
         python3 main.py \
           --id_sensor 4003 \
           --begin_time "2019-08-01 00:00:00" \
           --end_time   "2019-08-20 00:00:00" \
           --pollutant_label "no" \
           --dill_file_name calibrator001.dill \
           --do_persist_data false \
           --interval_of_aggregation 10T \
           --action applyCalibrationSensorPollutantDillDf
        """
        #
        # variable check
        framework = calibrationAlgorithmFramework.CalibrationAlgorithmFramework()


        #
        #


        with open(options.dill_file_name, 'rb') as dill_file:
            calibrator = dill.load(dill_file)
        if ('number_of_previous_observations' in calibrator.info_dictionary):
            begin_time=datetime.datetime.strptime(options.begin_time, '%Y-%m-%d %H:%M:%S')
            time_diff = int(options.interval[:-1]) * (calibrator.info_dictionary["number_of_previous_observations"]+1)
            if options.interval[-1:] == 'T':
                begin_time = begin_time - datetime.timedelta(minutes=time_diff)
            else:
                begin_time = begin_time - datetime.timedelta(hours=time_diff)
            begin_time=str(begin_time)

        if(options.number_of_previous_observations!=calibrator.info_dictionary["number_of_previous_observations"]):
            print("The model was trained with number_of_previous_observations=",calibrator.info_dictionary["number_of_previous_observations"])
            return
        if(list(options.feature_list.split("-"))!=calibrator.info_dictionary["feat_order"]):
            print("Error: The feature of the model and the feature listed in the options are different.\
            The model was trained with these features:")
            print(str(calibrator.info_dictionary["feat_order"]))
            return
        #
        # save to file of the dataset

        frameApply = calibrationApplyFramework.CalibrationApplyFramework()
        frameApply.applyCalibrationSensorPollutantDillDf(calibrator
                                                         , begin_time
                                                         , options.end_time
                                                         , options.id_sensor
                                                         , framework.getIntervalInMinutesFromString(options.interval)
                                                         , options.pollutant_label
                                                         ,
                                                         True if (options.do_persist_data.lower() == "true") else False
                                                         , ""
                                                         ,options.anomaly
                                                         )
    #
    #
    elif (action == "trainToDB"):
        framework = calibrationAlgorithmFramework.CalibrationAlgorithmFramework()
        framework.initFromInfo(optionsToInfo(options))
        strnote = '';
        if(options.anomaly):
            framework.createTrainingAndTestingDB(True)
            strnote = 'excluding anomalies'
        else:
            framework.createTrainingAndTestingDB()
            strnote = 'not excluding anomalies'
        framework.trainCalibrator()
        calibrator = framework.getCalibrator()

        # print(" --- calibrator: " + str(calibrator))

        # inserimento nella tabella
        id_dill=framework.insertDillToDB(calibrator)
        #cambiare nome file
        name_file = str(options.id_sensor)+"_"+str(options.pollutant_label)+"_"+str(id_dill)+".dill"
        calibrator.info_dictionary["dill_file_name"] = name_file
        calibrator.info_dictionary["anomaly"] = strnote
        try:
            with open(os.path.join(os.getcwd()+"/dill/",name_file), 'wb') as dill_file:
                dill.dump(calibrator, dill_file)
            if (calibrator.info_dictionary["name"]=="LSTM"):
                model = tensorflow.keras.models.load_model("tmp")
                with open(os.path.join(os.getcwd() + "/dill/", name_file[:-5]), 'wb') as dill_file:
                    model.save(dill_file)
                os.remove("tmp")
                print("\n dill calibrator saved as " + calibrator.info_dictionary["dill_file_name"] + "\n")
        except:
            rows_deleted=framework.removeDillFromDB(id_dill)
            if(rows_deleted==1):
                print("Error while saving dill file, the entry has been removed")
            else:
                print("Error while saving dill file. Couldn't eliminate the entry, must be done manually")
    elif (action == "TrainToDBtestRepairing"):
        framework = calibrationAlgorithmFramework.CalibrationAlgorithmFramework()
        framework.initFromInfo(optionsToInfo(options))
        strnote = '';
        framework.createTrainingAndTestingDBRepairing()
        strnote = 'excluding anomalies'
        framework.trainCalibrator()
        calibrator = framework.getCalibrator()

        # print(" --- calibrator: " + str(calibrator))

        # inserimento nella tabella
        id_dill=framework.insertDillToDB(calibrator)
        #cambiare nome file
        name_file = str(options.id_sensor)+"_"+str(options.pollutant_label)+"_"+str(id_dill)+".dill"
        calibrator.info_dictionary["dill_file_name"] = name_file
        calibrator.info_dictionary["repaired"] = True
        try:
            with open(os.path.join(os.getcwd()+"/dill/",name_file), 'wb') as dill_file:
                dill.dump(calibrator, dill_file)
            if (calibrator.info_dictionary["name"]=="LSTM"):
                model = tensorflow.keras.models.load_model("tmp")
                with open(os.path.join(os.getcwd() + "/dill/", name_file[:-5]), 'wb') as dill_file:
                    model.save(dill_file)
                os.remove("tmp")
                print("\n dill calibrator saved as " + calibrator.info_dictionary["dill_file_name"] + "\n")
        except:
            rows_deleted=framework.removeDillFromDB(id_dill)
            if(rows_deleted==1):
                print("Error while saving dill file, the entry has been removed")
            else:
                print("Error while saving dill file. Couldn't eliminate the entry, must be done manually")
    elif (action == "applyDillsToSensor"):
        framework = calibrationAlgorithmFramework.CalibrationAlgorithmFramework()
        frameApply = calibrationApplyFramework.CalibrationApplyFramework()
        row = frameApply.getSensorCalibration(options.id_calibration)
        print(row)
        pollutants = ['no','no2','co','o3']
        prediction = pd.DataFrame()

        prediction['phenomenon_time']=pd.date_range(start=options.begin_time, end=options.end_time,
                                                    freq=options.interval)

        prediction['result_time']=str(datetime.datetime.now()).split('.')[0]
        prediction.insert(loc=0,column='id_sensor_calibration',value=row['id'])

        for p in pollutants:
            if(row[p]):
                name_dill=str(row['sensor'])+'_'+p+'_'+str(row[p])+'.dill'
                calibrator,path_dill=frameApply.openDill(row[p],name_dill)
                if(not calibrator):
                    prediction[p]=0
                    continue

                begin_time=options.begin_time
                if ('number_of_previous_observations' in calibrator.info_dictionary):
                    begin_time = datetime.datetime.strptime(options.begin_time, '%Y-%m-%d %H:%M:%S')
                    time_diff = int(options.interval[:-1]) * (
                                calibrator.info_dictionary["number_of_previous_observations"] + 1)
                    if options.interval[-1:] == 'T':
                        begin_time = begin_time - datetime.timedelta(minutes=time_diff)
                    else:
                        begin_time = begin_time - datetime.timedelta(hours=time_diff)
                    begin_time = str(begin_time)

                prediction = prediction.merge(frameApply.applyCalibrationSensorPollutantDillDf(calibrator
                                                                 , begin_time
                                                                 , options.end_time
                                                                 , options.id_sensor
                                                                 , framework.getIntervalInMinutesFromString(options.interval)
                                                                 , p
                                                                 ,
                                                                 True if (
                                                                             options.do_persist_data.lower() == "true") else False
                                                                 ,path_dill
                                                                 , options.anomaly),


                                      how='left', on='phenomenon_time',suffixes=('','_DROPME'))
            else:
                prediction[p]=0
        to_drop = [x for x in prediction if x.endswith('_DROPME')]
        prediction.drop(to_drop,axis=1,inplace=True)

        prediction['co_out_of_range'] = False
        prediction['no_out_of_range'] = False
        prediction['no2_out_of_range'] = False
        prediction['o3_out_of_range'] = False

        sensor_feat=frameApply.getSensorFeat(options.id_sensor,options.end_time)
        sensor_feat['datetime']=pd.to_datetime(sensor_feat['datetime'])
        prediction['id_sensor_low_cost_feature'] = prediction['phenomenon_time'].apply(lambda x:
                                                                                       int(sensor_feat[sensor_feat['datetime']<x]['id_sensor_low_cost_feature'][-1:]))

        prediction['coverage']=prediction.pop('coverage')
        frameApply.insertPredictionToDB(prediction)
    elif (action == "ApplyDillsToSensorRepaired"):
        framework = calibrationAlgorithmFramework.CalibrationAlgorithmFramework()
        frameApply = calibrationApplyFramework.CalibrationApplyFramework()
        row = frameApply.getSensorCalibration(options.id_calibration)
        print(row)
        pollutants = ['no','no2','co','o3']
        prediction = pd.DataFrame()

        prediction['phenomenon_time']=pd.date_range(start=options.begin_time, end=options.end_time,
                                                    freq=options.interval)

        prediction['result_time']=str(datetime.datetime.now()).split('.')[0]
        prediction.insert(loc=0,column='id_sensor_calibration',value=row['id'])

        for p in pollutants:
            if(row[p]):
                name_dill=str(row['sensor'])+'_'+p+'_'+str(row[p])+'.dill'
                calibrator,path_dill=frameApply.openDill(row[p],name_dill)
                if(not calibrator):
                    prediction[p]=0
                    continue

                begin_time=options.begin_time
                if ('number_of_previous_observations' in calibrator.info_dictionary):
                    begin_time = datetime.datetime.strptime(options.begin_time, '%Y-%m-%d %H:%M:%S')
                    time_diff = int(options.interval[:-1]) * (
                                calibrator.info_dictionary["number_of_previous_observations"] + 1)
                    if options.interval[-1:] == 'T':
                        begin_time = begin_time - datetime.timedelta(minutes=time_diff)
                    else:
                        begin_time = begin_time - datetime.timedelta(hours=time_diff)
                    begin_time = str(begin_time)

                prediction = prediction.merge(frameApply.applyCalibrationDillDfRepaired(calibrator
                                                                 , begin_time
                                                                 , options.end_time
                                                                 , options.id_sensor
                                                                 , framework.getIntervalInMinutesFromString(options.interval)
                                                                 , p
                                                                 ,
                                                                 True if (
                                                                             options.do_persist_data.lower() == "true") else False
                                                                 ,path_dill
                                                                 , options.anomaly),


                                      how='left', on='phenomenon_time',suffixes=('','_DROPME'))
            else:
                prediction[p]=0
        to_drop = [x for x in prediction if x.endswith('_DROPME')]
        prediction.drop(to_drop,axis=1,inplace=True)

        prediction['co_out_of_range'] = False
        prediction['no_out_of_range'] = False
        prediction['no2_out_of_range'] = False
        prediction['o3_out_of_range'] = False

        sensor_feat=frameApply.getSensorFeat(options.id_sensor,options.end_time)
        sensor_feat['datetime']=pd.to_datetime(sensor_feat['datetime'])
        prediction['id_sensor_low_cost_feature'] = prediction['phenomenon_time'].apply(lambda x:
                                                                                       int(sensor_feat[sensor_feat['datetime']<x]['id_sensor_low_cost_feature'][-1:]))

        prediction['coverage']=prediction.pop('coverage')
        frameApply.insertPredictionToDB(prediction)
        return






    ##
    ##
    # print(framework)


main()