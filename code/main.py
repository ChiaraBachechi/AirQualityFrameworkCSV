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
    parser.add_argument('--target_label', dest='target_label', type=str \
                        , help='Insert the name of the target label' \
                        , default='')
    parser.add_argument('--label_list', dest='label_list', type=str \
                        , help='Insert the name of the all the target label you want in the csv generated from the db.' \
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
    parser.add_argument('--csv_calibrated_data',dest='csv_calibrated_data',type=str,
                        help="Insert the name of the CSV where to put the calibrated data.",default='calibratedData.csv')
    return parser

def optionsToInfo(options):
  status={}
  status["dates"] = {'start': options['begin_time'], 'end': options['end_time']}
  status['id_sensor'] = options['id_sensor']
  status['feat_order'] = options['feature_list'].split('-')
  status['trainer_module_name'] = options['trainer_module_name']
  status['trainer_class_name'] = options['trainer_class_name']
  status['interval'] = options['interval']
  status['target_label'] = options['target_label']
  status['test_size'] = options['test_size']
  status['pollutant_label'] = options['pollutant_label']
  status['dill_file_name'] = options['dill_file_name']
  status['csv_target_data'] = options['csv_target_data']
  status['csv_feature_data'] = options['csv_feature_data']
  #status['label_list']= options['label_list']
  status['number_of_previous_observations'] = options['number_of_previous_observations']
  if (options['algorithm_parameters'] == ""):
      status['algorithm_parameters'] = {}
  else:
      status['algorithm_parameters'] = json.load(options['algorithm_parameters'])
  status['units_of_measure'] = {'no': {'unit_of_measure': 'ug/m^3', 'conversions': [{'from': 'ppb', 'factor': 1.25}]}, 'no2': {'unit_of_measure': 'ug/m^3', 'conversions': [{'from': 'ppb', 'factor': 1.912}]}, 'o3': {'unit_of_measure': 'ug/m^3', 'conversions': [{'from': 'ppb', 'factor': 2.0}]}, 'co': {'unit_of_measure': 'ug/m^3', 'conversions': [{'from': 'mg/m^3', 'factor': 1000}, {'from': 'ppm', 'factor': 1160}, {'from': 'ppb', 'factor': 1.16}]}}

  print(" --- optionsToInfo:\n", json.dumps(status, sort_keys=True, indent=2))
  return status
def trainAndSaveDillToFile(options):
    """ example
         config.json file:
        {
        id_sensor: 4011, 
         begin_time: "2019-09-01 00:00:00", 
         end_time: "2019-09-15 00:00:00", 
         feature_list: "no_we-no_aux", 
         csv_feature_data: input_raw.csv, 
         csv_target_data: input_target.csv,
         target_label: "label_no",
         trainer_class_name: "Calib_LSTM_FunctionTrainer_001",
         trainer_module_name: "calib_LSTM_FunctionTrain",
         action: "trainAndSaveDillToFile",
         csv_feature_data: input_raw.csv,
         csv_target_data: input_target.csv,
         pollutant_label: "no",
         dill_file_name: "4011_new.dill"
         }
        """
    framework = calibrationAlgorithmFramework.CalibrationAlgorithmFramework()
    framework.initFromInfo(optionsToInfo(options))
    framework.createTrainingAndTestingFromCSV()
    framework.trainCalibrator()
    calibrator = framework.getCalibrator()
    # print(" --- calibrator: " + str(calibrator))
    with open('../results/' + options['dill_file_name'], 'wb') as dill_file:
        dill.dump(calibrator, dill_file)
    return
def trainAndSaveDillToFileFromInfo(options):
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
           --dill_file_name      calibrator001.dill
    example
         config.json file:
        {
         info_file_name: 4011_info_file.json
         }
        """
    framework = calibrationAlgorithmFramework.CalibrationAlgorithmFramework()
    with open('../data/' + options['info_file_name'], 'r') as f:
        info = json.load(f)
    framework.initFromInfo(info)
    framework.createTrainingAndTestingFromCSV()
    framework.trainCalibrator()
    calibrator = framework.getCalibrator()
    with open('../results/' + options['dill_file_name'], 'wb') as dill_file:
        dill.dump(calibrator, dill_file)
    return
def getInfoFromDillFile(options):
    """
    example
         config.json file:
        {
         dill_file_name: "4011_new.dill"
         }
    """
    with open('../data/' +  options['dill_file_name'], 'rb') as dill_file:
            calibrator = dill.load(dill_file)
    info = json.dumps(calibrator.get_info(), sort_keys=True, indent=2)
    with open('../results/' + options['info_file_name'], 'w') as outfile:
        outfile.write(info)
    return
def applyCalibrationSensorPollutantDillDf(options):
    framework = calibrationAlgorithmFramework.CalibrationAlgorithmFramework()
    with open('../data/' + options['dill_file_name'], 'rb') as dill_file:
        calibrator = dill.load(dill_file)
    if ('number_of_previous_observations' in calibrator.info_dictionary):
        begin_time=datetime.datetime.strptime(options['begin_time'], '%Y-%m-%d %H:%M:%S')
        time_diff = int(options['interval'][:-1]) * (calibrator.info_dictionary["number_of_previous_observations"]+1)
        if options['interval'][-1:] == 'T':
            begin_time = begin_time - datetime.timedelta(minutes=time_diff)
        else:
            begin_time = begin_time - datetime.timedelta(hours=time_diff)
        begin_time=str(begin_time)

    if(options['number_of_previous_observations'] != calibrator.info_dictionary["number_of_previous_observations"]):
        print("The model was trained with number_of_previous_observations=",calibrator.info_dictionary["number_of_previous_observations"])
        return
    if(list(options['feature_list'].split("-")) != calibrator.info_dictionary["feat_order"]):
        print("Error: The feature of the model and the feature listed in the options are different.\
        The model was trained with these features:")
        print(str(calibrator.info_dictionary["feat_order"]))
        return
    #
    # save to file of the dataset

    frameApply = calibrationApplyFramework.CalibrationApplyFramework()
    prediction = frameApply.applyCalibrationSensorPollutantDillCSV(calibrator
                                                        , begin_time
                                                        , options['end_time']
                                                        , framework.getIntervalInMinutesFromString(options['interval'])
                                                        ,options['csv_feature_data'])
    prediction.to_csv('../results/' + options['csv_calibrated_data'])
    return

def checkOptions(options):
    all_feature_available = ['no_we', 'no_aux', 'no2_we', 'no2_aux', 'ox_we', 'ox_aux', 'co_we', 'co_aux',
                             'temperature', 'humidity']
    if not options['feature_list']:  # check if empty -  if so, get the entire list
        features = all_feature_available
    else:  # gets the input and convert to int list
        features = options['feature_list'].split('-')  # split the str to list
        for p in features:  # check the numbers of the sensors match
            if p not in all_feature_available:
                raise ValueError(p + ' is not a valid pollutant')
    if options['interval'].find('T') < 0 and options['interval'].find('H') < 0:
        raise ValueError(
            options['interval'] + ' is not a valid interval. It should contain T (for minutes) or H (for hours).')
    return


def main(args=None):
    #argParser = addOptions()
    #options = argParser.parse_args(args=args)
    with open('../data/config_train.json', 'r') as config_file:
        options = json.load(config_file)
    checkOptions(options)
    trainAndSaveDillToFile(options)
    with open('../data/config_getInfo.json', 'r') as config_file:
        options = json.load(config_file)   
    getInfoFromDillFile(options)
    with open('../data/config_fromInfo.json', 'r') as config_file:
        options = json.load(config_file) 
    trainAndSaveDillToFileFromInfo(options)
    with open('../data/config_apply.json', 'r') as config_file:
        options = json.load(config_file)
    checkOptions(options)
    applyCalibrationSensorPollutantDillDf(options)
    return



main()