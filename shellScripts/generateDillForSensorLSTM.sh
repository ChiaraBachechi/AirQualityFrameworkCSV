#!/bin/bash

#!/usr/bin/env python
# vim:tabstop=2:autoindent:
#
# Authors: Chiara Bachechi
# Purpose: Launch the calibration for all the pollutants of a given sensor with LSTM model
# Usage:
#     generateDillForSensor.sh <id_sensor> <begintime> <endtime>
#
					   
										   															
# Command-line input parameters

ID_SENSOR="$1"
BEGIN = "$2"
END = "$3"
echo $ID_SENSOR


python3 main.py --id_sensor "$ID_SENSOR" --begin_time "$BEGIN" --end_time "$END" --feature_list "no_we-no_aux-no2_we-no2_aux-ox_we-ox_aux-co_we-co_aux-temperature-humidity" --label_list "no" --pollutant_label "no" --trainer_class_name "Calib_LSTM_FunctionTrainer_001" --trainer_module_name "calib_LSTM_FunctionTrain" --action "trainToDB" 														  
python3 main.py --id_sensor "$ID_SENSOR" --begin_time "$BEGIN" --end_time "$END" --feature_list "no_we-no_aux-no2_we-no2_aux-ox_we-ox_aux-co_we-co_aux-temperature-humidity" --label_list "no2" --pollutant_label "no2" --trainer_class_name "Calib_LSTM_FunctionTrainer_001" --trainer_module_name "calib_LSTM_FunctionTrain" --action "trainToDB"															  
python3 main.py --id_sensor "$ID_SENSOR" --begin_time "$BEGIN" --end_time "$END" --feature_list "no_we-no_aux-no2_we-no2_aux-ox_we-ox_aux-co_we-co_aux-temperature-humidity" --label_list "o3" --pollutant_label "o3" --trainer_class_name "Calib_LSTM_FunctionTrainer_001" --trainer_module_name "calib_LSTM_FunctionTrain" --action "trainToDB"													  

   
   
