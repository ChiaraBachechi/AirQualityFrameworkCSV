# AQCalibrationFramework
A calibration framework that allows automatically calibration electrochemical air quality sensors. The models are generated in 2 ways: using csv data as input, reading directly from a postgreSQL database instance generated with the provided script.

The framework allow users that have electrochemical sensors and legal station or laboratory measurements as reference data generating a calibration model for each sensor, and finally to apply it in oreder to obtain pollutants concentration. 

The framework is based on a data structure that can be generated in the PostgreSQL DBMS executing the attached script. This data strucutre must be populated with sensors data and target measurements of pollutants concentration and takes trace of all the changes in the status of the sensors. The data collected in 'calibration' status are associated directly with the legal station data and the model is generated automatically given the calibration interval of time to consider. This data structure also allows associating the calibrated measurements with the description and parameters of the algorithm used to generate them. In this way, the results of different approaches are easy to compare. Moreover, when the model is applied the table sensor_calibrated_observation is directly populated with the predicted values.
In order to set up the connecting with the database the file trafair_DB_config should be modified inserting in 'trafair_db_getConnection' the right password and database name.
Alternatively, the framework can be used with 'csv' files but this is suggested only for teasting purpose since the user must take trace of all the changes in the sensor status, associate correctly the measurements of the sensors with the target measurements and compare the results of different approaches can be more difficult.

Adaptability of the framework to other models and pollutants:
To add a new calibration model, it is necessary to add a new class which implements the proper functions to train that algorithm and apply the obtained model following the before mentioned interface and taking as examples the algorithms we implemented (see the scripts named “calib_VR_SVR_FunctionTrain.py” and “calib_LSTM_FunctionTrain.py”).   

Using the parameters trainer_class_name and trainer_module_name, the name of the class containing the training function needs to be specified when generating the dill of the corresponding model.

There are several actions that can be performed:
- trainAndSaveDillToFile: this action performs the training of the given model on the given period of time and create a dill file that is saved locally.
- trainToDB: this action performs the training of the given model on the given period of time and create a dill file saved locally and generate the instace of the new                model in the postgreSQL database.
- trainAndSaveDillToFileFromInfo: this action is created for the regeneration of an old dill. It uses a info file containing the information of the dill to generate a                                     new dill. It can be used also to avoid the insertion of the parameters that will be specified in the info file. The data to train the                                   model should be saved in a csv file and are not obtained directly querying the database. The generated dill is saved locally.
- getInfoFromDillFile: this function allows obtaining the info file from the dill. It is used to regenerate an old dill before applying trainAndSaveDillToFileFromInfo.
- applyCalibrationSensorPollutantDillDf: this action allows applying a dill to test data int eh given itnerval saved in the database, generating a csv file with the                                              pollutant concentration for the given sensor. It is useful for testing the dills but it does not write the calibrated values in                                          the dataframe.
- applyDillsToSensor: this action is the right way to apply the calirbation and save the calibrated values for all the pollutants in a unique row in the dataframe. Once                        all the models have been generated for a sensor a new row should be inserted in the sensor_calirbation table of the database with a reference to                        the models you are willing to use. Then the id of this row (id_calirbation) will be specified and all the calirbated observations with the                              pollutants concentration will refer to this specific calibration.

Examples of shell scripts that combine these functionalities are available in the ShellScripts folder in thsi repository.

## Generation of the models and apply them

    Generate dill file for calibration, training, testing and tuning
  - _id_sensor_ The id of the sensors which is willing to be calibrated separeted by a - or all for all the available sensors.
  - _begin_time_, _b_ Insert the date and time to start the calibration from. Formatted as YYYY-MM-DD HH:MM:SS
  - _end_time_, _e_ Insert the date and time to end the calibration. Formatted as YYYY-MM-DD HH:MM:SS
  - _feature_list_, _l_ Insert the name of the pollutants separated by a -.
  - _label_list_ Insert the name of the pollutants separated by a -.
  - _pollutant_label_, _p_ Insert the name of the pollutant to calibrate.
  - _trainer_class_name_ Insert the name of the class that contains the definition of the trainer.
  - _trainer_module_name_ Insert the name of the module (the file) that contains the definition of the trainer class.
  - _df_csv_file_prefix_ The name of the csv file, set --from_csv True
  - _interval_of_aggregation_, _t_ The number of minutes to aggregate the raw data and station data.
  - _test_size_ A number between 0 and 1 indicating the percentage of data to use to test the algorithm.
  - _action_ The framework action to perform.
  - _dill_file_name_ The file name of a trained calibrator.
  - _info_file_name_ The file name of a trained calibrator.
  - _algorithm_parameters_ A Json string with specific algorithm information.
  - _do_persist_data_ Makes the db values persistent - it is not a dry run.
  - _number_of_previous_observations_ The temporal window to consider with LSTM
  - _id_calibration_ Insert sensor_calibration's row id to calibrate



Example: trainAndSaveDillToFile
````shell command
python3 main.py 
           --dill_file_name tmp_calibrator.dill 
           --action getInfoFromDillFile
           --anomaly True
           python3 main.py --id_sensor 4003
           --begin_time "2019-08-01 00:00:00"
           --end_time   "2019-08-20 00:00:00"
           --feature_list "no_we-no_aux"
           --label_list     "no"
           --pollutant_label "no"
           --trainer_module_name calib_LSTM_FunctionTrain
           --trainer_class_name  Calib_LSTM_FunctionTrainer_001
           --dill_file_name provaAnomaly.dill
           --action trainAndSaveDillToFile
 ````
 Example: trainAndSaveDillToFileFromInfo
 ````shell command
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
         python3 main.py 
           --action trainAndSaveDillToFileFromInfo 
           --info_file_name      Calib_RF_FunctionTrainer_001.json 
           --dill_file_name      calibrator001.dill 
           --df_csv_file_prefix "data/calibrator001"
  ````
  
Example: getInfoFromDillFile
````shell command
python3 main.py 
           --dill_file_name tmp_calibrator.dill 
           --action getInfoFromDillFile
 ````
 Example: saveTrainingAndTestingDataToCsv
 ````shell command
 python3 main.py 
           --id_sensor 4003 
           --begin_time "2019-08-01 00:00:00" 
           --end_time   "2019-08-20 00:00:00" 
           --feature_list "no_we-no_aux" 
           --label_list "no-o3" 
           --pollutant_label "no" 
           --df_csv_file_prefix "data/df_csv" 
           --action saveTrainingAndTestingDataToCsv
 ````
 Example: applyCalibrationSensorPollutantDillDf: this is a function for developers,
         usualy YOU MUST NOT RUN this method with do_persist_data==true
         it is intended for Dill testing only.
  ````shell command
 python3 main.py
           --id_sensor 4003
           --begin_time "2019-08-01 00:00:00"
           --end_time   "2019-08-20 00:00:00"
           --pollutant_label "no"
           --dill_file_name calibrator001.dill
           --do_persist_data false
           --interval_of_aggregation 10T
           --action applyCalibrationSensorPollutantDillDf
  ````
 
Example: trainToDB: generate LSTM model for sensor 4007 using a training dataset considering the calibration period between 2020-06 and 2021-06 for the NO pollutant
````shell command
 python3 main.py --id_sensor 4007 --begin_time "2020-06-01 00:00:00" --end_time "2021-06-01 00:00:00" --feature_list "no_we-no_aux-no2_we-no2_aux-ox_we-ox_aux-co_we-co_aux-temperature-humidity" --label_list "no" --pollutant_label "no" --trainer_class_name "Calib_LSTM_FunctionTrainer_001" --trainer_module_name "calib_LSTM_FunctionTrain" --action "trainToDB"
 ````
 Example: trainToDB: generate VRSVR model for sensor 4007 using a training dataset considering the calibration period between 2020-06 and 2021-06 for the O3 pollutant
 ````shell command
 python3 main.py --id_sensor 4007 --begin_time "2020-06-01 00:00:00" --end_time "2021-06-01 00:00:00" --feature_list "no_we-no_aux-no2_we-no2_aux-ox_we-ox_aux-co_we-co_aux-temperature-humidity" --label_list "o3" --pollutant_label "o3" --trainer_class_name "Calib_VR_SVR_FunctionTrainer_001" --trainer_module_name "calib_VR_SVR_FunctionTrain" --action "trainToDB"
 ````
 
 Example: applyDillsToSensor: apply/testing calibration models on sensor 4005 data form the beginning of June to the end of September 2020 with the d of the row in the sensor_calibraation table where we find the refernce to all the models that we need to apply for each sensor
 ````shell command
 python3 main.py --id_sensor 4005 --begin_time "2020-06-01 00:00:00" --end_time "2020-09-30 00:00:00" --action "applyDillsToSensor" --id_calibration=146
  ````
  
  
