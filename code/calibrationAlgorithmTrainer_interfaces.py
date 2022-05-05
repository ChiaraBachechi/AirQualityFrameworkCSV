from interface import *

"""Definition of the interface for CalibartionAlgorithm and CalibartionAlgorithmTrainer"""
class CalibartionAlgorithmTrainer_interface(metaclass=Interface):

    @abstractfunc
    def init(self,info_dictionary):
        """
        info_dictionary is a "json" dictionary like this:
          {
            "algorithm_parameters": {
              "hyper_parameters": {
                "leaves": 10,
                "trees": 1500
              }
            },
            "dates": {
              "end": "2019-08-20 00:00:00",
              "start": "2019-08-01 00:00:00"
            },
            "feat_order": [
              "no_we",
              "no_aux",
              "o3_we",
              "o3_aux"
            ],
            "label_order": [
              "no",
              "o3"
            ],
            "id_sensor": "4003",
            "interval": "10T",
            "pollutant_label": "no",
            "test_size": 0.2,
            "trainer_class_name": "CalibrationAlgorithm_RF_FunctionTrainer_001",
            "trainer_module_name": "calibrationAlgorithm_RF_FunctionTrain"
          }

        """
        self.info_dictionary=info_dictionary
        self.calibrator=None

    @abstractfunc
    def getCalibrator(self):
        return self.calibrator

    @abstractfunc
    def doTrain(self,train_features, train_labels):
        """
        trains and produces a calibrator
        --
        train_features:
          is a datafrome with the feature declared in 
            info_dictionary.feat_order
        train_labels:
          is a datafrome with the labels declared in 
            info_dictionary.label_order
        """
        self.calibrator=None

class CalibartionAlgorithm_interface(metaclass=Interface):
    #
    def init(self,info_dictionary):
        self.info_dictionary=info_dictionary
        
    @abstractfunc
    def apply_df(self, data_frame_in):
        """
        DataFrame:  apply_df(DataFrame: in)
        multiple row processor.
        Compute calibration for several rows.
        for the single line
         in: only the required features
             in the order specified in "getInfo"
         returns: NaN if there is something wrong
                 else a Double
        """
    #
    #
    @abstractfunc
    def get_info(self):
        """
        Dictionary: get_info().
        See dief_2019_trafair/code/sensor_low_cost/0note.txt
        for the reference example of such dictionary format.
        """

