#!/usr/bin/python3
"""trafair python code configuration file
"""
import psycopg2
import datetime
import json

"""Singleton for the db_connection"""
_db_conn = None;
#
# 
#

#
# 
#
def trafair_db_getConnection(application_name="a_trafair_application"):
  """get the trafair-db connection"""
  global _db_conn;
  if (_db_conn is None):
    connection_string="dbname='*****' user='*****' host='localhost' password='*****' application_name='{}'".format(application_name)
    #print("\nconnection_string: " + connection_string +"\n")
    _db_conn = psycopg2.connect(connection_string)
  return(_db_conn)


#
#
#
def trafair_db_getConfigurationById(conn, configuration_id):
  rv=None
  cur = conn.cursor()
  cur.execute(""" select 
                    application_code, configuration_json, creation_date, note
                  from applications_configurations
                    where id = %s
                  """, (configuration_id))
  record = cur.fetchone()
  if (not(record is None)):
    rv={
      "application_code": record[0],
      "configuration_json": record[1],
      "creation_date": record[2],
      "note": record[3]
    }
  cur.close()
  return(rv)
  
def trafair_db_getConfigurationId(conn, application_code, configuration_json_string):
  """Retrieve or save the configuration-ID for a given
  application/configuration parameters.

  Several applications in Trafair uses configurations that may
  change and in the project we need to track such configurations.

  :param conn: 
  the trafair db-connection if "None" trafair_db_getConnection will be invoked.

  :param application_code: 
  identifies the application that uses the given configuration.

  :param configuration_json_string:
  full application configuration in json format as string.
  This string will be normalized.

  Usage example:
  --------------
   #
   #
   application_code="trafair_db_config.tests";
   conn = trafair_db_getConnection(application_code)
   configuration_json_string = ""
     { "username": "acorni", "email": "alberto.corni@unimore.it",
       "first_name": "Alberto", "last_name": "CORNI" }
   ""
   #
   configuration_id = trafair_db_getConfigurationId(conn, application_code,configuration_json_string)
   #
   #
  
 works on this table:
   CREATE TABLE applications_configurations (
       id                  serial PRIMARY KEY,
       application_code    varchar(255),
       configuration_json  text,
       creation_date       TIMESTAMP,
       note text
   );
"""
  configuration_json_string_normalized = ""
  if (configuration_json_string != ""):
    configuration_json = json.loads(configuration_json_string)
    configuration_json_string_normalized = \
      json.dumps(configuration_json, indent=0, sort_keys=True)
  #
  rv = -1
  #
  cur = conn.cursor()
  qs = """
   select id 
     from applications_configurations
    where application_code = %s
      and configuration_json = %s
  """
  cur.execute(qs, (application_code, configuration_json_string_normalized))
  record = cur.fetchone()
  if (record is None):
    # insert
    creation_date = str(datetime.datetime.now())
    cur.execute(""" INSERT INTO applications_configurations  
                     (application_code, configuration_json, creation_date, note)
                    values (%s, %s, %s, %s)
                    RETURNING id
                    """, (application_code, configuration_json_string_normalized, creation_date, ""))
    conn.commit()
    rv = cur.fetchone()[0]
  else:
    rv = record[0]
  return(rv)
  cur.close()
  




class Trafair_units_of_measure_manager():
    def __init__(self, units):
      """
      units: unit of measures in the format like trafair_db_get_units_of_measure
      """
      rv = { }
      for pollutant in units.keys():
        rv[pollutant] = {}
        for unit in units[pollutant]['conversions']:
          rv[pollutant][unit['from']] = unit['factor']
      self.index = rv
      # print(" ---  db_units_of_measure_factor_index: " + str(rv));
      _db_units_of_measure_factor_index = rv
    def getFactor(self, pollutant, from_unit_of_measure):
      if (not pollutant in self.index):
        raise Exception("Pollutant not recognized ["+pollutant+"]")
      if (not from_unit_of_measure in self.index[pollutant]):
        raise Exception("Not recognized from_unit_of_measure ["+from_unit_of_measure+"] for pollutant ["+pollutant+"]")
      return(self.index[pollutant][from_unit_of_measure])

_db_units_of_measure_factor_manager = None;
def trafair_db_get_units_of_measure_factor(pollutant, from_unit_of_measure):
  global _db_units_of_measure_factor_manager
  if (_db_units_of_measure_factor_manager == None):
    _db_units_of_measure_factor_manager = \
      Trafair_units_of_measure_manager(trafair_db_get_units_of_measure())
  return(_db_units_of_measure_factor_manager.getFactor(pollutant, from_unit_of_measure))


def productWithNone(value_may_be_none, value):
  rv = value_may_be_none
  if (rv != None):
    rv = rv * value
  return(rv)

def trafair_db_get_units_of_measure():
  rv = {
    'no': {
      'unit_of_measure' : 'ug/m^3'
      , 'conversions' : [
        { 'from': 'ppb' , 'factor': 1.250 }
      ]
    } 
    , 'no2': {
      'unit_of_measure' : 'ug/m^3'
      , 'conversions' : [
        { 'from': 'ppb' , 'factor': 1.912 }
      ]
    }
    , 'o3': {
      'unit_of_measure' : 'ug/m^3'
      , 'conversions' : [
        { 'from': 'ppb' , 'factor': 2.0 }
      ]
    }
    , 'co': {
      'unit_of_measure' : 'ug/m^3'
      , 'conversions' : [
        { 'from': 'mg/m^3' , 'factor': 1000 }
        , { 'from': 'ppm' , 'factor': 1160 }
        , { 'from': 'ppb' , 'factor': 1.160 }
      ]
    }
  }
  return(rv)
