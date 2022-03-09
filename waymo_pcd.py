import os
import re
import tensorflow as tf
import numpy as np
import pandas as pd
import pymysql
import sys
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset
from os import close
import pymysql
import sqlalchemy
from sqlalchemy import create_engine
import sys




host="106.15.102.18"
db = 'waymo'
user = 'root'
password = 'some_pass'

def write2sql(df_camera, df_lidar):
    
    try:
        conn = pymysql.connect(host=host,user=user, password=password, db=db, charset='utf8')
    except pymysql.err.OperationalError as e:
        print('Error is '+str(e))
        sys.exit()
        
    try:
        engine = create_engine(str(r"mysql+pymysql://%s:" + '%s' + "@%s/%s?charset=utf8") % (user, password, host, db))
    except sqlalchemy.exc.OperationalError as e:
        print('Error is '+str(e))
        sys.exit()
    except sqlalchemy.exc.InternalError as e:
        print('Error is '+str(e))
        sys.exit()
        
    try:   
        # df = pd.read_sql(sql, con=conn)
        df_camera.to_sql('test_camera', con = engine, if_exists = 'append', index = False)
        df_lidar.to_sql('test_lidar', con = engine, if_exists = 'append', index = False)

    except pymysql.err.ProgrammingError as e:
        print('Error is '+str(e))
        sys.exit()
    conn.close()
    
    # return df
    # # print(df.shape[0])
    # # conn.close()
    # # print('ok')
    
def translate_waymo2dataframe(FILE_PATH):
    dataset = tf.data.TFRecordDataset(FILE_PATH, compression_type = '')
    count = 0
    # pbar = tqdm(total=199)
    pcd_dict = {}
    