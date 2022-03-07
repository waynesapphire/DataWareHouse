import os
import re
import tensorflow as tf
import numpy as np
import pandas as pd
import pymysql
import sys
import tqdm
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

def write2sql(df):
    
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
        df.to_sql('test_table', con = engine, if_exists = 'append', ignore_index = True)
    except pymysql.err.ProgrammingError as e:
        print('Error is '+str(e))
        sys.exit()
    conn.close()
    
    # return df
    # # print(df.shape[0])
    # # conn.close()
    # # print('ok')

def translate_waymo2dataframe(FILE_PATH):
    dataset = tf.data.TFRecordDataset(FILE_PATH, compression_type='')
    count = 0
    # pbar = tqdm(total=199)
    scene_context_dict = {}
    df_context = pd.DataFrame()
    for data in dataset:
        if count == 0:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            
            camera_calib = frame.context.camera_calibrations
            laser_calib = frame.context.laser_calibrations
            stats = frame.context.stats
            scene_id = ''.join(re.findall(r'\d+\_', FILE_PATH))[0:-1]
            
            scene_context_dict['scene_id'] = str(scene_id)
            scene_context_dict['camera_calibrations'] = str(camera_calib).replace('\n',',')
            scene_context_dict['laser_calibrations'] = str(laser_calib).replace('\n',',')
            scene_context_dict['stats'] = str(stats).replace('\n',',')
            
            
            df_context = pd.DataFrame.from_dict([scene_context_dict])
        else:
            break
        
    write2sql(df_context)
    return
    


def main():
    # file_list = ['segment-15832924468527961_1564_160_1584_160_with_camera_labels.tfrecord']
    file_list = ['segment-15832924468527961_1564_160_1584_160_with_camera_labels.tfrecord',
                'segment-16102220208346880_1420_000_1440_000_with_camera_labels.tfrecord',
                'segment-33101359476901423_6720_910_6740_910_with_camera_labels.tfrecord',
                'segment-54293441958058219_2335_200_2355_200_with_camera_labels.tfrecord',
                'segment-57132587708734824_1020_000_1040_000_with_camera_labels.tfrecord',
                'segment-80599353855279550_2604_480_2624_480_with_camera_labels.tfrecord',
                'segment-141184560845819621_10582_560_10602_560_with_camera_labels.tfrecord',
                'segment-169115044301335945_480_000_500_000_with_camera_labels.tfrecord',
                'segment-175830748773502782_1580_000_1600_000_with_camera_labels.tfrecord',
                'segment-183829460855609442_430_000_450_000_with_camera_labels.tfrecord']
    
    local_path = '/home/PJLAB/weixingjian/shared_data/training/'
    
    # OUTPUT_PATH = '/data/4T_disk/waymo/waymo_sensetime_test/training_alpha=0'
    for FILENAME in file_list:
        FILE_PATH = local_path +FILENAME
        translate_waymo2dataframe(FILE_PATH)

if __name__ == '__main__':
    main()
