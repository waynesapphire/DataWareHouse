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
    dataset = tf.data.TFRecordDataset(FILE_PATH, compression_type='')
    count = 0
    # pbar = tqdm(total=199)
    camera_dict = {}
    lidar_dict = {}
    # df_meta = pd.DataFrame()
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        break
    camera_calib = frame.context.camera_calibrations
    lidar_calib = frame.context.laser_calibrations
    scene_id = ''.join(re.findall(r'\d+\_', FILE_PATH))[0:-1]
    
    camera_dict['scene_id'] = str(scene_id)
    lidar_dict['scene_id'] = str(scene_id)
    
    df_camera = pd.DataFrame()
    df_lidar = pd.DataFrame()
    for i in range(5):
        camera_dict['channel'] = int(camera_calib[i].name)
        camera_dict['modality'] = str('camera')
        camera_dict['intrinsic'] = str(camera_calib[i].intrinsic)
        camera_dict['extrinsic'] = str(camera_calib[i].extrinsic).replace('\n',',')
        camera_dict['width_height'] = str([camera_calib[i].width, camera_calib[i].height])
        camera_dict['rolling_shutter'] = str(camera_calib[i].rolling_shutter_direction)

        lidar_dict['channel'] = int(lidar_calib[i].name)
        lidar_dict['modality'] = str('lidar')
        lidar_dict['beam_inclinations'] = str(lidar_calib[i].beam_inclinations)
        lidar_dict['beam_inclination_range'] = str([lidar_calib[i].beam_inclination_min, lidar_calib[i].beam_inclination_max])
        lidar_dict['extrinsic'] = str(lidar_calib[i].extrinsic).replace('\n',',')
        

        df_camera_tmp = pd.DataFrame.from_dict([camera_dict])
        df_camera = pd.concat([df_camera, df_camera_tmp])
        df_lidar_tmp = pd.DataFrame.from_dict([lidar_dict])
        df_lidar = pd.concat([df_lidar, df_lidar_tmp])

        
    write2sql(df_camera,df_lidar)
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
    pbar = tqdm(total= 10)
    for FILENAME in file_list:
        FILE_PATH = local_path +FILENAME
        translate_waymo2dataframe(FILE_PATH)
        pbar.update(1)
        pass
    pbar.close()

if __name__ == '__main__':
    main()
