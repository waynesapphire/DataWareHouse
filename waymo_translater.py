import os
from matplotlib import image
from numpy.lib.arraysetops import isin
import tensorflow as tf
import math
import numpy as np
import itertools

#tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json
import glob
from tqdm import tqdm


def show_camera_image(camera_image, camera_labels, layout, cmap=None):
  """Show a camera image and the given camera labels."""

  ax = plt.subplot(*layout)

  # Draw the camera labels.
  for camera_labels in frame.camera_labels:
    # Ignore camera labels that do not correspond to this camera.
    if camera_labels.name != camera_image.name:
      continue

    # Iterate over the individual labels.
    for label in camera_labels.labels:
      # Draw the object bounding box.
      ax.add_patch(patches.Rectangle(
        xy=(label.box.center_x - 0.5 * label.box.length,
            label.box.center_y - 0.5 * label.box.width),
        width=label.box.length,
        height=label.box.width,
        linewidth=1,
        edgecolor='red',
        facecolor='none'))

  # Show the camera image.
  plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
  plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
  plt.grid(False)
  plt.axis('off')


def project_point(point, camera_calibration):
  # vehicle frame to camera sensor frame.
  extrinsic = tf.reshape(camera_calibration.extrinsic.transform, [4, 4])
  vehicle_to_sensor = tf.linalg.inv(extrinsic)
  point1 = point
  point1 = np.vstack((point1, np.ones((1, point1.shape[1]))))
  point_camera_frame = tf.einsum('ij,jk->ik', vehicle_to_sensor, tf.constant(point1, dtype=tf.float32))
  u_d = - point_camera_frame[1] / point_camera_frame[0]
  v_d = - point_camera_frame[2] / point_camera_frame[0]
  
  # add distortion model here if you'd like.
  f_u = camera_calibration.intrinsic[0]
  f_v = camera_calibration.intrinsic[1]
  c_u = camera_calibration.intrinsic[2]
  c_v = camera_calibration.intrinsic[3]
  u_d = u_d * f_u + c_u
  v_d = v_d * f_v + c_v

  return [u_d.numpy(), v_d.numpy()]

def plot_range_image_helper(data, name, layout, vmin = 0, vmax=1, cmap='gray'):
  """Plots range image.

  Args:
    data: range image data
    name: the image title
    layout: plt layout
    vmin: minimum value of the passed data
    vmax: maximum value of the passed data
    cmap: color map
  """
  plt.subplot(*layout)
  plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
  plt.title(name)
  plt.grid(False)
  plt.axis('off')

def get_range_image(laser_name, return_index):
  """Returns range image given a laser name and its return index."""
  return range_images[laser_name][return_index]

def show_range_image(range_image, layout_index_start = 1):
  """Shows range image.

  Args:
    range_image: the range image data from a given lidar of type MatrixFloat.
    layout_index_start: layout offset
  """
  range_image_tensor = tf.convert_to_tensor(range_image.data)
  range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
  lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
  range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
                                tf.ones_like(range_image_tensor) * 1e10)
  range_image_range = range_image_tensor[...,0] 
  range_image_intensity = range_image_tensor[...,1]
  range_image_elongation = range_image_tensor[...,2]
  plot_range_image_helper(range_image_range.numpy(), 'range',
                   [8, 1, layout_index_start], vmax=75, cmap='gray')
  plot_range_image_helper(range_image_intensity.numpy(), 'intensity',
                   [8, 1, layout_index_start + 1], vmax=1.5, cmap='gray')
  plot_range_image_helper(range_image_elongation.numpy(), 'elongation',
                   [8, 1, layout_index_start + 2], vmax=1.5, cmap='gray')

def rgba(r):
  """Generates a color based on range.

  Args:
    r: the range value of a given point.
  Returns:
    The color for a given range
  """
  c = plt.get_cmap('jet')((r % 20.0) / 20.0)
  c = list(c)
  c[-1] = 0.5  # alpha
  return c

def plot_image(camera_image):
  """Plot a cmaera image."""
  plt.figure(figsize=(20, 12))
  plt.imshow(tf.image.decode_jpeg(camera_image.image))
  plt.grid("off")

def plot_points_on_image(projected_points, camera_image, rgba_func,
                         point_size=5.0):
  """Plots points on a camera image.

  Args:
    projected_points: [N, 3] numpy array. The inner dims are
      [camera_x, camera_y, range].
    camera_image: jpeg encoded camera image.
    rgba_func: a function that generates a color from a range value.
    point_size: the point size.

  """
  plot_image(camera_image)

  xs = []
  ys = []
  colors = []

  for point in projected_points:
    xs.append(point[0])  # width, col
    ys.append(point[1])  # height, row
    colors.append(rgba_func(point[2]))

  plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")

def project_point(point, camera_calibration):
  # vehicle frame to camera sensor frame.
  extrinsic = tf.reshape(camera_calibration.extrinsic.transform, [4, 4])
  vehicle_to_sensor = tf.linalg.inv(extrinsic)
  point1 = point
  point1 = np.vstack((point1, np.ones((1, point1.shape[1]))))
  point_camera_frame = tf.einsum('ij,jk->ik', vehicle_to_sensor, tf.constant(point1, dtype=tf.float32))
  u_d = - point_camera_frame[1] / point_camera_frame[0]
  v_d = - point_camera_frame[2] / point_camera_frame[0]
  
  # add distortion model here if you'd like.
  f_u = camera_calibration.intrinsic[0]
  f_v = camera_calibration.intrinsic[1]
  c_u = camera_calibration.intrinsic[2]
  c_v = camera_calibration.intrinsic[3]
  u_d = u_d * f_u + c_u
  v_d = v_d * f_v + c_v

  return [u_d.numpy(), v_d.numpy()]
  

def boundary_gen(object, extend_x, extend_y):
  '''
  function: boundary_gen
    This function is to calculate a boundary x, y, z for a object, which is applied in lidar object points elimination.
  ---------------------------
  input: 
    object: a element in waymo dataset tool output frame.laser_label
    extend_x: extend boundary in x direction, it means boundary in x direction change from [x_min, x_max] to [x_min - extend_x, x_max + extend_x]
    extend_y: extend boundary in y direction.
  return:
    a list consist of boundary [x_min, x_max, y_min, y_max, z_min, z_max]
  '''
  x = object.box.center_x
  y = object.box.center_y
  z = object.box.center_z
  length = object.box.length + extend_x
  width = object.box.width + extend_y
  height = object.box.height
  heading = object.box.heading
  x_y_corner = np.array([[0.5 * length, 0.5 * length, - 0.5 * length, - 0.5 * length], [0.5 * width, - 0.5 * width, - 0.5 * width, 0.5 * width]])
  x_y_corner = np.array([[np.math.cos(heading), -np.math.sin(heading)], [np.math.sin(heading), np.math.cos(heading)]]) @ x_y_corner + np.array([[x, x, x, x], [y, y, y, y]])
  boundary = [np.min(x_y_corner[0]), np.max(x_y_corner[0]), np.min(x_y_corner[1]), np.max(x_y_corner[1]), z - height * 0.5, z + height * 0.5]

  return boundary



def boundary_coordinate_gen(object):
  '''
  function : boundary_coordinate_gen
    This function generate eight boundary coordinate for a object
  ------------------------------------------------------------------
  input :
    object
  return:
    a numpy array contains eight coordinate in vehicle frame. format : [[x0, x1, ..., x7], [y0, y1, ..., y7], [z0, z1, ..., z7]]
    points order : 
                     0--------1
                    /|       /|
                   2--------3 |
                   | 4------|-5
                   |/       |/ 
                   6--------7
  '''
  boundary = boundary_gen(object, 0, 0)
  boundary_cor = np.array([[boundary[1], boundary[1], boundary[0], boundary[0], boundary[1], boundary[1], boundary[0], boundary[0]], 
                           [boundary[3], boundary[2], boundary[3], boundary[2], boundary[3], boundary[2], boundary[3], boundary[2]], 
                           [boundary[5], boundary[5], boundary[5], boundary[5], boundary[4], boundary[4], boundary[4], boundary[4]]])
  return boundary_cor


def save_context_to_parameter(frame_context, output_path):
  if output_path[-1] =='/':
    output_path = output_path[0:-1]

  intrinsic = frame_context.camera_calibrations[0].intrinsic
  f_u = intrinsic[0]
  f_v = intrinsic[1]
  c_u = intrinsic[2]
  c_v = intrinsic[3]
  parameter = dict()
  parameter["camera_intrinsic"] = np.array([[f_u, 0.0, c_u, 0.0], [0, f_v, c_v, 0.0], [0.0, 0.0, 1.0, 0.0]])
  parameter["camera_dist"] = np.array([])
  parameter['img_dist_w'] = frame_context.camera_calibrations[0].width
  parameter['img_dist_h'] = frame_context.camera_calibrations[0].height
  parameter['top_center_lidar_to_center_camera_extrinsic'] = np.array([])
  parameter['top_center_lidar_to_car_center_extrinsic'] = np.array([])
  parameter['car_center_to_top_center_lidar_extrinsic'] = np.array([])
  for i in parameter:
    if isinstance(parameter[i], np.ndarray):
      parameter[i] = parameter[i].tolist()

  with open(output_path + '/parameter.json', 'w') as json_file:
    json.dump(parameter, json_file)

def if_object_in_camera_fov(object, camera_calibration, threshold):
  '''
  function: if_object_in_camer_fov
    using for judge if an object could be found in specific camera.
  ---------------------------------------------------
  Args:
    object: waymo laser object label.
    camer_calibration: waymo dataset specific camera calibration.
  Return:
    if_in_image: if object in image. True or False.
  '''
  boundary_object_cor = boundary_coordinate_gen(object)
  image_cor = np.array(project_point(boundary_object_cor, camera_calibration))
  if_in_col = np.logical_and(image_cor[0,:] >= threshold, image_cor[0,:] <= (camera_calibration.width - threshold))
  if_in_row = np.logical_and(image_cor[1,:] >= threshold, image_cor[1,:] <= (camera_calibration.height - threshold))
  if_in_image = np.max(np.logical_and(if_in_col, if_in_row))

  return if_in_image

def if_in_image_numpy(object_center_vehicle_frame, sizes, yaws, camera_calibrations):
  '''
  not finished
  function if_in_image_numpy
    This function is used to accelerate the vehicle elimination process. So there are some meta data input directely.
  -------------------------------------------------------------------------
  Args:
    Plese refer to the invoke. All input are numpy array for parallel.
  Returns:
    A mask stands for whether an object in image.
  '''
  raise Exception("not finished!")
  # obj_num = len(sizes)
  # object_corner_x_0 =    sizes[0, :] / 2
  # object_corner_x_1 =  - sizes[0, :] / 2
  # object_corner_y_0 =    sizes[1, :] / 2
  # object_corner_y_1 =  - sizes[1, :] / 2
  # object_corner_z_0 =    sizes[2, :] / 2
  # object_corner_z_1 =  - sizes[2, :] / 2
  # object_corner_0 = np.vstack((object_corner_x_0, object_corner_y_0, object_corner_z_0))
  # object_corner_1 = np.vstack((object_corner_x_0, object_corner_y_0, object_corner_z_1))
  # object_corner_2 = np.vstack((object_corner_x_0, object_corner_y_1, object_corner_z_0))
  # object_corner_3 = np.vstack((object_corner_x_0, object_corner_y_1, object_corner_z_1))
  # object_corner_4 = np.vstack((object_corner_x_1, object_corner_y_0, object_corner_z_0))
  # object_corner_5 = np.vstack((object_corner_x_1, object_corner_y_0, object_corner_z_1))
  # object_corner_6 = np.vstack((object_corner_x_1, object_corner_y_1, object_corner_z_0))
  # object_corner_7 = np.vstack((object_corner_x_1, object_corner_y_1, object_corner_z_1))
  # object_corner_all = np.hstack((object_corner_0, object_corner_1, object_corner_2, object_corner_3, object_corner_4, object_corner_5, object_corner_6, object_corner_7))
  # object_corner_all = np.array([[np.math.cos(heading), -np.math.sin(heading)], [np.math.sin(heading), np.math.cos(heading)]])


def translate_waymo2sensetime(FILENAME, output_path):
  if output_path[-1] == '/':
    output_path = output_path[0:-1]
  if not os.path.exists(FILENAME):
    raise Exception("{} doesn't exists!".format(FILENAME))
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  if not os.path.exists(output_path + '/images'):
    os.mkdir(output_path + '/images')
  if not os.path.exists(output_path + '/lidar_data'):
    os.mkdir(output_path + '/lidar_data')
  if not os.path.exists(output_path + '/lidar_object'):
    os.mkdir(output_path + '/lidar_object')

  dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
  count = 0
  pbar = tqdm(total=199)
  for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    if count == 0:
      save_context_to_parameter(frame.context, output_path)
      vehicle2camera_extrinsic = np.linalg.inv(np.array(frame.context.camera_calibrations[0].extrinsic.transform).reshape(4,4))
      count += 1

    
    object_center_vehicle_frame = np.ones((3, 0))
    sizes = np.ones((3, 0)) # length, width, height
    yaws = []
    ids = []
    types = []
    for laser_label in frame.laser_labels:
      if not if_object_in_camera_fov(laser_label, frame.context.camera_calibrations[0], 20):
        continue
      object_center_vehicle_frame = np.hstack((object_center_vehicle_frame, np.array([laser_label.box.center_x, laser_label.box.center_y, laser_label.box.center_z]).reshape(3, -1)))
      yaws.append(laser_label.box.heading)
      sizes = np.hstack((sizes, np.array([laser_label.box.length, laser_label.box.width, laser_label.box.height]).reshape(3, -1)))
      ids.append(laser_label.id)
      types.append(laser_label.type)
    yaws = np.array(yaws)
    ids = np.array(ids)
    types = np.array(types, dtype=np.int)

    object_center_vehicle_frame = np.vstack((object_center_vehicle_frame, np.ones((1, object_center_vehicle_frame.shape[1]))))
    object_center_camera_frame = np.einsum('ij,jk->ik', vehicle2camera_extrinsic, object_center_vehicle_frame)[0:3, :] 

    object_json = []
    object_des = {"Cam3Dpoints" : [], "id" : 1, "pry_ZYX" : [], "size" : [], "type" : 0}

    # if_in_image_mask = if_in_image_numpy(object_center_vehicle_frame, sizes, yaws, frame.context.camera_calibrations[0])
    front_mask = object_center_camera_frame[0, :] + sizes[0, :] / 2 >= 0
    object_center_camera_frame = object_center_camera_frame[:, front_mask]
    yaws = yaws[front_mask]
    ids = ids[front_mask]
    types = types[front_mask]
    sizes = sizes[:, front_mask]

    for idx in range(len(ids)):
      object_des["Cam3Dpoints"] = np.array([-object_center_camera_frame[1, idx], -object_center_camera_frame[2, idx], object_center_camera_frame[0, idx]]).tolist().copy()
      # object_des["Cam3Dpoints"] = np.array([object_center_camera_frame[0, idx], object_center_camera_frame[1, idx], object_center_camera_frame[2, idx]]).tolist().copy()
      object_des["id"] = ids[idx]
      # object_des["pry_ZYX"] = np.array([0, -yaws[idx], 0]).tolist().copy()
      object_des["pry_ZYX"] = np.array([-yaws[idx], 0, 0]).tolist().copy()
      object_des["size"] = sizes[[1, 2, 0], idx].tolist().copy()
      object_des["type"] = int(types[idx])
      object_json.append(object_des.copy())
    
    with open(output_path + '/lidar_object/{}.json'.format(frame.timestamp_micros * 100), 'w') as json_file:
      json.dump(object_json, json_file)

    images = sorted(frame.images, key=lambda i:i.name)



    image_save = tf.image.decode_jpeg(images[0].image).numpy()
    image_save = image_save[..., [2, 1, 0]]
    image_save = np.ascontiguousarray(image_save)
    # cv2.imshow('waymo_test_show_before_undistortion', image_save)
    
    f_u = frame.context.camera_calibrations[0].intrinsic[0]
    f_v = frame.context.camera_calibrations[0].intrinsic[1]
    c_u = frame.context.camera_calibrations[0].intrinsic[2]
    c_v = frame.context.camera_calibrations[0].intrinsic[3]
    k_1 = frame.context.camera_calibrations[0].intrinsic[4]
    k_2 = frame.context.camera_calibrations[0].intrinsic[5]
    k_3 = frame.context.camera_calibrations[0].intrinsic[8]
    p_1 = frame.context.camera_calibrations[0].intrinsic[6]
    p_2 = frame.context.camera_calibrations[0].intrinsic[7]
    camera_matrix = np.array([[f_u, 0.0, c_u], [0.0, f_v, c_v], [0.0, 0.0, 1.0]])
    dist = np.array([k_1, k_2, p_1, p_2, k_3])
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(camera_matrix, dist,(frame.context.camera_calibrations[0].width,frame.context.camera_calibrations[0].height),0,(frame.context.camera_calibrations[0].width,frame.context.camera_calibrations[0].height))
    image_save = cv2.undistort(image_save, camera_matrix, dist, None, newcameramtx)

    
    cv2.imwrite(output_path + '/images/{}.jpg'.format(frame.timestamp_micros * 100), image_save)

    #cv2.imshow('waymo_test_show_after_undistortion', image_save)
    #cv2.waitKey(0)

    pbar.update(1)
    pass
  pbar.close()


def main():
  # FILENAME = '/data/4T_disk2/waymo/training/segment-8700094808505895018_7272_488_7292_488_with_camera_labels.tfrecord'
  file_list = glob.glob('/data/4T_disk2/waymo/training/*.tfrecord')
  OUTPUT_PATH = '/data/4T_disk/waymo/waymo_sensetime_test/training_alpha=0'
  for FILENAME in file_list:
    translate_waymo2sensetime(FILENAME, OUTPUT_PATH + '/' + os.path.splitext(os.path.basename(FILENAME))[0])

if __name__ == "__main__":
    main()
    
    
