import numpy as np
import tensorflow as tf

def project_point(point, camera_calibration):
    # vehicle frame to camera sensor frame.
    extrinsic = tf.reshape(camera_calibration.extrinsic.transform, [4, 4])
    vehicle_to_sensor = tf.linalg.inv(extrinsic)
    point1 = point
    point1 = np.concatenate((point1,np.ones([1,point1.shape[1]])),axis = 0)
    point_camera_frame = tf.einsum(
        'ij,jk->ik', vehicle_to_sensor, tf.constant(point1, dtype=tf.float32))

    u_d = - point_camera_frame[1] / point_camera_frame[0]
    v_d = - point_camera_frame[2] / point_camera_frame[0]

    # add distortion model here if you'd like.

    f_u = camera_calibration.intrinsic[0]
    f_v = camera_calibration.intrinsic[1]
    c_u = camera_calibration.intrinsic[2]
    c_v = camera_calibration.intrinsic[3]

    u_d = u_d * f_u + c_u
    v_d = v_d * f_v + c_v

    return np.array([u_d.numpy(), v_d.numpy()])