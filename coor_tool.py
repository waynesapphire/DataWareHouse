import numpy as np

def cam_to_image(point_camera_frame, camera_calibration):
    """

    Args:
        point_camera_frame (_type_): _description_
        camera_calibration (_type_): _description_

    Returns:
        _type_: _description_
    """
    f_u   = camera_calibration.intrinsic[0]
    f_v   = camera_calibration.intrinsic[1]
    c_u   = camera_calibration.intrinsic[2]
    c_v   = camera_calibration.intrinsic[3]
    u_d =  -1*point_camera_frame[1] / point_camera_frame[0]
    v_d =  -1*point_camera_frame[2] / point_camera_frame[0]
    camera_intrinsic = np.array([[f_u, 0, c_u],
                                 [0, f_v, c_v],
                                 [0, 0, 1]])
    u_d = u_d * f_u + c_u
    v_d = v_d * f_v + c_v

    return np.array([u_d, v_d])

def cam_to_frame(points_xyz, cam_extrinsic):
    """using points and extrinsic project to ego frame

    Args:
        points_xyz (_type_): --- 3 x N
        cam_extrinsic (_type_): ---  ４ x 4

    Returns:
        _type_: _description_
    """

    points_xyz1 = np.vstack(
            (points_xyz, np.ones((1, points_xyz.shape[1]))))
    extrinsic = np.array(cam_extrinsic).reshape([4, 4])
    points_frame_xyz = extrinsic @ points_xyz1
    
    return points_frame_xyz[:3,:]

def frame_to_global(points_xyz, frame_pose):
    """_summary_

    Args:
        points_xyz (_type_): _description_
        frame_pose (_type_): _description_

    Returns:
        _type_: _description_
    """

    points_xyz1 = np.vstack(
        (points_xyz, np.ones((1, points_xyz.shape[1]))))
        
    points_global_xyz1 = np.array(
            frame_pose).reshape(4, 4) @ points_xyz1
    
    return points_global_xyz1[0:3, :]

def global_to_frame(points_xyz, frame_pose):
    '''
    input: points_xyz   ---3 x N
           frame_pose  ---４ x 4
    '''
    points_xyz1 = np.concatenate((points_xyz,np.ones([1,points_xyz.shape[1]])),axis = 0)
    frame_xyz1 = np.linalg.inv(np.array(frame_pose).reshape(4, 4)) @ points_xyz1
    return frame_xyz1[0:3,:]

def frame_to_cam(self, points_xyz, cam_extrinsic):
    '''
    input: points_xyz     ---3 x N
           cam_extrinsic  ---４ x 4
    '''
    points_xyz1 = np.concatenate((points_xyz,np.ones([1,points_xyz.shape[1]])),axis = 0)

    extrinsic = np.array(cam_extrinsic).reshape([4, 4])
    vehicle_to_sensor = np.linalg.inv(extrinsic)

    point_camera_frame = vehicle_to_sensor @ points_xyz1

    return point_camera_frame[0:3,:]



