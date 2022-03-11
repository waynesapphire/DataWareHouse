import glob
import json
import os
import time

import cv2
import numpy as np

import lidar
import video_parser


class Lidar2Camera_Projecter:
    def __init__(self, lidar_rec_path, video_path, video_time_stamp, lidar_calib_file, camera_intrinsic_file, camera_extrinsic_file):
        self.lidar_rec_path = lidar_rec_path
        self.video_path = video_path
        self.video_time_stamp = video_time_stamp
        self.lidar_calib_file = lidar_calib_file
        self.camera_intrinsic_file = camera_intrinsic_file
        self.camera_extrinsic_file = camera_extrinsic_file

    def lidar_camer_parser(self, lidar_savedir, cam_image_save_dir):
        self.lidar_save_dir = lidar_savedir
        self.camera_save_dir = cam_image_save_dir
        print("parser lidar data......")
        vlp32 = lidar.Vlp32C(self.lidar_rec_path)
        vlp32.rec2frame(lidar_savedir)
        print("parser camera data......")
        video_par = video_parser.VideoParser(self.video_path, self.video_time_stamp)
        video_par.extract_frame_to(cam_image_save_dir)

    def undistort_image(self, camera_timestamp):
        with open(self.camera_intrinsic_file, 'r') as f:
            camera_calib = json.load(f)
        camera_matrix = np.array(camera_calib['center_camera_fov120-intrinsic']['param']['cam_K']['data'])
        dist = np.array(camera_calib['center_camera_fov120-intrinsic']['param']['cam_dist']['data'][0])
        image_distort = cv2.imread(f"{self.camera_save_dir}/{str(camera_timestamp)[:-3]}.jpg")
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (camera_calib['center_camera_fov120-intrinsic']['param']["img_dist_w"],camera_calib['center_camera_fov120-intrinsic']['param']["img_dist_h"]),0, (camera_calib['center_camera_fov120-intrinsic']['param']["img_dist_w"],camera_calib['center_camera_fov120-intrinsic']['param']["img_dist_h"]))
        image_undistort = cv2.undistort(image_distort, camera_matrix, dist, None, newcameramtx)
        # cv2.imwrite(f"/data/4T_disk/honda_dataset/undistort_test/{camera_timestamp}.jpg", image_undistort)
        return image_undistort, newcameramtx
        # return image_distort, camera_matrix

    def timestamp_match(self, lidar_savedir, cam_image_save_dir, thresh_hold):
        lidar_timestamp_ = glob.glob(f"{lidar_savedir}/*.npy")
        lidar_timestamp = list(map(lambda x: int(os.path.basename(x).strip(".npy")), lidar_timestamp_))
        lidar_timestamp.sort()
        camera_timestamp_ = glob.glob(f"{cam_image_save_dir}/*.jpg")
        camera_timestamp = list(map(lambda x: int(os.path.basename(x).strip(".jpg") + '000'), camera_timestamp_))
        camera_timestamp.sort()
        lidar_pos = 0
        camera_pos = 0
        for i in range(len(lidar_timestamp)):
            if lidar_timestamp[i] < camera_timestamp[0]:
                lidar_pos = i
        matched = []
        for i in range(lidar_pos, len(lidar_timestamp)):
            for j in range(camera_pos, len(camera_timestamp)-1):
                if lidar_timestamp[i] > camera_timestamp[j] and lidar_timestamp[i] < camera_timestamp[j+1]:
                    if lidar_timestamp[i] - camera_timestamp[j] < camera_timestamp[j+1] - lidar_timestamp[i]:
                        if lidar_timestamp[i] - camera_timestamp[j] < thresh_hold:
                            matched.append((lidar_timestamp[i], camera_timestamp[j]))
                        else:
                            break
                    else:
                        if camera_timestamp[j+1] - lidar_timestamp[i] < thresh_hold:
                            matched.append((lidar_timestamp[i], camera_timestamp[j+1]))
                            camera_pos += 1
                        else:
                            break
                if lidar_timestamp[i] > camera_timestamp[j] and lidar_timestamp[i] > camera_timestamp[j+1]:
                    continue
                if lidar_timestamp[i] < camera_timestamp[j]:
                    break
        return matched

    def project(self, matched, output_dir):
        for lidar_timestamp, camera_timestamp in matched:
            self.project_frame(lidar_timestamp, camera_timestamp, output_dir)

    def project_frame(self, lidar_timestamp, camera_timestamp, output_dir):
        image, intrinsic = self.undistort_image(camera_timestamp)
        pc = self.lidar2camera(lidar_timestamp)
        self.lidar_frame_project2image_distance(image, intrinsic, pc, output_dir)
        # self.lidar_frame_project2image_intensity(image, intrinsic, pc, output_dir)

    def lidar2camera(self, lidar_timestamp):
        with open(self.camera_extrinsic_file) as cam_ex_file:
            cam_ex = json.load(cam_ex_file)
        cam_extrinsic = np.array(cam_ex['center_camera_fov120-to-car_center-extrinsic']['param']['sensor_calib']['data'])
        with open(self.lidar_calib_file) as lidar_calib_file:
            lidar_calib = json.load(lidar_calib_file)
        lidar_extrinsic = np.array(lidar_calib['front_lidar-to-car_center-extrinsic']['param']['sensor_calib']['data'])
        lidar2camera_mtx = np.linalg.inv(cam_extrinsic) @ lidar_extrinsic @ np.array([[0., 1., 0., 0.], [-1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        lidar_pc = np.load(f"{self.lidar_save_dir}/{lidar_timestamp}.npy")
        lidar_in_cam = lidar2camera_mtx @ np.vstack((lidar_pc[0:3][:], np.ones_like(lidar_pc[1][:])))
        lidar_in_cam = np.vstack((lidar_in_cam[:3][:], lidar_pc[3][:]))

        lidar_in_cam = lidar_in_cam[:,lidar_in_cam[2] > 0]

        return lidar_in_cam

    def lidar_frame_project2image_intensity(self, image, intrinsic, pc, output_dir):
        pc_xyz_ = pc[:3, :]
        pc_xyz = pc_xyz_ / pc_xyz_[2,:]
        image_point = intrinsic @ pc_xyz
        image_point = image_point[:2,:]
        image_point = np.vstack((image_point, pc[3,:]))
        image_limit_w = np.logical_and(image_point[0, :] >= 0, image_point[0, :] <= image.shape[1])
        image_limit_h = np.logical_and(image_point[1, :] >= 0, image_point[1, :] <= image.shape[0])
        image_limit = np.logical_and(image_limit_h, image_limit_w)
        image_point = image_point[:, image_limit]
        intensity = np.log2(pc[3,image_limit] + 1).flatten()
        im_color = cv2.applyColorMap((intensity/np.max(intensity) * 255).astype(np.uint8), cv2.COLORMAP_JET)

        for idx in range(image_point.shape[1]):
            color = (int(im_color[idx][0][0]),int(im_color[idx][0][1]),int(im_color[idx][0][2]))
            cv2.circle(image, (round(image_point[0][idx]), round(image_point[1][idx])), 2, color, -1)
        cv2.imwrite(f"{output_dir}/{int(time.time()*1000000)}.jpg", image)


    def lidar_frame_project2image_distance(self, image, intrinsic, pc, output_dir):
        pc_xyz_ = pc[:3, :]
        pc_xyz = pc_xyz_ / pc_xyz_[2,:]
        image_point = intrinsic @ pc_xyz
        image_point = image_point[:2,:]
        image_point = np.vstack((image_point, pc[3,:]))
        image_limit_w = np.logical_and(image_point[0, :] >= 0, image_point[0, :] <= image.shape[1])
        image_limit_h = np.logical_and(image_point[1, :] >= 0, image_point[1, :] <= image.shape[0])
        image_limit = np.logical_and(image_limit_h, image_limit_w)
        image_point = image_point[:, image_limit]
        # distance
        cam_z = pc_xyz_[2,image_limit]
        im_color = cv2.applyColorMap((cam_z/200 * 255).astype(np.uint8), cv2.COLORMAP_JET)

        for idx in range(image_point.shape[1]):
            color = (int(im_color[idx][0][0]),int(im_color[idx][0][1]),int(im_color[idx][0][2]))
            cv2.circle(image, (round(image_point[0][idx]), round(image_point[1][idx])), 2, color, -1)
        cv2.imwrite(f"{output_dir}/{int(time.time()*1000000)}.jpg", image)








def main():

    projecter = Lidar2Camera_Projecter("/data/4T_disk/honda_dataset/16_33_35/sensors_record/front_lidar.dump.rec", "/data/4T_disk/honda_dataset/16_33_35/port_0_camera_0_2021_9_6_16_33_37.h264", "/data/4T_disk/honda_dataset/16_33_35/port_0_camera_0_2021_9_6_16_33_37.txt",
                                       "/data/4T_disk/honda_dataset/16_33_35/JP-002/front_lidar/front_lidar-to-car_center-extrinsic.json", "/data/4T_disk/honda_dataset/16_33_35/JP-002/center_camera_fov120/center_camera_fov120-intrinsic.json", "/data/4T_disk/honda_dataset/16_33_35/JP-002/center_camera_fov120/center_camera_fov120-to-car_center-extrinsic.json" )
    projecter.lidar_camer_parser("/data/4T_disk/honda_dataset/lidar_extract", "/data/4T_disk/honda_dataset/images_extract")

    matched_pair = projecter.timestamp_match(projecter.lidar_save_dir, projecter.camera_save_dir, 50000000)

    projecter.project(matched_pair, "/data/4T_disk/honda_dataset/lidar_project2image")




if __name__ == "__main__":
    main()
