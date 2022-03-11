import numpy as np
from struct import *
import copy
import open3d as o3d

class Lidar:
    def __init__(self):
        self.data = []
    


class Vlp32C(Lidar):
    # factor distance centimeter value to meter
    FACTOR_CM2M = 0.01

    # factor distance value to cm, each velodyne distance unit is 4 mm
    FACTOR_MM2CM = 0.4
    def __init__(self, path):
        super().__init__()
        self.recdata_path = path
        self.elevation = np.array([-25, -1, -1.667, -15.639, 
                                   -11.31, 0, -0.667, -8.843,
                                   -7.254,0.333, -0.333, -6.148,
                                   -5.333,1.333, 0.667, -4,
                                   -4.667, 1.667, 1, -3.667,
                                   -3.333,3.333, 2.333, -2.667,
                                   -3,7, 4.667, -2.333,
                                   -2,15, 10.333, -1.333])
        self.azimuth_offset = np.array([1.4, -4.2, 1.4, -1.4,
                                        1.4, -1.4, 4.2, -1.4,
                                        1.4, -4.2, 1.4, -1.4,
                                        4.2, -1.4, 4.2, -1.4,
                                        1.4, -4.2, 1.4, -4.2,
                                        4.2, -1.4, 1.4, -1.4,
                                        1.4, -1.4, 1.4, -4.2, 
                                        4.2, -1.4, 1.4, -1.4])

    def read_firing_data(self, data):
        block_id = data[0] + data[1]*256
        # 0xeeff is upper block
        assert block_id == 0xeeff

        azimuth_ = (data[2] + data[3] * 256) / 100
        azimuth_std = np.repeat(azimuth_, 32) 
        azimuth = azimuth_std + self.azimuth_offset

        firings = data[4:].reshape(32, 3)
        distances = firings[:, 0] + firings[:, 1] * 256
        intensities = firings[:, 2]
        return distances, intensities, azimuth

    def calc_precise_azimuth(self, azimuth):

        org_azi = azimuth.copy()

        precision_azimuth = []
        # iterate through each block
        for n in range(12): # n=0..11
            azimuth = org_azi.copy()
            try:
                # First, adjust for an Azimuth rollover from 359.99° to 0°
                if azimuth[n + 1][0] < azimuth[n][0]:
                    azimuth[n + 1][0] += 360.

                # Determine the azimuth Gap between data blocks
                azimuth_gap = azimuth[n + 1][0] - azimuth[n][0]
            except:
                azimuth_gap = azimuth[n][0] - azimuth[n-1][0]

            # iterate through each firing
            precise_azimuth = np.zeros_like(azimuth[0])
            for k in range(32):
                # Determine if you’re in the first or second firing sequence of the data block
                    # Interpolate
                precise_azimuth[k] = azimuth[n][k] + (azimuth_gap * 2.304 * k) / 55.296

                # if precise_azimuth > 361.:
                #     print("Error")
                # print(precise_azimuth)
            precision_azimuth.append(copy.deepcopy(precise_azimuth))
        for idx1 in range(12):
            for idx2 in range(32):
                if precision_azimuth[idx1][idx2] > 360:
                    precision_azimuth[idx1][idx2] -= 360
                if precision_azimuth[idx1][idx2] < 0:
                    precision_azimuth[idx1][idx2] += 360
        # precision_azimuth = np.array(precision_azimuth)
        return precision_azimuth

    def calc_cart_coord(self, distances, azimuth):
        # convert distances to meters
        distances = distances * self.FACTOR_MM2CM * self.FACTOR_CM2M

        # convert deg to rad
        longitudes = np.tile(self.elevation * np.pi / 180., 12).reshape(12, 32)
        latitudes = azimuth * np.pi / 180.

        hypotenuses = distances * np.cos(longitudes)

        X = (hypotenuses * np.sin(latitudes)).flatten()
        Y = (hypotenuses * np.cos(latitudes)).flatten()
        Z = (distances * np.sin(longitudes)).flatten()
        return X, Y, Z


    def rec2frame(self, save_dir):

        BYTE_PER_SEC = {
        "lidar": 1206*1507.041,
        "radar": 8568*40,
        "ins": 192*100,
        "canbus": 14*2174}

        with open(self.recdata_path, mode="rb") as f:
        # Read header
            try:
                magic_num = unpack("B", f.read(1))[0]
                topic_num = unpack("B", f.read(1))[0]
                print("topic_id", "topic_name")
                for i in range(topic_num):
                    topic_id = unpack("B", f.read(1))[0]
                    topic_max_length = unpack("I", f.read(4))[0]
                    topic_len = unpack("B", f.read(1))[0]
                    topic = f.read(topic_len).decode()
                    print(topic_id, topic)
                extra_len = unpack("I", f.read(4))[0]
                if extra_len > 0:
                    extra = f.read(extra_len)
            except:
                print("broken data")
                return       
    
        # Read data
            times = []
            byte = 0
            print("topic_id", "timestamp", "byte")
            while True:
                try:
                    for frame_num in range(10):
                        pc_frame = np.zeros((4, 0))
                        timestamp = 0
                        for i in range(150):
                            topic_id = unpack("B", f.read(1))[0]
                            timestamp_ = unpack("Q", f.read(8))[0]
                            if i == 149:
                                timestamp = timestamp_
                            cap_len = unpack("I", f.read(4))[0]
                            data = f.read(cap_len)
                            pc_blk = self.data_parse_struct(data)
                            pc_frame = np.hstack((pc_frame, pc_blk))
                        np.save(f"{save_dir}/{timestamp}.npy", pc_frame)
                        
            ##----------------------visualization test--------------------
                    # pcd = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(pc_frame[0:3].T)
                    # o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
                    # # color_vec = np.zeros((pc_scene_all.shape[1], 3))
                    # # color_vec[:, 2] = pc_scene_all[3, :] / np.max(pc_scene_all[3, :])
                    # # pcd.colors = o3d.utility.Vector3dVector(color_vec)
                    # o3d.visualization.draw_geometries([pcd])


                except:
                    break
                # times.append(time_ns)
                # byte += cap_len
                # print(topic_id, time_ns, cap_len)

    def data_parse_struct(self, data):
        if len(data) != 1206:
            raise Exception("error data length!!")
        
        lidar_data = data[:1200]
        lidar_timestamp = data[1200:1204]
        lidar_factory = data[1204:]

        lidar_data = np.frombuffer(lidar_data, dtype=np.uint8).astype(np.uint32).reshape(12, 100)
        lidar_timestamp = np.frombuffer(lidar_timestamp, dtype=np.uint32)
        lidar_factory = np.frombuffer(lidar_factory, dtype=np.uint8)
        dis_one_pack = []
        inten_one_pack = []
        azimuth_one_pack = []
        for i, blk in enumerate(lidar_data):
            dis_blk, inten_blk, azimuth_blk = self.read_firing_data(blk)
            dis_one_pack.append(dis_blk)
            inten_one_pack.append(inten_blk)
            azimuth_one_pack.append(azimuth_blk)
        azimuth_one_pack = self.calc_precise_azimuth(azimuth_one_pack)
        dis_one_pack = np.stack(dis_one_pack, axis=0)
        inten_one_pack = np.stack(inten_one_pack, axis=0)
        azimuth_one_pack = np.stack(azimuth_one_pack, axis=0)
        dis_mask = (dis_one_pack >= 200).flatten()
        # print(dis_one_pack)
        X, Y, Z = self.calc_cart_coord(dis_one_pack, azimuth_one_pack)
        pc = np.vstack((X, Y, Z, inten_one_pack.flatten().astype(np.float64)))
        pc = pc[:, dis_mask]
        # pc_zero_ = (pc == 0)
        # pc_zero = np.logical_not(np.logical_and(pc_zero_[0,:], np.logical_and(pc_zero_[1, :], pc_zero_[2, :])))
            # pc = pc[:, pc_zero]
        # pc_eliminate_ = np.logical_and((pc >= 0), (pc <= 0.5))
        
        return pc


def test():
    vlp32 = Vlp32C("/data/4T_disk/honda_dataset/16_33_35/sensors_record/front_lidar.dump.rec")
    vlp32.rec2frame("/data/4T_disk/honda_dataset/lidar_extract")

if __name__ == '__main__':
    test()
