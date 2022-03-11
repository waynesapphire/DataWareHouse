import cv2
from tqdm import tqdm

class VideoParser:
    def __init__(self, video_path, timestamp_file):
        self.video_path = video_path
        self.timestamp_file = timestamp_file

    def extract_frame_to(self, out_dir):
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened() == False:
            print("Error opening video file!")

        with open(self.timestamp_file, 'r') as timestamp_file:
            time_stamp_content = timestamp_file.readlines()
        timestamp_list = list(map(lambda x: x[6:].strip(), time_stamp_content))
        with tqdm(total = len(timestamp_list)) as pbar:
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True :
                    cv2.imwrite("{}/{}.jpg".format(out_dir, timestamp_list[count]), frame)
                    count += 1
                    pbar.update(1)
                else:
                    break

def video_test():
    video_par = VideoParser("/data/4T_disk/honda_dataset/16_33_35/port_0_camera_0_2021_9_6_16_33_37.h264", "/data/4T_disk/honda_dataset/16_33_35/port_0_camera_0_2021_9_6_16_33_37.txt")
    video_par.extract_frame_to("/data/4T_disk/honda_dataset/images_extract")

if __name__ == "__main__":
    video_test()
