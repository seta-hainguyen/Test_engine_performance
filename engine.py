from face_detection.face_detector import face_detector
from utils import *
import numpy as np
import argparse
import time
import cv2
import sys


def parse_args():
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_network', default='traced_resnet50', type=str, # mobile0.25 or resnet50
                        help='model which is used for detect faces')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    detector = face_detector(args.face_network, cpu=False)
    if args.face_network == "mobile0.25":
        skip_frame = 1
    else:
        skip_frame = 1

    biometrics, names = load_biometric_data()
    # cap_1 = cv2.VideoCapture("rtsp://admin:admin@123@192.168.81.201:554/profile2/media.smp")
    # cap_2 = cv2.VideoCapture("rtsp://admin:admin@123@192.168.81.202:554/profile2/media.smp")
    cap_1 = cv2.VideoCapture('./video/camera1.avi')
    # cap_2 = cv2.VideoCapture('./video/camera2.avi')
    # cap = cv2.VideoCapture("./video_data/demo_zoom_x3.avi")

    if not cap_1.isOpened():
        sys.exit('Failed to open camera 1!')
    # if not cap_2.isOpened():
    #     sys.exit('Failed to open camera 2!')
    
    frame_index = 0
    FPS_Max = 0
    FPS_Min = 100


    if os.path.exists('box_faces'):
        shutil.rmtree(r'box_faces')

    global_start_time = time.time()
    total_run_time = 0
    total_run_count = 0

    warmup_count = 10

    while True:
        ret_1, frame_1 = cap_1.read()
        ret_2 = True
        # ret_2, frame_2 = cap_2.read()

        if (not ret_1) or (not ret_2):
            break

        frame_index += 1

        # Check frame indx to process 
        if frame_index % skip_frame == 0:
            # folder1 = f"./box_faces/camera1/{frame_index}_box_faces/"
            # folder2 = f"./box_faces/camera2/{frame_index}_box_faces/"

            # os.makedirs(folder1, exist_ok=True)
            # os.makedirs(folder2, exist_ok=True)
            start_time = time.time()
            if ret_1:
                frame_1 = cam_predict(frame_1, f"camera1/{frame_index}", detector, biometrics, names, 0.98, 0.45)
            # if ret_2:
            #     frame_2 = cam_predict(frame_2, f"camera2/{frame_index}", detector, biometrics, names, 0.98, 0.45)

            run_time = time.time() - start_time
            
            if frame_index < warmup_count:
                continue

            # print(f'runtime: {run_time}')
            total_run_time += run_time
            total_run_count += 1

            fps = 1/run_time
            FPS_Max = max(FPS_Max, fps)
            FPS_Min = min(FPS_Min, fps)


    print("Maximum FPS Predict: ", FPS_Max)
    print("Minimum FPS Predict: ", FPS_Min)
    print("Avg time : ", total_run_time/total_run_count)
    print("Total time all: ", time.time() - global_start_time)
    cap_1.release()
    # cap_2.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()