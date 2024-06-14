import argparse
import os
import torch
import cv2
from torchvision import transforms
import numpy as np
from datetime import datetime
from tqdm import tqdm
import time
import requests
import line
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, check_img_size, set_logging
from utils.torch_utils import select_device
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from models.experimental import attempt_load

global post_button, pre_movement, no_person_frames, person_leaved, person_entered
post_button = 1
pre_movement = 'init'
no_person_frames = 0
person_entered = False
person_leaved = False

#read config
with open('config.json') as config_file:
    config = json.load(config_file)

yolo_api_ip = config['yolo_api_ip']
rtsp_source = config['rtsp_source']

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # Load FP32 model
    stride = int(model.stride.max())  # Model stride
    imgsz = check_img_size(960, s=stride)  # Check image size
    return model, stride, imgsz

def process_frame(frame, model, device):
    # Resize frame
    image = letterbox(frame, 960, stride=64, auto=True)[0]
    
    # Convert to tensor
    image = transforms.ToTensor()(image).unsqueeze(0)  # Add batch dimension
    
    # Send to device
    image = image.to(device)
    
    # Inference
    with torch.no_grad():
        t1 = time.time()
        output = model(image)[0]
        t2 = time.time()
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        t3 = time.time()
        output = output_to_keypoint(output)
        t4 = time.time()
    
    # Convert image back to BGR format
    nimg = image[0].permute(1, 2, 0).cpu().numpy() * 255
    nimg = nimg.astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    keypoints = []
    avg_positions = []
    # Process keypoints
    for idx in range(output.shape[0]):
        kpts = output[idx, 7:].T
        keypoints.append(kpts)
        # Calculate average X and Y positions
        x_coords = kpts[0::3]
        y_coords = kpts[1::3]
        avg_x = np.mean(x_coords)
        avg_y = np.mean(y_coords)
        avg_positions.append([avg_x, avg_y])
        plot_skeleton_kpts(nimg, kpts, 3)
    
    # Draw average positions
    for i, (avg_x, avg_y) in enumerate(avg_positions):
        cv2.circle(nimg, (int(avg_x), int(avg_y)), 5, (0, 0, 255), -1)
        cv2.putText(nimg, f'{i}', (int(avg_x), int(avg_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    return nimg, t2 - t1, t3 - t2, t4 - t3, np.array(avg_positions)

def check_movement(prev_positions, curr_positions, frame_counts, fps, threshold=3, time_threshold=1):
    movements = []
    lock=True
    for i, (prev, curr) in enumerate(zip(prev_positions, curr_positions)):
        distance = np.linalg.norm(np.array(prev) - np.array(curr))
        if distance < threshold:
            frame_counts[i] += 1
        else:
            frame_counts[i] = 0
        if frame_counts[i] >= fps * 3:
            movements.append("ambulance")
            lock=False  
        if frame_counts[i] >= fps * time_threshold and lock:
            movements.append("stay")
            # lock=True 
        elif lock:
            movements.append("move")
    return movements, frame_counts

#request api server(Raspi)
def make_request(data):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        response = requests.post(yolo_api_ip, json=data)
        if response.status_code == 200:
            print(f"Data sent successfully at {current_time}")
        else:
            print(f"Failed to send data at {current_time}, status code: {response.status_code}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred at {current_time}: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred at {current_time}: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred at {current_time}: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred at {current_time}: {req_err}")

def process_rtsp(rtsp_url, model, device, log_interval=10):
    global post_button, pre_movement, no_person_frames, person_leaved, person_entered

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error opening RTSP stream: {rtsp_url}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    prev_positions = None
    frame_counts = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        nimg, inference_time, nms_time, post_process_time, avg_positions = process_frame(frame, model, device)
        
        
        if len(avg_positions) == 0:
            no_person_frames += 1
            if no_person_frames >= fps and not person_leaved:
                person_leaved = True
                person_entered = False
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print('send to line',current_time)
                cv2.imwrite('output.jpg', nimg)
                line.lineNotify(current_time+'關燈節能',nimg,'6632','11825389')
                data = {
                    "time": current_time,
                    "case": "light_off_sound"
                }
                make_request(data)                
                no_person_frames = 0
        else:
            no_person_frames = 0
        
        if prev_positions is not None:
            if len(frame_counts) < len(avg_positions):
                frame_counts.extend([0] * (len(avg_positions) - len(frame_counts)))

            movements, frame_counts = check_movement(prev_positions, avg_positions, frame_counts, fps)
            for i, movement in enumerate(movements):
                # print(movement)
                cv2.putText(nimg, movement, (int(avg_positions[i][0]), int(avg_positions[i][1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                if pre_movement == movement:
                    post_button = 0
                else:
                    post_button = 1
                    pre_movement ='0'

                if movement == "ambulance" and post_button:
                    pre_movement = movement
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print('send to line',current_time)
                    cv2.imwrite('output.jpg', nimg)
                    line.lineNotify(current_time+'呼叫救護車',nimg,'6632','11825389')
                    data = {
                        "time": current_time,
                        "case": "ambulance_sound"
                    }
                    make_request(data)
        
        if len(avg_positions) == 1 and not person_entered:
            person_entered = True
            person_leaved = False
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('send to line',current_time)
            cv2.imwrite('output.jpg', nimg)
            line.lineNotify(current_time+'電燈已開',nimg,'6362','11087939')
            data = {
                "time": current_time,
                "case": "light_on_sound"
            }
            make_request(data)
        prev_positions = avg_positions
        
        # Resize frame to original size
        nimg = cv2.resize(nimg, (width, height))
        
        # Show the frame
        cv2.imshow('RTSP Stream', nimg)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Print log every log_interval frames
        frame_count += 1
        if frame_count % log_interval == 0:
            print(f"Frame {frame_count} - Done. ({(1E3 * inference_time):.1f}ms) Inference, ({(1E3 * nms_time):.1f}ms) NMS, ({(1E3 * post_process_time):.1f}ms) Post-processing")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=rtsp_source, help='RTSP source')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    
    set_logging()
    device = select_device(opt.device)
    
    # Load model
    model, stride, imgsz = load_model(opt.weights, device)
    model = model.float()  # Ensure FP32 model
    
    # Process RTSP stream
    rtsp_url = opt.source
    process_rtsp(rtsp_url, model, device)
