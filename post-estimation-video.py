import argparse
import os
import torch
import cv2
from torchvision import transforms
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, check_img_size, set_logging, increment_path
from utils.torch_utils import select_device
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from models.experimental import attempt_load

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # ���頛� FP32 璅∪��
    stride = int(model.stride.max())  # 璅∪��甇亙��
    imgsz = check_img_size(960, s=stride)  # 瑼Ｘ�亙�����憭批��
    return model, stride, imgsz

def process_frame(frame, model, device):
    # 隤踵�游�����憭批��
    image = letterbox(frame, 960, stride=64, auto=True)[0]
    
    # 撠�������頧������箏撐���
    image = transforms.ToTensor()(image).unsqueeze(0)  # 憓���� batch 蝬剖漲
    
    # 撠�������蝘餃����啗身���銝�
    image = image.to(device)
    
    # ������璅∪����脣��頛詨��
    with torch.no_grad():
        t1 = time.time()
        output = model(image)[0]
        t2 = time.time()
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        t3 = time.time()
        output = output_to_keypoint(output)
        t4 = time.time()
    
    # 撠�������敺�撘菟��頧������� NumPy ������銝西�������� BGR ��澆��
    nimg = image[0].permute(1, 2, 0).cpu().numpy() * 255
    nimg = nimg.astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    keypoints = []
    avg_positions = []
    # 蝜芾ˊ撉冽�園����菟��
    for idx in range(output.shape[0]):
        kpts = output[idx, 7:].T
        keypoints.append(kpts)
        # ������X���Y���璅�
        x_coords = kpts[0::3]
        y_coords = kpts[1::3]
        avg_x = np.mean(x_coords)
        avg_y = np.mean(y_coords)
        avg_positions.append([avg_x, avg_y])
        print(len(kpts))
        # print(f"Keypoints for object {idx}: {kpts}")
        plot_skeleton_kpts(nimg, kpts, 3)
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
     # 蝜芾ˊ撟喳��雿�蝵桅�����璅�蝐�
    for i, (avg_x, avg_y) in enumerate(avg_positions):
        cv2.circle(nimg, (int(avg_x), int(avg_y)), 5, (0, 0, 255), -1)
        cv2.putText(nimg, f'{i}', (int(avg_x), int(avg_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    return nimg, t2 - t1, t3 - t2, t4 - t3,np.array(avg_positions)

def check_movement(prev_positions, curr_positions, frame_counts, fps, threshold=5, time_threshold=1):
    movements = []
    for i, (prev, curr) in enumerate(zip(prev_positions, curr_positions)):
        distance = np.linalg.norm(np.array(prev) - np.array(curr))
        if distance < threshold:
            frame_counts[i] += 1
        else:
            frame_counts[i] = 0
            
        if frame_counts[i] >= fps * time_threshold:
            movements.append("stay")
        else:
            movements.append("move")
    return movements, frame_counts

def process_video(video_path, model, device, save_dir, log_interval=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video stream or file: {video_path}")
        return
    
    # ���敺�敶梁�����鞈�閮�
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # ��萄遣靽�摮�蝯�������鞈����憭�
    os.makedirs(save_dir, exist_ok=True)
    
    # 閮剔蔭敶梁��撖怠�亙��
    output_path = os.path.join(save_dir, 'result.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 雿輻�� H.264 蝺函Ⅳ
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # ���憪������脣漲璇�
    progress_bar = tqdm(total=total_frames, desc="Processing video", unit="frame")

    prev_positions = None
    frame_counts = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # ������瘥�銝�撟�
        nimg, inference_time, nms_time, post_process_time, avg_positions = process_frame(frame, model, device)
        
        if prev_positions is not None:
            if len(frame_counts) < len(avg_positions):
                frame_counts.extend([0] * (len(avg_positions) - len(frame_counts)))

            movements, frame_counts = check_movement(prev_positions, avg_positions, frame_counts, fps)
            for i, movement in enumerate(movements):
                cv2.putText(nimg, movement, (int(avg_positions[i][0]), int(avg_positions[i][1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        prev_positions = avg_positions
        
        # 蝣箔��撟�憭批�����頛詨�亙��憭批��銝����
        nimg = cv2.resize(nimg, (width, height))
        
        # 撖怠�亙蔣���
        out.write(nimg)
        
        # ��湔�圈�脣漲璇�
        progress_bar.update(1)
        
        # �����唳�����鞈�閮�嚗�瘥������� log_interval 撟������唬��甈�
        frame_count += 1
        if frame_count % log_interval == 0:
            print(f"Frame {frame_count}/{total_frames} - Done. ({(1E3 * inference_time):.1f}ms) Inference, ({(1E3 * nms_time):.1f}ms) NMS, ({(1E3 * post_process_time):.1f}ms) Post-processing")

    cap.release()
    out.release()
    progress_bar.close()
    print(f"saveto {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./lab-sit-edit.avi', help='source')
    parser.add_argument('--output', type=str, default='./w6-pose/video', help='output folder')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    
    set_logging()
    device = select_device(opt.device)
    
    # 頛���交芋���
    model, stride, imgsz = load_model(opt.weights, device)
    model = model.float()  # 蝣箔��璅∪��雿輻�冽筑暺���貊移摨�
    
    # ��萄遣靽�摮�蝯�������摮�鞈����憭�
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join(opt.output, current_time)
    
    # 閬���餉楝敺�
    video_path = opt.source
    process_video(video_path, model, device, save_dir)
