from base64 import encode
from pathlib import Path
from pyexpat import model
import socket
import os
import sys
import struct
import threading
import time
import cv2
import json
from cv2 import threshold
import numpy as np
from torch import le

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) 
from elements.yolo import OBJ_DETECTION


Object_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush' ]

# Object_classes = ['car', 'person']

Object_colors = list(np.random.rand(80,3)*255)
Object_detector = OBJ_DETECTION('C:\\Users\\MorningStar\\Desktop\\ITS-EC\\Edge01\\src\\weights\\yolov5s.pt', Object_classes)



loaded_models = {}
model_in_use = ''

class Server:
    def __init__(self):
        # 设置TCP服务端的socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置重复使用
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 绑定端口
        self.sock.bind(('127.0.0.2', 6000))
        # 监听端口
        self.sock.listen(1)
       
    
    def run(self):
        while True:
            # 等待客户端连接
            print('waiting for connection...')
            conn, addr = self.sock.accept()
            print('connected from:', addr)
            # 接收并处理客户端发送的数据
            ProcessClient(conn).run()

class ProcessClient(threading.Thread):
    def __init__(self, conn):
        threading.Thread.__init__(self)
        self.conn = conn
        
    def run(self):
        t = 0
        #global img
        while True:
            send_ts0 = time.time()
            data = self.conn.recv(4)
            send_ts1 = time.time()

            send_head_delay = (send_ts1 - send_ts0) * 1000
            if not data:
                break
            ## 获取图片尺寸
            length = struct.unpack('i', data)
            length = length[0]
            print(f'data length: {length}')
            # 存放最终图片的数据
            buf = b''
            send_body_delay = 0
            while length:
                # 接收图片数据
                send_ts2 = time.time()
                temp_size = self.conn.recv(length)
                send_ts3 = time.time()

                send_body_delay += ((send_ts3-send_ts2) * 1000)
                if not temp_size:
                    break
                # 每次减去收到的数据大小
                length -= len(temp_size)
                # 将收到的数据拼接到img_data中
                buf += temp_size
            if buf == b'':
                print('img_data is empty')
                break
            # print('img_data:', buf)
            # 将收到的数据解码成图片
            img_data = np.frombuffer(buf, np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            #send_ts1 = time.time()
            send_delay = send_head_delay + send_body_delay

            # start infer
            print("start edge infer")
            infer_ts0 = time.time()
            results = Object_detector.detect(img)
            infer_ts1 = time.time()
            infer_delay = (infer_ts1 - infer_ts0) * 1000
            print("edge infer finished")

            # 结果编码
            results_data = json.dumps(results)
            print("edge result packageing...")
            # 打包信息
            final_data = struct.pack('iff', len(results_data), infer_delay, send_delay) + results_data.encode('utf-8')
            # 发送结果
            print("edge result sending...")
            self.conn.send(final_data)


        


def start_server():
    server = Server()
    server.run()


if __name__ == '__main__':
    start_server()

