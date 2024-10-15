import json
import socket
import os
import sys
import struct
from time import time
import traceback
from typing import Tuple

import cv2
import numpy as np


class Client(object):
    def __init__(self, host='127.0.0.1', port=6000, resolution=(640, 360), quality=100):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.resolution = resolution
        self.quality  = quality
    
    def connect(self):
        try:
            self.sock.connect((self.host, self.port))
            return True
        except Exception as e:
            traceback.print_exc()
            print('connect failed')
            return False
    
    def send(self, data):
        # try:
        # start = time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        # frame = cv2.imread(data)
        frame = cv2.resize(data, self.resolution)
        result, imgencode = cv2.imencode('.jpg', data, encode_param)
        # print(frame)
        data = np.array(imgencode)
        stringData = data.tostring()
        length = len(stringData)
        # print(f'frame length: {length}')
        all_data = struct.pack('i', length) + stringData
        # 计算传输时间
        #start = time()
        # 先发送图片编码后的长度
        #self.sock.send(str.encode(str(length)).ljust(16))
        # 再发送图片编码后的数据
        self.sock.send(all_data)
        #end = time()
        #send_delay = (end - start) * 1000
        # print(f'send delay: {(end - start) * 1000}ms')
        # 接收服务端发送的数据
        ## 获取图片尺寸
        recive_start_1 = time()
        data = self.sock.recv(12)
        recive_end_1 = time()

        recive_head_delay = (recive_end_1 - recive_start_1) * 1000

        results_length, infer_delay, send_delay= struct.unpack('iff', data)
        # length = length[0]
        # 存放最终结果
        results_buf = b''
        recive_body_delay = 0
        while results_length:
            # 接收图片数据
            recive_start_2 = time()
            temp_size = self.sock.recv(results_length)
            recive_end_2 = time()

            recive_body_delay += ((recive_end_2 - recive_start_2) * 1000)

            if not temp_size:
                break
            # 每次减去收到的数据大小
            results_length -= len(temp_size)
            # 将收到的数据拼接到img_data中
            results_buf += temp_size
        if results_buf == b'':
            print('results_data is empty')
            return False
        # recive_end = time()
        # buf = np.fromstring(buf, dtype=np.uint8)
        results = json.loads(results_buf.decode('utf-8'))
        
        recive_delay = recive_body_delay + recive_head_delay
        # print('boxes:', buf[0]['boxes'])
        # print('scores:', buf[0]['scores'])
        # print('class_ids:', buf[0]['class_ids'])
        # print('labels:', buf[0]['labels'])

        return results, send_delay, recive_delay, infer_delay

        # except Exception as e:
        #    traceback.print_exc()
        #    print('send failed')
        #    # return False


        

if __name__ == '__main__':
    client = Client()
    client.connect()
    client.send(cv2.imread('test.jpg'))    

