import time
import cv2
import numpy as np
import tqdm

from obs import ObsClient
from typing import Tuple

from elements.yolo import OBJ_DETECTION
from api.client import Client
from obs_ops.ObsOps import ObsOperation

video = 'datasets/dataset.mp4'
outpath = './output/img_'
# 创建ObsClient实例
obsClient = ObsClient(
    access_key_id='3WNUTDBEXWYV0FYRZ4EO',
    secret_access_key='Xm4TLCUrDEI9ZzRPQLPc0Fpn1T5MY1Qw0z9sJTej',
    server='https://obs.cn-north-4.myhuaweicloud.com'
)

obsOps = ObsOperation(bucketname='its-ec')

Object_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush' ]
#
# Object_classes = ['car', 'person']

Object_colors = list(np.random.rand(80,3)*255)
Object_detector = OBJ_DETECTION('C:\\Users\\MorningStar\\Desktop\\ITS-EC\\Edge01\\src\\weights\\yolov5x.pt', Object_classes)


def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    cv2.imwrite(address, image)

def data_upload(bucketname, localfile, objectkey):
    try:
        from obs import PutObjectHeader
        print(f'frame uploading to {bucketname}/{objectkey}')
        resp = obsClient.putFile(bucketname, objectkey, localfile)

        if resp.status < 300:
            print('upload frame successfully')
            print('requestId:', resp.requestId)
            print('etag:', resp.body.etag)
            print('versionId:', resp.body.versionId)
            print('storageClass:', resp.body.storageClass)
            # print('objectkey:', objectkey)
        else:
            print('errorCode:', resp.errorCode)
            print('errorMessage:', resp.errorMessage)
    except:
        import traceback

        print(traceback.format_exc())


def calc_yolo_data(hei, wid, box_list : list):
    label_x = (box_list[1] + box_list[3]) / (2 * hei)
    label_y = (box_list[0] + box_list[2]) / (2 * wid)
    label_w = (box_list[3] - box_list[1]) / hei
    label_h = (box_list[2] - box_list[0]) / wid
    return str(label_x) + ' ' + str(label_y) + ' ' + str(label_w) + ' ' + str(label_h)

# path = ./output
def create_obj_name_file(path, bucketname = 'its-ec'):
    with open(f'{path}/obj.names', 'w', encoding='UTF-8') as op_file:
        for obj in Object_classes:
            op_file.write(obj + '\n')
    objectkey_objnames = f'edge_capture_data/obj.names'
    data_upload(bucketname=bucketname, localfile=f'{path}/obj.names', objectkey=objectkey_objnames)

def create_obj_data_file(path, class_num, names_addr, train, valid, bucketname = 'its-ec'):
    with open(f'{path}/obj.data', 'w', encoding='UTF-8') as op_file:
        op_file.write(f'classes = {class_num} \n names = {names_addr} \n train = {train} \n valid = {valid} \n')

    objectkey_objdata = f'edge_capture_data/obj.data'
    data_upload(bucketname=bucketname, localfile=f'{path}/obj.data', objectkey=objectkey_objdata)
def create_train_txt_file(train_num, bucketname = 'its-ec'):
    with open(f'./output/train.txt', 'w', encoding='UTF-8') as op_file0:
        for i in range(train_num):
            train_addr = f'train/img_{i}.jpg'
            op_file0.write(train_addr + '\n')
    data_upload(bucketname=bucketname, localfile=f'./output/train.txt', objectkey='edge_capture_data/train.txt')

def create_valid_txt_file(val_num, frame_num, bucketname = 'its-ec'):
    with open(f'./output/valid.txt', 'w', encoding='UTF-8') as op_file1:
        for i in range(val_num, frame_num):
            val_addr = f'val/img_{i}.jpg'
            op_file1.write(val_addr + '\n')
    data_upload(bucketname=bucketname, localfile=f'./output/valid.txt', objectkey='edge_capture_data/valid.txt')

def capture(
        stream: str = video,
        frame_skip: int = 0,
        window_size: Tuple = (1280, 720),
        bucketname: str = 'its-ec'
):
    ## 以下开始capture操作
    print(f'capture: from {stream}. sampleing rate: {frame_skip}')

    ## 开始视频流采集
    cap = cv2.VideoCapture(stream)
    ret, frame = cap.read()

    ## 获取视频帧数
    if frame_skip == 0:
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (frame_skip + 1)

    ## 初始化capture与annotation进度条
    progress_cap = tqdm.tqdm(range(int(num_frames) + 1), unit='decoded frames')
    train_num = int(num_frames * 0.8)
    i = 0
    ## 开始对每一帧图片进行infer操作
    while ret:
        ## 初化检测个数
        frame = cv2.resize(frame, window_size)

        save_image(image=frame, addr=outpath, num=i)
        if i <= train_num:
            objectkey_img = f'edge_capture_data/train/img_{i}.jpg'
            train_addr = f'train/img_{i}.jpg'
            objectkey_label = f'edge_capture_data/train/img_{i}.txt'

            with open(f'./output/train.txt', 'w', encoding='UTF-8') as op_file0:
                op_file0.write(train_addr + '\n')
        else:
            objectkey_img = f'edge_capture_data/val/img_{i}.jpg'
            val_addr = f'val/img_{i}.jpg'
            objectkey_label = f'edge_capture_data/val/img_{i}.txt'

            with open(f'./output/valid.txt', 'w', encoding='UTF-8') as op_file1:
                op_file1.write(val_addr + '\n')



        data_upload(bucketname=bucketname,localfile=f'{outpath}{i}.jpg', objectkey=objectkey_img)

        results = Object_detector.detect(frame)

        with open(f'./output/labels/img_{i}.txt', 'w', encoding='UTF-8') as op_file:
            for result in results:
                className = result['label']
                id = Object_classes.index(className)
                # score = result['score']
                ## 记录坐标
                [(x1, y1), (x2, y2)] = result['bbox']
                box_list = [x1, y1, x2, y2]
                op_file.write(f'{id} ' + calc_yolo_data(hei = window_size[1], wid = window_size[0], box_list = box_list) + '\n')
                # outputs['detection_boxes'] --> boxes

        data_upload(bucketname=bucketname, localfile=f'./output/labels/img_{i}.txt', objectkey=objectkey_label)
        ## 更新cap注释条
        progress_cap.update(1)
        print()


        # cv2.imshow(stream, frame)
        i += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # 获取下一帧
        next_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_skip
        # CV_CAP_PROP_POS_FRAMES 基于next_frame进行帧索引
        ## 跳帧
        cap.set(1, next_frame)
        ## 读取下一帧
        ret, frame = cap.read()
    # data_upload(bucketname=bucketname, localfile=f'./output/train.txt', objectkey='edge_capture_data/train.txt')
    data_upload(bucketname=bucketname, localfile=f'./output/valid.txt', objectkey='edge_capture_data/valid.txt')
    data_upload(bucketname=bucketname, localfile=f'./output/obj.names', objectkey='edge_capture_data/obj.names')
    path = './output'
    create_obj_data_file(path)
    obsClient.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # capture(frame_skip=20)
    # path = './output'
    # create_obj_name_file(path)
    create_train_txt_file(train_num=1140)
    create_valid_txt_file(val_num=1140, frame_num=1425)