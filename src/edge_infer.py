import time
import cv2
import numpy as np
import tqdm

from obs import ObsClient
from typing import Tuple

from elements.yolo import OBJ_DETECTION
from api.client import Client
from obs_ops.ObsOps import ObsOperation

# Object_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#                 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#                 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#                 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#                 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#                 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#                 'hair drier', 'toothbrush' ]

Object_classes = ['car', 'person']

Object_colors = list(np.random.rand(80,3)*255)
Object_detector = OBJ_DETECTION('C:\\Users\\MorningStar\\Desktop\\ITS-EC\\Edge01\\src\\weights\\best.pt', Object_classes)


video = 'datasets/dataset-Trim.mp4'

HOST = '127.0.0.1'
PORT = 6000
# client = Client(host=HOST, port=PORT)
# client.connect()

# 创建ObsClient实例
obsClient = ObsClient(
    access_key_id='3WNUTDBEXWYV0FYRZ4EO',
    secret_access_key='Xm4TLCUrDEI9ZzRPQLPc0Fpn1T5MY1Qw0z9sJTej',
    server='https://obs.cn-north-4.myhuaweicloud.com'
)



def run(
        stream: str = video,
        frame_skip: int = 1,
        host: str = '127.0.0.1',
        file: str = 'data.txt',
        save_to: str = None,
        eval_iterval: int = 10,
        window_size: Tuple = (1280, 720),
        debug: bool = True
):
    ## 是否要存储输出
    if save_to:
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        recorder = cv2.VideoWriter(save_to, fourcc, 25, window_size)

    ## 以下开始capture操作
    print(f'capture: from {stream}.')
    client = Client(host=host, port=PORT)
    client.connect()
    ## 加载模型

    # local_model = mnSSDm.mnSSD(model, threshold)

    infer_times_per_frame = []
    epoch_CC=[]

    ## 开始视频流采集
    cap = cv2.VideoCapture(stream)
    ret, frame = cap.read()

    ## 获取视频帧数
    if frame_skip == 0:
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (frame_skip + 1)

    ## 初始化capture与annotation进度条
    progress_cap = tqdm.tqdm(range(int(num_frames)), unit=' decoded frames')
    total_CC = []
    ## 开始对每一帧图片进行infer操作
    i = 0
    t = 0
    with open(f'./{file}', 'w', encoding='UTF-8') as op_file:
        while ret:

            CC = 0
            results, send_delay, recive_delay, infer_delay = client.send(frame)
            local_infer_time = infer_delay

            for result in results:
                className = result['label']
                score = result['score']
                CC += score
                ## 记录坐标
                ## shape[0]为图片高，shape[1]为图片长，shape[3]为图片通道数
                [(x1, y1), (x2, y2)] = result['bbox']
                display_str = f'{className}({score*100:.2f}%)'
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, display_str, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            total_CC.append((CC / len(results)))
            total_infer_times = local_infer_time
            total_delay = total_infer_times
            frame = cv2.resize(frame, window_size)
            op_file.write(f'{CC / len(results)} \n')

            if debug:
                cv2.rectangle(frame, (10,30), (235,100), (255,255,255), -1)
                ## 获取当前帧号

                cv2.putText(frame, f'frame num:{str(cap.get(cv2.CAP_PROP_POS_FRAMES))}', (15,45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
                ## 显示分辨率
                cv2.putText(frame, f'resolution: {str(window_size[0])} x {str(window_size[1])}', (15,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
                cv2.putText(frame, f'FPS: {int(1.0/total_delay*1000)}', (15,75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
                ## 显示检测数量
                cv2.putText(frame, f'Cumu Conf: {CC:.3f}', (15,90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

            infer_times_per_frame.append(total_infer_times)
            ## 更新cap注释条
            progress_cap.update(1)
            print()
            if i % eval_iterval == 0:
                t += 1
                epoch_CC.append(np.mean(total_CC))
                print(f'average cumulative confidence in epoch {t}: {np.mean(total_CC)}')
                # total_CC = []
            ## 若要存储
            if save_to:
                recorder.write(frame)


            cv2.imshow(stream, frame)
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

        cv2.destroyAllWindows()
        if save_to:
            recorder.release()
    
    return  epoch_CC, total_CC


if __name__ == '__main__':
    total_cc0 = run(host='127.0.0.1', file='data0.txt')[1]
    total_cc1 = run(host='127.0.0.2', file='data1.txt')[1]
    # total_cc0 = [0.44000000613076345, 0.42682738009071536, 0.4155083328343573, 0.527190475875423, 0.5545000007996956, 0.5457619058943931,
    #              0.5323154720699504, 0.5278511846940669, 0.5540011909308414, 0.5105833310633898, 0.4909361084609751, 0.5207380940516789,
    #              0.5494285698447909, 0.5228591260574167, 0.4980813513731673, 0.5620853136574466, 0.5338214261723417, 0.48402976012300875,
    #              0.45566269769555046, 0.46933531572539655, 0.5321547624965509, 0.4933928574834551, 0.4889015840313265, 0.49684999826645093,
    #              0.5135615084142913, 0.52487499602139, 0.5380714279200349]
    #
    # total_cc1 = [0.6649999916553497, 0.6810000002384186, 0.698999997973442, 0.587750000009934, 0.5748333302636942, 0.5813333349923293,
    #              0.544333333770434, 0.5172523808195477, 0.5320999973764022, 0.5835333338876565, 0.6152499971290429, 0.5109166711568832,
    #              0.5337500050663948, 0.5337333352367082, 0.530100003182888, 0.5426499888300895, 0.5325999981164932, 0.4843166693548361,
    #              0.4852999942501387, 0.52003332922856, 0.5444999955594539, 0.5472500014305115, 0.5731999969482422, 0.44494999736547475,
    #              0.4530000006159147, 0.5224999969204266, 0.513250000278155]

    import  matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(total_cc0)), total_cc0, '->', label='before trained')
    plt.plot(np.arange(len(total_cc1)), total_cc1, '-*', label='after trained')
    plt.xlabel('frame')
    plt.ylabel('Cumulative Confidence')
    plt.legend()
    plt.show()