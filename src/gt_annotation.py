import time
import cv2
import numpy as np
import tqdm

from obs import ObsClient
from typing import Tuple

from elements.yolo import OBJ_DETECTION
from api.client import Client
from obs_ops.ObsOps import ObsOperation

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

def calc_yolo_data(hei, wid, box_list : list):
    label_x = (box_list[1] + box_list[3]) / (2 * wid)
    label_y = (box_list[0] + box_list[2]) / (2 * hei)
    label_w = (box_list[3] - box_list[1]) / wid
    label_h = (box_list[2] - box_list[0]) / hei
    return str(label_x) + ' ' + str(label_y) + ' ' + str(label_w) + ' ' + str(label_h)

if __name__ == '__main__':
    # 创建ObsClient实例
    obsClient = ObsClient(
        access_key_id='3WNUTDBEXWYV0FYRZ4EO',
        secret_access_key='Xm4TLCUrDEI9ZzRPQLPc0Fpn1T5MY1Qw0z9sJTej',
        server='https://obs.cn-north-4.myhuaweicloud.com'
    )
    # warnings.filterwarnings('ignore')
    url = 'https://2df181d9180d495487c0567b29adcfe9.apig.cn-north-4.huaweicloudapis.com/v1/infers/1b8baf9f-6f77-46ea-bf7e-d6abbde373ef'
    # 部署在线服务的url接口
    headers = {'x-auth-token': 'MIITdgYJKoZIhvcNAQcCoIITZzCCE2MCAQExDTALBglghkgBZQMEAgEwghGIBgkqhkiG9w0BBwGgghF5BIIRdXsidG9rZW4iOnsiZXhwaXJlc19hdCI6IjIwMjItMTItMTJUMDY6MzU6MzQuNTIwMDAwWiIsIm1ldGhvZHMiOlsicGFzc3dvcmQiXSwiY2F0YWxvZyI6W10sInJvbGVzIjpbeyJuYW1lIjoidGVfYWRtaW4iLCJpZCI6IjAifSx7Im5hbWUiOiJ0ZV9hZ2VuY3kiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9jc2JzX3JlcF9hY2NlbGVyYXRpb24iLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF91Y3MtaW50bCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Vjc19kaXNrQWNjIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZHNzX21vbnRoIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfY3NnIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZWNzX3Nwb3RfaW5zdGFuY2UiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9pdmFzX3Zjcl92Y2EiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9hX2NuLXNvdXRoLTRjIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZWNzX2thZTEiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9kZWNfbW9udGhfdXNlciIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2llZl9lZGdlYXV0b25vbXkiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9Lb29TZWFyY2hDT0JUIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfdmlwX2JhbmR3aWR0aCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Vjc19vbGRfcmVvdXJjZSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2R3c19wb2MiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF93ZWxpbmticmlkZ2VfZW5kcG9pbnRfYnV5IiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfY2JyX2ZpbGUiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lY3Nfa2MxX3VzZXJfZGVmaW5lZCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX21lZXRpbmdfZW5kcG9pbnRfYnV5IiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfbWFwX25scCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2VnX2NuIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZGNzX2RjczItcmVkaXM2LWdlbmVyaWMiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lY3NfdGhpcmRfaW1hZ2UiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9yZWRpczYtZ2VuZXJpYy1pbnRsIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZGNzX2RjczItZW50ZXJwcmlzZSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3BzdG5fZW5kcG9pbnRfYnV5IiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfbWFwX29jciIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Rsdl9vcGVuX2JldGEiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9vYnNfZHVhbHN0YWNrIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZWRjbSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3ZjcCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2NzYnNfcmVzdG9yZSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2l2c2NzIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfY3ZyIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZWNzX2M2YSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX211bHRpX2JpbmQiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF92cG5fdmd3IiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfc21uX2NhbGxub3RpZnkiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9laXBfcG9vbCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2FfYXAtc291dGhlYXN0LTNkIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfb3BfZ2F0ZWRfbGFrZWZvcm1hdGlvbl9iZXQiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9wcm9qZWN0X2RlbCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3NoYXJlQmFuZHdpZHRoX3FvcyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2NlciIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2NzYnNfcHJvZ3Jlc3NiYXIiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9nYV9jbiIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2lkdF9kbWUiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9jZXNfcmVzb3VyY2Vncm91cF90YWciLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lY3Nfb2ZmbGluZV9hYzciLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9ldnNfcmV0eXBlIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfYV9hZi1zb3V0aC0xYyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Vjc19pcjN4IiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZXZzX3Bvb2xfY2EiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9hX2NuLXNvdXRod2VzdC0yYiIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2NzZV9uYWNvcyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3Vjc19jaWEiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lY3Nfb2ZmbGluZV9kaXNrXzQiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9od2RldiIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3Nmc3R1cmJvIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfaW50bF9jb21wYXNzIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfaHZfdmVuZG9yIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfYV9jbi1ub3J0aC00ZSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2FfY24tbm9ydGgtNGQiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9kYXl1X2RsbV9jbHVzdGVyIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZWNzX2FjNyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2VwcyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2NjZV9tY3BfdGhhaSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2NzYnNfcmVzdG9yZV9hbGwiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9jb21wYXNzIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZWRzIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfc2VydmljZXN0YWdlX21ncl9kdG0iLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9hX2NuLW5vcnRoLTRmIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfZmNzX3BheSIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2NwaCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2FfYXAtc291dGhlYXN0LTFlIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfYV9ydS1tb3Njb3ctMWIiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9hX2FwLXNvdXRoZWFzdC0xZCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2dhIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfYV9hcC1zb3V0aGVhc3QtMWYiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9ybXMiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9zbW5fYXBwbGljYXRpb24iLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9nZWlwIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfcmFtIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfb3JnYW5pemF0aW9ucyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Vjc19ncHVfZzVyIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfb3BfZ2F0ZWRfbWVzc2FnZW92ZXI1ZyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3drc19rcCIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2Vjc19jNyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX2djYiIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3JpX2R3cyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX21hcF92aXNpb24iLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9lY3NfcmkiLCJpZCI6IjAifSx7Im5hbWUiOiJvcF9nYXRlZF9hX3J1LW5vcnRod2VzdC0yYyIsImlkIjoiMCJ9LHsibmFtZSI6Im9wX2dhdGVkX3JhbV9pbnRsIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfaWVmX3BsYXRpbnVtIiwiaWQiOiIwIn0seyJuYW1lIjoib3BfZ2F0ZWRfYWFkX2JldGFfaWRjIiwiaWQiOiIwIn1dLCJwcm9qZWN0Ijp7ImRvbWFpbiI6eyJuYW1lIjoic3RldmVuaW5zIiwiaWQiOiJmYjNiZjc1MTA3YjI0Y2E1OWJmNjEzZGExYzU4ZmNhNiJ9LCJuYW1lIjoiY24tbm9ydGgtNCIsImlkIjoiYjZhMjFiZWU0Njg0NDdkOGEzY2I4NTViNTE1MmQxZGIifSwiaXNzdWVkX2F0IjoiMjAyMi0xMi0xMVQwNjozNTozNC41MjAwMDBaIiwidXNlciI6eyJkb21haW4iOnsibmFtZSI6InN0ZXZlbmlucyIsImlkIjoiZmIzYmY3NTEwN2IyNGNhNTliZjYxM2RhMWM1OGZjYTYifSwibmFtZSI6Inh1d2VpIiwicGFzc3dvcmRfZXhwaXJlc19hdCI6IiIsImlkIjoiYzA5NmRhOTM4MjUzNGE4YThmZjBmYzUxNmYwZTAzNDIifX19MYIBwTCCAb0CAQEwgZcwgYkxCzAJBgNVBAYTAkNOMRIwEAYDVQQIDAlHdWFuZ0RvbmcxETAPBgNVBAcMCFNoZW5aaGVuMS4wLAYDVQQKDCVIdWF3ZWkgU29mdHdhcmUgVGVjaG5vbG9naWVzIENvLiwgTHRkMQ4wDAYDVQQLDAVDbG91ZDETMBEGA1UEAwwKY2EuaWFtLnBraQIJANyzK10QYWoQMAsGCWCGSAFlAwQCATANBgkqhkiG9w0BAQEFAASCAQAyvgiVZdgJe3RsBrpDJ3h36vUZz4LLhP8AgSvStUZzPgia1pueMpDsl2g+WEQzds2jyukYN2or3b2+0WIdRu1fm5-BPGHnz1imRDjeQbZghJo0LfZvtF9CU0IHLZlKnlaEa2WZgijDDfHJQNMrfsltG98i9fLPC1ND-woApj5CGxppKNTg28pK5EchznjleNvCQj39hKaQaipIawgsZ+k5z8vPktXYPuWJXoiRe34Q+ulmT7E35kaokUQ30MywDtssfk11qfKzrYxK6mfHC8vR4HYGHKYPKZRFNGv8KzmA6azMrf9-4Ey72xJO1OKmn4dEtnqJ4SfWqur8uMtipnNC'}

    for i in range(2):
        addr = './output/images/'
        op_imgfile = f'img_{i}'
        imgfile = op_imgfile +'.jpg'
        # imgfile_download(obsClient=obsClient,objectname=f'edge_capture_data/{imgfile}',
        #              localfile=f'./output/images/{imgfile}')
        filename = f'./output/images/{imgfile}'  # 推理图片路径

        img = cv2.imread(filename)

        results = Object_detector.detect(img)
        for result in results:
            className = result['label']
            score = result['score']
            ## 记录坐标
            [(x1, y1), (x2, y2)] = result['bbox']

        height, width = img.shape[:2]
        # files = {'images': (filename, open(filename, 'rb'), "image/jpeg")}
        # r = requests.post(url, files=files, headers=headers, verify=False)
        # outputs = json.loads(r.content)
        # print(outputs)  # 输出预测结果
        print("***************************************************")
        # with open(f'./output/labels/{op_imgfile}.txt', 'w', encoding='UTF-8') as op_file:
        #     for index, value in enumerate(outputs['detection_classes']):
        #         if value == 'car':
        #             op_file.write('1 ' + calc_yolo_data(height, width, outputs['detection_boxes'][index]) + '\n')
                    # outputs['detection_boxes'] --> boxes

        # infer_result_upload(obsClient=obsClient, objectkey=f'edgedata_infer/{op_imgfile}.txt',
        #                 localfile=f'./output/labels/{op_imgfile}.txt')

    # 关闭obsClient
    obsClient.close()