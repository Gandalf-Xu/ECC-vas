from obs import ObsClient
from obs_ops.ObsOps import ObsOperation

video = 'datasets/dataset-Trim.mp4'
outpath = './output/img_'
# 创建ObsClient实例
obsClient = ObsClient(
    access_key_id='3WNUTDBEXWYV0FYRZ4EO',
    secret_access_key='Xm4TLCUrDEI9ZzRPQLPc0Fpn1T5MY1Qw0z9sJTej',
    server='https://obs.cn-north-4.myhuaweicloud.com'
)

obsOps = ObsOperation(bucketname='its-ec')


def weights_downloads(bucketname, objectname, localfile):
    try:
        print(f'model weights file downloading from {bucketname}/{objectname}')
        resp = obsClient.getObject(bucketname, objectname, downloadPath=localfile)

        if resp.status < 300:
            print('upload successfully')
            print('requestId:', resp.requestId)
            print('url:', resp.body.url)
            print('localpath:', localfile)
        else:
            print('errorCode:', resp.errorCode)
            print('errorMessage:', resp.errorMessage)
    except:
        import traceback
        print(traceback.format_exc())




if __name__ == '__main__':
    localfile = './weights/best.pt'
    objectname = 'edge_outputs/model/best.pt'
    bucketname = 'its-ec'
    weights_downloads(bucketname=bucketname, objectname=objectname, localfile=localfile)
    # 关闭Obs
    obsClient.close()