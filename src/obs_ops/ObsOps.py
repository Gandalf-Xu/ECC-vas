from obs import ObsClient

class ObsOperation():
    def __init__(self, bucketname):
        # 创建ObsClient实例
        self.obsClient = ObsClient(
            access_key_id='3WNUTDBEXWYV0FYRZ4EO',
            secret_access_key='Xm4TLCUrDEI9ZzRPQLPc0Fpn1T5MY1Qw0z9sJTej',
            server='https://obs.cn-north-4.myhuaweicloud.com'
        ),

        # self.obsClient = obsClient,
        self.bucketname = bucketname

    ## 图片上传
    # bucketname = 'its-ec'
    # localfile = 'C:\\Users\\MorningStar\\Desktop\\ITS-EC\\Edge01\\src\\test\\best.pt'
    # objectkey = 'test/test.pt'
    def data_upload(self, localfile, objectkey):
        try:
            from obs import PutObjectHeader
            print(f'frame uploading to {self.bucketname}/{objectkey}')
            resp = self.obsClient.putFile(self.bucketname, objectkey, localfile)

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

    ## 文件下载
    # localfile = 'C:\\Users\\MorningStar\\Desktop\\ITS-EC\\Edge01\\src\\test\\best.pt'
    # objectname = 'edge_outputs/model/best.pt'
    # bucketname = 'its-ec'
    def weights_downloads(self, objectname, localfile):
        try:
            print(f'model weights file downloading from {self.bucketname}/{objectname}')
            resp = self.obsClient.getObject(self.bucketname, objectname, downloadPath=localfile)

            if resp.status < 300:
                print('upload successfully')
                print('requestId:', resp.requestId)
                print('url:', resp.body.url)
                print('localpath:',localfile)
            else:
                print('errorCode:', resp.errorCode)
                print('errorMessage:', resp.errorMessage)
        except:
            import traceback
            print(traceback.format_exc())

    # 关闭Obs
    def obs_close(self):
        self.obsClient.close()




# 使用Obs




