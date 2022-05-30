import os
import pickle
import h5py
import numpy as np
import base64
from extract_cnn_vgg16_keras import VGGNet
import matplotlib.pyplot as plt
import redis

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
  pool = redis.ConnectionPool(host='localhost', port=6379, db=1,decode_responses=True)
  pool1 = redis.ConnectionPool(host='localhost', port=6379, db=2, decode_responses=True)
  print("connected success.")
except:
  print("could not connect to redis.")
r = redis.Redis(connection_pool=pool)
r1 = redis.Redis(connection_pool=pool1)
'''
 Returns a list of filenames for all jpg images in a directory. 
'''


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg','.jpeg'))]

def get_package_list(path):
    return [os.path.join(path, f) for f in os.listdir(path)]
'''
 Extract features and index the images
'''
if __name__ == "__main__":
    database ='animals10'
    index = 'model/test_future.h5'
    package_list = get_package_list(database)

    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")

    feats = []
    names = []

    model = VGGNet()
    for j, package in enumerate(package_list):
        img_list=get_imlist(package)
        for i, img_path in enumerate(img_list):
            norm_feat = model.vgg_extract_feat(img_path)  # 修改此处改变提取特征的网络

            img_name = os.path.split(img_path)[1]
            feats.append(norm_feat)
            names.append(img_name)
            pickled_norm_feats=pickle.dumps(norm_feat)#  redis  存储向量进行包装
            # unpacked_object = pickle.loads(r.get('some_key'))   读取redis 后进行解包装
            # obj == unpacked_object
            with open(img_path,"rb") as f:
                base64_data=base64.b64encode(f.read())
                r.set(img_name,base64_data)
                r1.set(img_name,pickled_norm_feats)#
            print("extracting feature from image No. %d , %d_%d images in total" % ((i + 1), j,len(img_list)))

    feats = np.array(feats)
    # print(feats)
    # directory for storing extracted features
    # output = args["index"]
    output = index
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")

    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=feats)
    # h5f.create_dataset('dataset_2', data = names)
    h5f.create_dataset('dataset_2', data=np.string_(names))
    h5f.close()
