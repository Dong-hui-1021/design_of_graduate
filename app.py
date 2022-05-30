from flask import Flask,render_template,request
import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import base64
import os,sys,random,string
from extract_cnn_vgg16_keras import VGGNet

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))
@app.route('/')
def hello_world():  # put application's code here
    return render_template("search.html")

@app.route('/img_research',methods=['POST','GET'])
def img_research():
    if request.method == 'GET':
        return render_template('fail.html')
    elif request.method == 'POST':
        if 'image' in request.files:
            image=request.files.get('image')
            # 生成随机字符串，防止图片名字重复
            ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 16))
            # 获取图片文件 name = upload
            img = request.files.get('image')
            print(img)
            # 定义一个图片存放的位置 存放在static下面
            path = basedir + "/static/searched_img/"
            # 图片名称 给图片重命名 为了图片名称的唯一性
            imgName = ran_str + '.jpg'
            # 图片path和名称组成图片的保存路径
            file_path = path + imgName
            # 保存图片
            img.save(file_path)
            # 这个是图片的访问路径，需返回前端（可有可无）
            query = 'static/searched_img/' + imgName

            index = 'models/vgg_featureCNN.h5'
            result = 'database'
            h5f = h5py.File(index, 'r')
            feats = h5f['dataset_1'][:]
            imgNames = h5f['dataset_2'][:]
            h5f.close()

            print("--------------------------------------------------")
            print("               searching starts")
            print("--------------------------------------------------")

            queryImg = mpimg.imread(query)
            # plt.title("Query Image")
            # plt.imshow(queryImg)
            # plt.show()

            # init VGGNet16 model
            model = VGGNet()

            # extract query image's feature, compute simlarity score and sort
            queryVec = model.vgg_extract_feat(query)  # 修改此处改变提取特征的网络

            print('--------------------------')

            print('--------------------------')
            scores = np.dot(queryVec, feats.T)
            # scores = np.dot(queryVec, feats.T)/(np.linalg.norm(queryVec)*np.linalg.norm(feats.T))
            rank_ID = np.argsort(scores)[::-1]
            rank_score = scores[rank_ID]
            # print (rank_ID)
            print(rank_score)

            # number of top retrieved images to show
            maxres = 10  # 检索出10张相似度最高的图片
            imlist = []
            for i, index in enumerate(rank_ID[0:maxres]):
                imlist.append(imgNames[index])
                # print(type(imgNames[index]))
                print("image names: " + str(imgNames[index]) + " scores: %f" % rank_score[i])
            print("top %d images in order are: " % maxres, imlist)
            # show top #maxres retrieved result one by one
            # for i, im in enumerate(imlist):
            #     image = mpimg.imread(result + "/" + str(im, 'utf-8'))
            #     plt.title("search output %d" % (i + 1))
            #     plt.imshow(image)
            #     plt.show()
            img_stream = ''
            img_stream_list = []

            for i,im in enumerate(imlist):
                ab_path=basedir+'/database/'+im.decode();
                with open(ab_path, 'rb') as img_f:
                    img_stream = base64.b64encode(img_f.read()).decode()
                    img_stream_list.append(img_stream)
            with open(file_path, 'rb') as img_f:
                img_stream = base64.b64encode(img_f.read()).decode()

            return render_template("show.html",imlist=img_stream_list,input_img=img_stream)
        return render_template('fail.html')


if __name__ == '__main__':
    app.run()
