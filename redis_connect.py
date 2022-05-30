import redis
import base64
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mping
try:
  pool = redis.ConnectionPool(host='localhost', port=6379,db=1, decode_responses=True)
  pool1 = redis.ConnectionPool(host='localhost', port=6379,db=2, decode_responses=True)
  print("connected success.")
except:
  print("could not connect to redis.")
r = redis.Redis(connection_pool=pool)
r1 = redis.Redis(connection_pool=pool1)
img=r.get("2.jpg")
img=base64.b64decode(img)
with open("test_base64.jpg", 'wb') as f:
  f.write(img)
  f.close()
lena=mping.imread("test_base64.jpg")
lena.shape
plt.title("search output")
plt.imshow(lena)
plt.show()




