import config
from PIL import Image
import numpy as np

def generateds(path, txt):
  x, y = [], []
  f = open(txt, 'r')
  contents = f.readlines()
  f.close()
  for c in contents:
    value = c.split()
    img_path = path + value[0]
    img = Image.open(img_path)
    img = np.array(img.convert('L'))
    img = img / 255.0
    x.append(img)
    y.append(value[1])
    print('Load: ' + c)
  x = np.array(x)
  y = np.array(y)
  y = y.astype(np.int64)
  return x, y

x_train, y_train = generateds(config.train_images_path, config.train_label_path)
x_test, y_test = generateds(config.test_images_path, config.test_label_path)

np.savez_compressed(config.ouput, x_train=x_train, y_train=y_train,x_test=x_test,y_test=y_test)