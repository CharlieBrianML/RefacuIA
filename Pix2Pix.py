import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

#Ruta raiz
PATH = '/content/drive/My Drive/'

CHECKPATH = PATH + 'checkpoints'
INPATH = PATH + 'faces_blur'
OUTPATH = PATH + 'faces'

imgurls = !ls -1 "{INPATH}"

n = 7
train_n = round(n*0.80)

#Listado random
randurls = np.copy(imgurls)

np.random.seed(2)
np.random.shuffle(randurls)

#Partition train/test
tr_urls = randurls[:train_n]
ts_urls = randurls[train_n:n]

print(len(imgurls),len(tr_urls),len(ts_urls))

IMG_WIDTH = 256
IMG_HEIGHT = 256

#Reescalar imagenes
def resize(inimg, tgimg, width, height):
  inimg = tf.image.resize(inimg, [width, height])
  tgimg = tf.image.resize(tgimg, [width, height])

  return inimg, tgimg

#Normalizado de las imagenes
def normalize(inimg, tgimg):
  inimg = (inimg/127.5)-1
  tgimg = (tgimg/127.5)-1
  
  return inimg, tgimg

#Aumento de datos
def random_jitter(inimg,tgimg):
  inimg,tgimg = resize(inimg,tgimg,286,286)
  stacked_img = tf.stack([inimg,tgimg],axis=0)
  cropped_img = tf.image.random_crop(stacked_img,[2,IMG_WIDTH,IMG_HEIGHT,3])

  inimg, tgimg = cropped_img[0], cropped_img[1]

  if tf.random.uniform(()) > 0.5:
    inimg = tf.image.random_flip_left_right(inimg)
    tgimg = tf.image.random_flip_left_right(tgimg)

  return inimg, tgimg

def load_image(filename, argument = True):
  inimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(INPATH+'/'+filename)), tf.float32)[..., :3]
  tgimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(OUTPATH+'/'+filename)), tf.float32)[..., :3]

  inimg,tgimg = resize(inimg,tgimg,IMG_WIDTH,IMG_HEIGHT)

  if argument:
    inimg, tgimg = random_jitter(inimg, tgimg)

  inimg, tgimg = normalize(inimg, tgimg)

  return inimg, tgimg

def load_train_image(filename):
  return load_image(filename, True)
def load_test_image(filename):
  return load_image(filename, False)

plt.imshow(((load_train_image(randurls[0])[1])+1)/2)