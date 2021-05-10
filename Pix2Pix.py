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
train_dataset = tf.data.Dataset.from_tensor_slices(tr_urls)
train_dataset = train_dataset.map(load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(1)

test_dataset = tf.data.Dataset.from_tensor_slices(tr_urls)
test_dataset = test_dataset.map(load_test_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(1)

from tensorflow.keras.layers import *
from tensorflow.keras import *

def downsample(filters, apply_batchnorm = True):
  result = Sequential()

  initializer = tf.random_normal_initializer(0,0.02)

  #Capa convolucional
  result.add(Conv2D(filters, kernel_size = 4, strides = 2, padding = 'same', kernel_initializer = initializer,
                    use_bias=not apply_batchnorm))

  #Capa de BatchNorm
  result.add(BatchNormalization())

  #Capa de acticación
  result.add(LeakyReLU())

  return result

downsample(64)

def upsample(filters, apply_dropout = False):
  result = Sequential()

  initializer = tf.random_normal_initializer(0,0.02)

  #Capa convolucional
  result.add(Conv2DTranspose(filters, kernel_size = 4, strides = 2, padding = 'same', kernel_initializer = initializer,
                    use_bias=False))

  #Capa de BatchNorm
  result.add(BatchNormalization())

  if apply_dropout:
    #Capa dropout
    result.add(Dropout(0.5))

  #Capa de acticación
  result.add(ReLU())

  return result

downsample(64)

def generator():
  inputs = tf.keras.layers.Input(shape = [None, None, 3])
  downstack = [
               downsample(64, apply_batchnorm = False),
               downsample(128),
               downsample(256),
               downsample(512),
               downsample(512),
               downsample(512),
               downsample(512),
               downsample(512),
  ]

  upstack = [
             upsample(512, apply_dropout = True),
             upsample(512, apply_dropout = True),
             upsample(512, apply_dropout = True),
             upsample(512),
             upsample(256),
             upsample(128),
             upsample(64),
  ]

  initializer = tf.random_normal_initializer(0,0.02)

  #Generación de la última capa
  last = Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same', kernel_initializer= initializer,
                         activation = 'tanh')
  x = inputs
  s = []

  concat = Concatenate()

  #Se connecta la red convolucional
  for down in downstack:
    x = down(x)
    s.append(x)

  s = reversed(s[:-1])

  #Se conecta la red deconvolucional
  for up, sk in zip(upstack,s):
    x = up(x)
    x = concat([x,sk])

  last = last(x)
  return Model(inputs=inputs, outputs=last)

generator = generator()

def discriminator():
  ini = Input(shape=[None,None,3], name = 'input_img')
  gen = Input(shape=[None,None,3], name = 'gener_img')
  con = concatenate([ini,gen])

  initializer = tf.random_normal_initializer(0,0.02)
  down1 = downsample(64, apply_batchnorm = False)(con)
  down2 = downsample(128)(down1)
  down3 = downsample(256)(down2)
  down4 = downsample(512)(down3)

  last = Conv2D(filters=1, kernel_size=4, strides=1, padding='same', kernel_initializer= initializer)(down4)
  return tf.keras.Model(inputs=[ini,gen],outputs=last)
discriminator = discriminator()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
  #Diferencia entre los true por se real y el detectado por el discriminador
  real_loss = loss_object(tf.ones_like(disc_real_output),disc_real_output)
  #Diferencia entre los false por ser generado y el detectado por el discriminador
  generated_loss = loss_object(tf.zeros_like(disc_generated_output),disc_generated_output)

  total_disc_loss = real_loss + generated_loss
  return total_disc_loss
  
LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output),disc_generated_output)

  #mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target-gen_output))

  total_gen_loss = gan_loss + (LAMBDA*l1_loss)
  return total_gen_loss

import os
generator_optimazer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimazer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_prefix = os.path.join(CHECKPATH,'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimazer=generator_optimazer, discriminator_optimazer=discriminator_optimazer, generator=generator, discriminator=discriminator)
#checkpoint.restore(tf.train.latest_checkpoint(INPATH)).assert_consumed()

def generated_images(model, test_input,tar,save_filename=False, display_imgs=True):
  prediction = model(test_input, training=True)

  if save_filename:
    tf.keras.preprocessing.image.save_img(PATH + '/output/'+save_filename+'.jpg', prediction[0,...])

  plt.figure(figsize=(10,10))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image','Real Image','Generated Image']

  if display_imgs:
    for i in range(3):
      plt.subplot(1,3,i+1)
      plt.title(title[i])
      #Getting the pixel value between [0,1] to plot it
      plt.imshow(display_list[i]*0.5+0.5)
      plt.axis('off')
  plt.show()
@tf.function()
def train_step(input_image, target):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:
    output_image = generator(input_image, training=True)
    output_gen_discr = discriminator([output_image,input_image],training=True)
    output_trg_discr = discriminator([target,input_image],training=True)
    discr_loss = discriminator_loss(output_trg_discr, output_gen_discr)
    gen_loss = generator_loss(output_gen_discr, output_image, target)

    generator_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_grads = discr_tape.gradient(discr_loss, discriminator.trainable_variables)
    generator_optimazer.apply_gradients(zip(generator_grads,generator.trainable_variables))
    discriminator_optimazer.apply_gradients(zip(discriminator_grads, discriminator.trainable_variables))
from IPython.display import clear_output

def train(dataset, epochs):
  for epoch in range(epochs):
    imgi = 0
    for input_image, target in dataset:
      print('epoch'+str(epoch)+'- train: '+str(imgi)+'/'+str(len(tr_urls)))
      imgi+=1
      train_step(input_image, target)
      clear_output(wait=True)

    imgi = 0
    for inp, tar in test_dataset.take(5):
      generated_images(generator, inp, tar, str(imgi)+'_'+str(epoch), display_imgs=True)
      imgi += 1
    #Saving (checkpoint) the model every 20 epochs
    if (epoch+1)%50 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)
train(train_dataset, 100)