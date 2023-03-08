import check_for_errors
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model, Sequential # Подключаем модель типа Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.utils import to_categorical, plot_model # Полкючаем методы .to_categorical() и .plot_model()
sns.set_style('darkgrid') 
from tensorflow.keras import backend as K # Импортируем модуль backend keras'а
from tensorflow.keras.optimizers import Nadam, RMSprop, Adadelta,Adam # Импортируем оптимизатор Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, ReduceLROnPlateau
from tensorflow.keras.models import Model # Импортируем модели keras: Model
from tensorflow.keras.layers import Input, RepeatVector, Conv2DTranspose, concatenate, Activation, Embedding, Input, MaxPooling2D, Conv2D, BatchNormalization # Импортируем стандартные слои keras
import importlib.util, sys, gdown,os
import tensorflow as tf
from PIL import Image
import pandas as pd
import time
from IPython.display import clear_output
from tensorflow.keras.preprocessing import image
import termcolor
from termcolor import colored
from google.colab import files
import subprocess, os, warnings, time
from pandas.core.common import SettingWithCopyWarning
from subprocess import STDOUT, check_call
from IPython import display
import numpy as np
import requests
import random, pickle
import zipfile
from sklearn.model_selection import train_test_split
import ast
import json
from tabulate import tabulate
import getpass
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
import numpy as np
import warnings
from keras.utils.np_utils import to_categorical
warnings.filterwarnings('ignore')
import logging
tf.get_logger().setLevel(logging.ERROR)

@check_for_errors.decorator_check
def создать_слой(данные, **kwargs):
  args={}
  if 'входной_размер' in kwargs:
    args['input_shape'] = kwargs['входной_размер']  
  параметры = [данные]
  act = 'relu'
  if '-' in данные:
    параметры = данные.split('-')
  if параметры[0].upper() == 'ПОЛНОСВЯЗНЫЙ':    
    if len(параметры)>2:
      act = параметры[2]
    return Dense(int(параметры[1]), activation=act, **args)
  if параметры[0].upper() == 'ПОВТОР':
    return RepeatVector(int(параметры[1]))
  if параметры[0].upper() == 'ЭМБЕДДИНГ':
    return Embedding(int(параметры[2]), int(параметры[1]), input_length=int(параметры[3]))
  elif параметры[0].upper() == 'СВЕРТОЧНЫЙ2D':
    if len(параметры)<5:
      act = 'relu'
      pad='same'
    else:
      act = параметры[4]
      pad = параметры[3]
    if any(i in '()' for i in параметры[2]):
      return Conv2D(int(параметры[1]), (int(параметры[2][1]),int(параметры[2][3])), padding=pad,activation=act, **args)
    else:
      return Conv2D(int(параметры[1]), int(параметры[2]), padding=pad,activation=act, **args)
  elif параметры[0].upper() == 'СВЕРТОЧНЫЙ1D':
    if len(параметры)>4:
      act = параметры[4]
      pad = параметры[3]
    else:
      act = 'relu'
      pad = 'same'
    return Conv1D(int(параметры[1]), int(параметры[2]), padding=pad,activation=act, **args)

  elif параметры[0].upper() == 'ВЫРАВНИВАЮЩИЙ':
    if 'input_shape' in args:
      return Flatten(input_shape=args['input_shape'])
    else:
      return Flatten() 
  elif параметры[0].upper() == 'НОРМАЛИЗАЦИЯ':
    return BatchNormalization()
  elif параметры[0].upper() == 'НОРМАЛИЗАЦИЯ_01':
    return Lambda(normalize_01)
  elif параметры[0].upper() == 'НОРМАЛИЗАЦИЯ_11':
    return Lambda(normalize_m11)
  elif параметры[0].upper() == 'ДЕНОРМАЛИЗАЦИЯ':
    return Lambda(denormalize_m11)
  elif параметры[0].upper() == 'АКТИВАЦИЯ':
    return Activation(параметры[1])
  elif параметры[0].upper() == 'ЛСТМ':    
    return LSTM(int(параметры[1]), return_sequences=параметры[2]=='Последовательность', **args)
  
  elif параметры[0].upper() == 'МАКСПУЛЛИНГ2D':
    if any(i in '()' for i in параметры[1]):
      return MaxPooling2D((int(параметры[1][1]),int(параметры[1][3])))
    else:
      return MaxPooling2D(int(параметры[1]))  
  
  elif параметры[0].upper() == 'МАКСПУЛЛИНГ1D':
    if any(i in '()' for i in параметры[1]):
      return MaxPooling1D(int(параметры[1]))
    else:
      return MaxPooling1D(int(параметры[1]))
      
  elif параметры[0].upper() == 'ДРОПАУТ':
    return Dropout(float(параметры[1]))
  elif параметры[0].upper() == 'PRELU':
    return PReLU(shared_axes=[1, 2])
  elif параметры[0].upper() == 'LEAKYRELU':
    return LeakyReLU(alpha=float(параметры[1]))
  else:
    assert False, f'Невозможно добавить указанный слой: \'{параметры[0]}\'.\
     Возможно вы имели ввиду \'{check_for_errors.check_word(параметры[0], "layer")}\'.'

def upsample(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)

def normalize_01(x):
    return x / 255.0

# Нормализует RGB изображения к промежутку [-1, 1]
def normalize_m11(x):
    return x / 127.5 - 1

# Обратная нормализация
def denormalize_m11(x):
    return (x + 1) * 127.5

#Метрика
def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def создать_сеть_для_классификации(слои, вход, параметры_модели):
  layers = слои.split()
  model = Sequential()
  layer = создать_слой(layers[0], входной_размер=вход)
  model.add(layer)   
  for i in range(1, len(layers)):
    layer = создать_слой(layers[i])
    model.add(layer)
  #print('Создана модель нейронной сети!')
  параметры = параметры_модели.split()  
  loss = параметры[0].split(':')[1]
  opt = параметры[1].split(':')[1]
  metrica = ''
  if (len(параметры)>2):
    metrica = параметры[2].split(':')[1]
  if metrica=='':
    model.compile(loss=loss, optimizer = opt)
  else:
    model.compile(loss=loss, optimizer = opt, metrics=[metrica])
  return model

def создать_сеть_для_сегментации(слои, вход, параметры_модели):
  def точность(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
  layers = слои.split()
  model = Sequential()
  layer = создать_слой(layers[0], входной_размер=вход)
  model.add(layer) 
  for i in range(1, len(layers)):
    layer = создать_слой(layers[i])
    model.add(layer)
  #print('Создана модель нейронной сети!')
  параметры = параметры_модели.split()  
  loss = параметры[0].split(':')[1]
  opt = параметры[1].split(':')[1]
  metrica = ''  
  if (len(параметры)>2):
    if параметры[2].split(':')[1] == 'dice_coef':
      metrica = точность
  model.compile(loss=loss, optimizer = opt, metrics =[metrica])
  return model

def создать_дискриминатор_повышения_размерности(блок_дискриминатора,количество_блоков, финальный_блок):
  x_in = Input(shape=(96, 96, 3))
  x = Lambda(normalize_m11)(x_in)
  blocks = блок_дискриминатора.split()
  for i in range(количество_блоков):
    for b in blocks:
      x = создать_слой(b) (x)      
  x = Flatten() (x)
  blocks = финальный_блок.split()
  for b in blocks:
    x = создать_слой(b) (x)  
  return Model(x_in, x)

def создать_генератор_повышения_размерности(стартовый_блок, основной_блок, финальный_блок):
  x_in = Input(shape=(None, None, 3))
  layers = стартовый_блок.split()
  x = создать_слой(layers[0]) (x_in)
  for i in range(1, len(layers)):
    x = создать_слой(layers[i]) (x)
  layers = основной_блок.split()
  x2 = создать_слой(layers[0]) (x)
  for i in range(1, len(layers)-1):
    x2 = создать_слой(layers[i]) (x2)
  x2 = Add()([x2, x2]) 
  x2 = создать_слой(layers[-1]) (x2)
  x = Add()([x, x2])
  x = upsample(x, 64 * 4)
  x = upsample(x, 64 * 4)
  layers = финальный_блок.split()
  x = создать_слой(layers[0]) (x)
  for i in range(1, len(layers)):
    x = создать_слой(layers[i]) (x)
  return Model(x_in, x)

def создать_UNET(**kwargs):
  параметры_модели = 'Ошибка:categorical_crossentropy\
      оптимизатор:adam\
      метрика:dice_coef'
  def isConv2D(layer):
    return layer.split('-')[0]=='Сверточный2D'
      
  def точность(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

  if 'количество_выходных_классов' in kwargs:
    n_classes = kwargs['количество_выходных_классов']
  else:
    n_classes = 2

  #Блок вниз 
  block1 = []
  for i in range(len(kwargs['блоки_вниз'])):
    temp_block = kwargs['блоки_вниз'][i]
    split_block = temp_block.split()
    block1.append(split_block)
  
  #блок внизу
  s = kwargs['блок_внизу'].split('\n')
  down_blocks = ''
  for i in range(len(s)):
    temp_itm = s[i].split()
    for j in range(len(temp_itm)):
      if j + 1 == len(temp_itm) and i + 1 == len(s):
        down_blocks += str(temp_itm[j])
      else:
        down_blocks += str(temp_itm[j] + ' ')
  down_blocks = down_blocks.split() 
  
  #Блок вверх 
  block2 = []
  for i in range(len(kwargs['блоки_вверх'])):
    temp_block = kwargs['блоки_вверх'][i]
    split_block = temp_block.split()
    block2.append(split_block)

  img_input = Input(kwargs['входной_размер']) #входной размер
  nBlock = len(kwargs['блоки_вверх']) + 1 #количество блоков
  текущий_блок = 0
  b_o = []
  #Down
  #input layers
  layer = block1[0][0]
  if isConv2D(layer):
    layer = layer
  x = создать_слой(layer, входной_размер=kwargs['входной_размер'])(img_input)

  for i in range(len(block1)):
    for j in range(len(block1[i])):
      layer = block1[i][j]
      if isConv2D(layer):
        layer = layer
      x = создать_слой(layer)(x)
    b_o.append(x)
    x = MaxPooling2D()(b_o[-1])

  #Down block
  for i in range(nBlock-1):
    for j in range(len(down_blocks)):
      layer = down_blocks[j]
      if isConv2D(layer):
        layer = layer
      x = создать_слой(layer)(x)
    b_o.append(x)
    x = MaxPooling2D()(b_o[-1])
  x = b_o[i+1]   

  #Up
  for i in range(len(block2)):
    x = Conv2DTranspose(2**(2*nBlock-i), (2, 2), strides=(2, 2), padding='same')(x)    # Добавляем слой Conv2DTranspose с 256 нейронами
    for j in range(len(block2[i])):
      layer = block2[i][j]
      if layer=='Объединение':
        x = concatenate([x, b_o[nBlock-i-2]])
      else:
        if isConv2D(layer):
          layer = layer    
        x = создать_слой(layer) (x)
  x = Conv2D(n_classes, (3, 3), activation='softmax', padding='same')(x)  # Добавляем Conv2D-Слой с softmax-активацией на num_classes-нейронов

  мод = Model(img_input, x)
  параметры = параметры_модели.split()  
  loss = параметры[0].split(':')[1]
  opt = параметры[1].split(':')[1]
  metrica = ''  
  if (len(параметры)>2):
    if параметры[2].split(':')[1] == 'dice_coef':
      metrica = точность
  
  #print('Создана модель нейронной сети!')
  мод.compile(loss=loss, optimizer = opt, metrics =[metrica])
  return мод
  
def создать_PSP(**kwargs):
  параметры_модели = 'Ошибка:categorical_crossentropy\
    оптимизатор:adam\
    метрика:dice_coef'
  def точность(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
    
  img_input = Input(kwargs['входной_размер'])
  nBlock = kwargs['количество_блоков']
  if 'количество_выходных_классов' in kwargs:
    n_classes = kwargs['количество_выходных_классов']
  else:
    n_classes = 2
  start_block = kwargs['стартовый_блок']
  layers = start_block.split()
  layer = создать_слой(layers[0])  
  x = layer(img_input)
  for i in range(1, len(layers)):
    layer = создать_слой(layers[i])
    x = layer(x)
  
  x_mp = []
  conv_size = 32
  block_PSP = kwargs['блок_PSP']
  layers = block_PSP.split()  
  for i in range(nBlock):
    l = MaxPooling2D(2**(i+1))(x)
    for k in range(len(layers)):
      layer = создать_слой(layers[k])      
      l = layer(l)
    l = Conv2DTranspose(32, (2**(i+1), 2**(i+1)), strides=(2**(i+1), 2**(i+1)), activation='relu')(l)
    x_mp.append(l)

  fin = concatenate([img_input]+ x_mp)

  final_block = kwargs['финальный_блок']+'-same-softmax'
  layers = final_block.split()
  for i in range(len(layers)):
    layer = создать_слой(layers[i])
    fin = layer(fin)

  параметры = параметры_модели.split()  
  loss = параметры[0].split(':')[1]
  opt = параметры[1].split(':')[1]
  metrica = ''  
  if (len(параметры)>2):
    if параметры[2].split(':')[1] == 'dice_coef':
      metrica = точность
  
  мод = Model(img_input, fin)
  #print('Создана модель нейронной сети!')
  мод.compile(loss='categorical_crossentropy', optimizer=Adam(lr=3e-4), metrics =[точность])
  return мод
  
def создать_составную_сеть_квартиры(данные, *нейронки):    
    input1 = Input(данные[0].shape[1],)
    input2 = Input(данные[1].shape[1],)
    
    layers = нейронки[0].split()
    x1 = создать_слой(layers[0]) (input1)
    for i in range(1, len(layers)):
        layer = создать_слой(layers[i])
        assert layer!=0, 'Невозможно добавить указанный слой: '+layer
        x1 = создать_слой(layers[i]) (x1)

    layers = нейронки[1].split()
    x2 = создать_слой(layers[0]) (input2)
    for i in range(1, len(layers)):
        layer = создать_слой(layers[i])
        assert layer!=0, 'Невозможно добавить указанный слой: '+layer
        x2 = создать_слой(layers[i]) (x2)
           
    x = concatenate([x1, x2])
    layers = нейронки[2].split()
    x3 = создать_слой(layers[0]) (x)
    for i in range(1, len(layers)):
        layer = создать_слой(layers[i])
        assert layer!='0', 'Невозможно добавить указанный слой: '+layer
        x3 = создать_слой(layers[i]) (x3)    
    model = Model([input1, input2], x3)
    model.compile(loss="mae", optimizer=Nadam(lr=1e-3), metrics=["mae"])
    return model

def создать_составную_сеть(данные, метки, *нейронки):
    img_input1 = Input(данные[0].shape[1],)
    img_input2 = Input(данные[1].shape[1],)
    img_input3 = Input(данные[2].shape[1],)
    
    layers = нейронки[0].split()
    x1 = создать_слой(layers[0]) (img_input1)
    for i in range(1, len(layers)):
        layer = создать_слой(layers[i])
        assert layer!=0, 'Невозможно добавить указанный слой: '+layer
        x1 = создать_слой(layers[i]) (x1)

    layers = нейронки[1].split()
    x2 = создать_слой(layers[0]) (img_input2)
    for i in range(1, len(layers)):
        layer = создать_слой(layers[i])
        assert layer!=0, 'Невозможно добавить указанный слой: '+layer
        x2 = создать_слой(layers[i]) (x2)
    
    layers = нейронки[2].split()
    x3 = создать_слой(layers[0]) (img_input3)
    for i in range(1, len(layers)):
        layer = создать_слой(layers[i])
        assert layer!=0, 'Невозможно добавить указанный слой: '+layer
        x3 = создать_слой(layers[i]) (x3)
        
    x = concatenate([x1, x2, x3])
    x = Dense(100, activation="relu")(x)
    x = Dense(метки.shape[1], activation="softmax")(x)
    
    model = Model([img_input1, img_input2, img_input3], x)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=5e-5), metrics=["accuracy"])
    return model

def создать_составную_сеть_писатели(данные, *нейронки):
    img_input1 = Input(данные[0].shape[1],)
    img_input2 = Input(данные[1].shape[1],)
    img_input3 = Input(данные[2].shape[1],)
    
    layers = нейронки[0].split()
    x1 = создать_слой(layers[0]) (img_input1)
    for i in range(1, len(layers)):
        layer = создать_слой(layers[i])
        assert layer!=0, 'Невозможно добавить указанный слой: '+layer
        x1 = создать_слой(layers[i]) (x1)

    layers = нейронки[1].split()
    x2 = создать_слой(layers[0]) (img_input2)
    for i in range(1, len(layers)):
        layer = создать_слой(layers[i])
        assert layer!=0, 'Невозможно добавить указанный слой: '+layer
        x2 = создать_слой(layers[i]) (x2)
    
    layers = нейронки[2].split()
    x3 = создать_слой(layers[0]) (img_input3)
    for i in range(1, len(layers)):
        layer = создать_слой(layers[i])
        assert layer!=0, 'Невозможно добавить указанный слой: '+layer
        x3 = создать_слой(layers[i]) (x3)
        
    x = concatenate([x1, x2, x3])
    x = Dense(1024, activation="relu")(x)
    x = Dense(6, activation="softmax")(x)
    
    model = Model([img_input1, img_input2, img_input3], x)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=5e-5), metrics=["accuracy"])
    return model

def создать_сеть_чат_бот(размер_словаря, энкодер, декодер):
  encoderInputs = Input(shape=(None , )) # размеры на входе сетки (здесь будет encoderForInput)
  
  layers = энкодер.split()
  if '-' in layers[0]:
    буква, параметр = layers[0].split('-')
    x = Embedding(размер_словаря, int(параметр), mask_zero=True) (encoderInputs)      
  for i in range(1, len(layers)-1):
    layer = создать_слой(layers[i])
    assert layer!=0, 'Невозможно добавить указанный слой: '+layer
    x = создать_слой(layers[i]) (x)
  if '-' in layers[-1]:
    буква, параметр = layers[-1].split('-')
  encoderOutputs, state_h , state_c = LSTM(int(параметр), return_state=True)(x)
  encoderStates = [state_h, state_c]
    
  decoderInputs = Input(shape=(None, )) # размеры на входе сетки (здесь будет decoderForInput)
  layers = декодер.split()
  if '-' in layers[0]:
    буква, параметр = layers[0].split('-')
    x = Embedding(размер_словаря, int(параметр), mask_zero=True) (decoderInputs) 
  for i in range(1, len(layers)-1):
    layer = создать_слой(layers[i])
    assert layer!=0, 'Невозможно добавить указанный слой: '+layer
    x = создать_слой(layers[i]) (x)
  if '-' in layers[-1]:
    буква, параметр = layers[-1].split('-')
  decoderLSTM = LSTM(int(параметр), return_state=True, return_sequences=True)
  decoderOutputs , _ , _ = decoderLSTM (x, initial_state=encoderStates)
  # И от LSTM'а сигнал decoderOutputs пропускаем через полносвязный слой с софтмаксом на выходе
  decoderDense = Dense(размер_словаря, activation='softmax') 
  output = decoderDense (decoderOutputs)
  ######################
  # Собираем тренировочную модель нейросети
  ######################
  model = Model([encoderInputs, decoderInputs], output)
  model.compile(optimizer=RMSprop(), loss='sparse_categorical_crossentropy', metrics=["accuracy"])
  return model        
    
def схема_модели(модель):
  print('Схема модели:')
  return plot_model(модель, dpi=60) # Выводим схему модели

def создать_сеть(слои, входной_размер, параметры_модели, задача):
  layers = слои.split()
  задача = задача.lower()
  if задача == 'классификация изображений' or задача == 'классификация вакансий':
    параметры_модели = 'Ошибка:sparse_categorical_crossentropy\
            оптимизатор:adam\
            метрика:accuracy'
    return создать_сеть_для_классификации(слои+'-softmax', входной_размер, параметры_модели)
  if задача == 'временной ряд':
    # Указываем параметры создаваемой модели
    параметры_модели = 'Ошибка:mse\
    оптимизатор:adam\
    метрика:accuracy'
    return создать_сеть_для_классификации(слои+'-linear', входной_размер, параметры_модели)
  if задача == 'аудио':
    параметры_модели = 'Ошибка:categorical_crossentropy\
    оптимизатор:adam\
    метрика:accuracy'
    return создать_сеть_для_классификации(слои+'-softmax', входной_размер, параметры_модели)
  if задача == 'сегментация изображений':
    параметры_модели = 'Ошибка:categorical_crossentropy\
      оптимизатор:adam\
      метрика:dice_coef' 
    return создать_сеть_для_сегментации(слои+'-same-softmax', входной_размер, параметры_модели)
  if задача == 'сегментация текста':
    параметры_модели = 'Ошибка:categorical_crossentropy\
      оптимизатор:adam\
      метрика:dice_coef' 
    return создать_сеть_для_сегментации(слои+'-same-sigmoid', входной_размер, параметры_модели)
  assert False, f'Данная задача \'{задача}\' не распознана. Возможно вы имели ввиду \'{check_for_errors.check_word(задача,"task")}\''

def обучение_модели_квартиры(модель, x_train, y_train, x_test=None, y_test=None, batch_size=None, epochs=None, коэф_разделения = 0.2):
  cur_time = time.time()
  global loss, val_loss, history, result, best_result, idx_best
  result = ''
  idx_best = 0
  best_result = 5.000
  filepath="model.h5"
  model_checkpoint_callback = ModelCheckpoint(
    filepath=filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

  def on_epoch_end(epoch, log):
    global cur_time, loss, val_loss, result, best_result, idx_best, count_len
    yScaler = pickle.load(open('инструменты/квартиры/yscaler.pkl','rb'))
    pred = модель.predict(x_test) #Полуаем выход сети на проверочно выборке
    predUnscaled = yScaler.inverse_transform(pred).flatten() #Делаем обратное нормирование выхода к изначальным величинам цен квартир
    yTrainUnscaled = yScaler.inverse_transform(y_test).flatten() #Делаем такое же обратное нормирование yTrain к базовым ценам
    delta = predUnscaled - yTrainUnscaled #Считаем разность предсказания и правильных цен
    absDelta = abs(delta) #Берём модуль отклонения

    pred2 = модель.predict(x_train) #Полуаем выход сети на проверочно выборке
    predUnscaled2 = yScaler.inverse_transform(pred2).flatten() #Делаем обратное нормирование выхода к изначальным величинам цен квартир
    yTrainUnscaled2 = yScaler.inverse_transform(y_train).flatten() #Делаем такое же обратное нормирование yTrain к базовым ценам
    delta2 = predUnscaled2 - yTrainUnscaled2 #Считаем разность предсказания и правильных цен
    absDelta2 = abs(delta2) #Берём модуль отклонения
    loss.append(sum(absDelta2) / (1e+6 * len(absDelta2)))
    val_loss.append(sum(absDelta) / (1e+6 * len(absDelta)))
    
    p1 = 'Эпоха №' + str(epoch+1)
    p2 = p1 + ' '* (10 - len(p1)) + 'Время обучения: ' + str(round(time.time()-cur_time,2)) +'c'
    p3 = p2 + ' '* (33 - len(p2)) + 'Ошибка на обучающей выборке: ' + str(round(sum(absDelta2) / (1e+6 * len(absDelta2)), 3))+'млн'
    p4 = p3 + ' '* (77 - len(p3)) + 'Ошибка на проверочной выборке: ' + str(round(sum(absDelta) / (1e+6 * len(absDelta)), 3))+'млн'
    result += p4 + '\n' 
    
    a = round(sum(absDelta) / (1e+6 * len(absDelta)), 3)
    if a < best_result:
        best_result = a
        idx_best = epoch
    print(p4)
  
    # Коллбэки

  def on_train_begin(log):
    global cur_time, loss, val_loss
    loss=[]
    val_loss = []

  def on_epoch_begin(epoch, log):
    global cur_time
    cur_time = time.time()

  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-6)

  myCB = LambdaCallback(on_train_begin=on_train_begin, on_epoch_end = on_epoch_end, on_epoch_begin=on_epoch_begin)
  myCB23 = LambdaCallback(on_epoch_end = on_epoch_end, on_epoch_begin=on_epoch_begin)
  модель.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (x_test, y_test), callbacks=[model_checkpoint_callback, myCB, reduce_lr], verbose = 0)
  модель.load_weights('model.h5')
  модель.save('model_s.h5')
#   модель.compile(optimizer=Nadam(lr=1e-4), loss='mae')
#   модель.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (x_test, y_test), callbacks=[model_checkpoint_callback, myCB23], verbose = 0)
#   модель.load_weights('model.h5')
#   модель.compile(optimizer=Nadam(lr=1e-5), loss='mae')
#   модель.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (x_test, y_test), callbacks=[model_checkpoint_callback, myCB23], verbose = 0)
#   модель.load_weights('model.h5')
#   модель.save('model_s.h5')

  clear_output(wait=True)
  result = result.split('\n')
  
  idx = 0
  for i in range(len(result)):
    if str(best_result) in result[i]:
      idx = i

  for i in range(len(result)):
    s = result[i]
    if i == idx:
      s = colored(result[i], color='white', on_color='on_green')
    print(s)
      
  plt.figure(figsize=(12, 6)) # Создаем полотно для визуализации  
  plt.plot(loss, label ='Обучающая выборка') # Визуализируем график ошибки на обучающей выборке
  plt.plot(val_loss, label ='Проверочная выборка') # Визуализируем график ошибки на проверочной выборке
  plt.legend() # Выводим подписи на графике
  plt.title('График ошибки обучения') # Выводим название графика
  plt.show()
  
def обучение_модели_трафик(мод, ген1, ген2, количество_эпох=None):
  global result, idx_best, best_result, history
  result = ''
  idx_best = 0
  best_result = 1000000
  filepath="model.h5"
  model_checkpoint_callback = ModelCheckpoint(
    filepath=filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
    
  cur_time = time.time()

  def on_epoch_end(epoch, log):
    k = list(log.keys())
    global cur_time, result, idx_best, best_result
    p1 = 'Эпоха №' + str(epoch+1)
    p2 = p1 + ' '* (10 - len(p1)) + 'Время обучения: ' + str(round(time.time()-cur_time,2)) +'c'
    p3 = p2 + ' '* (33 - len(p2)) + 'Ошибка на обучающей выборке: ' + str(round(log[k[0]],5))
    p4 = p3 + ' '* (77 - len(p3)) + 'Ошибка на проверочной выборке: ' + str(round(log[k[2]],5))
    result += p4 + '\n'
    if log[k[2]] < best_result:
        best_result = log[k[2]]
        idx_best = epoch
    print(p4)
    cur_time = time.time()

  def on_epoch_begin(epoch, log):
    global cur_time
    cur_time = time.time()

  myCB = LambdaCallback(on_epoch_end = on_epoch_end, on_epoch_begin=on_epoch_begin)

  history = мод.fit_generator(ген1, epochs=количество_эпох, verbose=0, validation_data=ген2, callbacks=[model_checkpoint_callback, myCB])
  clear_output(wait=True)
  result = result.split('\n')
  for i in range(len(result)):
    s = result[i]
    if i == idx_best:
      s = colored(result[i], color='white', on_color='on_green')
    print(s)
  plt.plot(history.history['loss'], label='Ошибка на обучающем наборе')
  plt.plot(history.history['val_loss'], label='Ошибка на проверочном наборе')
  plt.ylabel('Средняя ошибка')
  plt.legend()

def обучение_модели(модель, x_train, y_train, x_test=[], y_test=[], 
                batch_size=None, epochs=None, коэф_разделения = 0.2, **kwargs):
  global result, idx_best, stage_history, best_result, history, epoch, stage_result, stage_best_result, best_result_on_train_arr
  cur_time = time.time()
  if 'количество_запусков' in kwargs:
    cur_time = time.time()
    global stage_result
    stage_result = ''
    stage_best_result = []
    best_result_on_train_arr = []
    stage_history = []
    for start in range(kwargs['количество_запусков']):
      result = ''
      idx_best = 0
      best_result = 0
      best_result_on_train = 0
      if batch_size == None:
        batch_size = 16
      if epochs == None:
        epochs = 10
      filepath="model.h5"

      model_checkpoint_callback = ModelCheckpoint(
        filepath=filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
        
      cur_time = time.time()
      def on_epoch_end(epoch, log):
        global cur_time, result, best_result, idx_best, stage_result, текущая_эпоха, stage_best_result, best_result_on_train, best_result_on_train_arr
        текущая_эпоха = epoch
        k = list(log.keys()) 
        # p1 = 'Запуск №' + str(start+1)
        # p3 = p1 + ' ' * (10 - len(p1)) + ' Точность на обучающей выборке: ' + str(round(log[k[1]]*100,2))+'%'
        
        p1 = 'Запуск №' + str(start+1)
        p2 = p1 + ' '* (12 - len(p1)) + 'Время обучения: ' + str(round((time.time()-cur_time) * epochs,2)) +'c'
        p3 = p2 + ' '* (34 - len(p2)) + 'Точность на обучающей выборке: ' + str(round(log[k[1]]*100,2))+'%'
        if len(k)>2:
            p4 = p3 + ' '* (77 - len(p3)) + 'Точность на проверочной выборке: ' + str(round(log[k[3]]*100,2))+'%'
            result += p4 + '\n'
            if log[k[3]]*100 >= best_result:
              best_result = log[k[3]]*100
              best_result_on_train = log[k[1]]*100
              idx_best = epoch
        else:
            result += p3 + '\n'
            if log[k[1]]*100 >= best_result:
              best_result = log[k[1]]*100
              best_result_on_train = log[k[1]]*100
              idx_best = epoch 

        if (текущая_эпоха+1) == epochs:
          result = result.split('\n')
          stage_best_result.append(best_result)
          best_result_on_train_arr.append(best_result_on_train)
          for i in range(len(result)):
            if i == idx_best:
              best_in_epoc = result[i]
              stage_result += best_in_epoc + '\n'
          print(best_in_epoc)
        cur_time = time.time()

      def on_epoch_begin(epoch, log):
        global cur_time
        cur_time = time.time()

      myCB = LambdaCallback(on_epoch_end = on_epoch_end, on_epoch_begin=on_epoch_begin)
      
      if len(x_test)==0:
        model_checkpoint_callback = ModelCheckpoint(
            filepath=filepath,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True)
        history = модель.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[model_checkpoint_callback, myCB], verbose = 0)
        stage_history.append(history)
      else:
        history = модель.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (x_test, y_test), callbacks=[model_checkpoint_callback, myCB], verbose = 0)
        stage_history.append(history)
      модель.load_weights('model.h5')
      модель.save('model_s.h5')

    clear_output(wait=True)
    stage_result = stage_result.split('\n')
    stage_best_result = np.asarray(stage_best_result)
    best_result_on_train_arr = np.asarray(best_result_on_train_arr)
    idx_best_stage = np.argmax(stage_best_result)

    for i in range(len(stage_result)):
      s = stage_result[i]
      if i == idx_best_stage:
        s = colored(stage_result[i], color='white', on_color='on_green')
      print(s)
    print('Средняя точность на обучающей выборке:   ' , str(round(np.mean(best_result_on_train_arr), 2)) +  '%', '\n' 
          'Средняя точность на проверочной выборке: ' , str(round(np.mean(stage_best_result), 2)) + '%')

    
    plt.figure(figsize=(12,6)) # Создаем полотно для визуализации
    keys = list(stage_history[idx_best_stage].history.keys())
    plt.plot(stage_history[idx_best_stage].history[keys[1]], label ='Обучающая выборка') # Визуализируем график точности на обучающей выборке
    if len(keys)>2:
      plt.plot(stage_history[idx_best_stage].history['val_'+keys[1]], label ='Проверочная выборка') # Визуализируем график точности на проверочной выборке
    plt.legend() # Выводим подписи на графике
    plt.title('График точности обучения, на лучшем запуске') # Выводим название графика
    plt.show()
    return history


  else:
    result = ''
    idx_best = 0
    best_result = 0
    if batch_size == None:
      batch_size = 16
    if epochs == None:
      epochs = 10
    filepath="model.h5"
    model_checkpoint_callback = ModelCheckpoint(
      filepath=filepath,
      save_weights_only=True,
      monitor='val_loss',
      mode='min',
      save_best_only=True)
      
    cur_time = time.time()
    def on_epoch_end(epoch, log):
      k = list(log.keys())
      global cur_time, result, idx_best, best_result   
      p1 = 'Эпоха №' + str(epoch+1)
      p2 = p1 + ' '* (12 - len(p1)) + 'Время обучения: ' + str(round(time.time()-cur_time,2)) +'c'
      p3 = p2 + ' '* (36 - len(p2)) + 'Точность на обучающей выборке: ' + str(round(log[k[1]]*100,2))+'%'
      if len(k)>2:
          p4 = p3 + ' '* (77 - len(p3)) + 'Точность на проверочной выборке: ' + str(round(log[k[3]]*100,2))+'%'
          result += p4 + '\n'
          if log[k[3]]*100 >= best_result:
            best_result = log[k[3]]*100
            idx_best = epoch
          print(p4)
      else:
          result += p3 + '\n'
          if log[k[1]]*100 >= best_result:
            best_result = log[k[1]]*100
            idx_best = epoch
          print(p3)    
      cur_time = time.time()   

    def on_epoch_begin(epoch, log):
      global cur_time
      cur_time = time.time()

    myCB = LambdaCallback(on_epoch_end = on_epoch_end, on_epoch_begin=on_epoch_begin)
    
    if len(x_test)==0:
      model_checkpoint_callback = ModelCheckpoint(
          filepath=filepath,
          save_weights_only=True,
          monitor='loss',
          mode='min',
          save_best_only=True)
      history = модель.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[model_checkpoint_callback, myCB], verbose = 0)
    else:
      history = модель.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (x_test, y_test), callbacks=[model_checkpoint_callback, myCB], verbose = 0)
    
    модель.load_weights('model.h5')
    модель.save('model_s.h5')
    clear_output(wait=True)
    result = result.split('\n')
    for i in range(len(result)):
      s = result[i]
      if i == idx_best:
        s = colored(result[i], color='white', on_color='on_green')
      print(s)
    plt.figure(figsize=(12,6)) # Создаем полотно для визуализации
    keys = list(history.history.keys())
    plt.plot(history.history[keys[1]], label ='Обучающая выборка') # Визуализируем график точности на обучающей выборке
    if len(keys)>2:
      plt.plot(history.history['val_'+keys[1]], label ='Проверочная выборка') # Визуализируем график точности на проверочной выборке
    plt.legend() # Выводим подписи на графике
    plt.title('График точности обучения') # Выводим название графика
    plt.show()
    return history

def создать_выборки_вакансии():
  global idx_val
  dir = 'вакансии'
  """
  Parameters:
  dir (str)              Путь к папке с датасетом

  Return:
  x_train, y_train, x_val, y_val
  """

  data = np.load(dir + '/X_train.npy')
  labels = np.load(dir + '/Y_train.npy')

  x_train, x_val, y_train, y_val = data[:500], data[300:], labels[:500], labels[300:]
  idx_val = [i for i in range(data[300:500].shape[0])]
  
  if not os.path.exists('test_sets'):
      os.mkdir('test_sets')
  np.save(f'test_sets/вакансии.npy', x_val)
  np.save(f'test_sets/вакансии_метки.npy', y_val)

  return (x_train, y_train), (x_val, y_val)

def тест_модели_вакансии(нейронка, x_val, y_val):
    dir='вакансии'
    num = 3
    data = pd.read_csv(dir + '/data.csv')
    data = data.iloc[300:500]
    data.reset_index(inplace=True, drop=True)

    columns=['Пол', 'Возраст', 'Город', 'Готовность к переезду',
             'Готовность к командировкам', 'Гражданство', 'Разрешение на работу',
             'Знания языков', 'Образование', 'Дополнительное образование',
             'Зарплата', 'Время в пути до работы', 'Занятость', 
             'График', 'Опыт работы (мес)', 'Обязанности на пред.работе']

    idx = np.random.randint(0, len(idx_val), num)

    for i in idx:
        pred = нейронка.predict(np.expand_dims(x_val[i], axis=0))

        print('Тестовое резюме:')
        print()

        for column in columns:
            info = str(data[column][i])
            info = info[:600]
            if info == 'no_data':
                info = 'Данные не указаны'
            step = 100
            for j in range(0, len(info), step):
                if j == 0:
                    print('%-28s' % (column + ':'), info[j:j+step])
                else:
                    print('%-28s' % (''), info[j:j+step])
        print()
        print(f'Модель уверена, что кандидат подходит в УИИ на: {round(100 - 100 * pred[0,1], 2)}%')
        print('---------------------------------------------------------------')
        result = data['Этап сделки'][i]
        print('%-28s' % 'Подходит ли кандидат на самом деле: ', result)
        print('---------------------------------------------------------------')
        print()


def тест_модели_классификации(модель=None, тестовый_набор=None, правильные_ответы=[], классы=None, количество=1):
  for i in range(количество):
    number = np.random.randint(тестовый_набор.shape[0]) # Задаем индекс изображения в тестовом наборе
    sample = тестовый_набор[number]
    if sample.shape == (784,):
      sample = sample.reshape((28,28))  
    if sample.shape == (28, 28, 1):
      sample = sample.reshape((28,28))
    print('Тестовое изображение:')
    plt.imshow(sample, cmap='gray') # Выводим изображение из тестового набора с заданным индексом
    plt.axis('off') # Отключаем оси
    plt.show() 

    sample = тестовый_набор[number].reshape((1 + модель.input.shape[1:]))
    pred = модель.predict(sample)[0] # Распознаем изображение с помощью обученной модели
    print()
    print('Результат предсказания модели:')

    def out_red(text):
      return "\033[4m\033[31m\033[31m{}\033[0m".format(text)

    def out_green(text):
      return "\033[4m\033[32m\033[32m{}\033[0m".format(text)

    def keywithmaxval(d):
      global r
      v = list(d.values())
      k = list(d.keys())
      return str(k[v.index(max(v))])
      

    dicts = {}

    for i in range(len(классы)):
      dicts[классы[i]] = round(100*pred[i],2)
      print('Модель распознала модель ',классы[i],' на ',round(100*pred[i],2),'%',sep='')
    print('---------------------------')

    answer = str(классы[правильные_ответы[number]])

    if len(правильные_ответы)>0:
      if keywithmaxval(dicts) == answer:
        print('Правильный ответ: ', out_green(answer))
        print('---------------------------')
        print()
        print()
      
      elif keywithmaxval(dicts) != классы[правильные_ответы[number]]:
        print('Правильный ответ: ', out_red(answer))
        print('---------------------------')
        print()
        print()

def тест_модели_трекер(модель=None, тестовый_набор=None, правильные_ответы=[], классы= ['not_same', 'same'], количество=1):
  classes = классы
  количество = количество
  тестовая_выборка = тестовый_набор
  predict = модель.predict(тестовая_выборка)
  fig = plt.figure(figsize=(16,7))
  for i, idx in enumerate(np.random.randint(0, predict.shape[0]-1, количество)):
      concat_img = тестовая_выборка[idx]
      label = np.argmax(predict[idx])
      first_img = concat_img[:,:,:3]
      second_img = concat_img[:,:,3:]
      ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
      ax.set_title(classes[label])
      plt.imshow(first_img)

      ax = fig.add_subplot(2, 5, i+6, xticks=[], yticks=[])
      plt.imshow(second_img)

def тест_на_своем_изображении(нейронка, размер_изображения, классы):
  fname = files.upload()
  fname = list(fname.keys())[0]
  sample = image.load_img(''+ fname, target_size=(размер_изображения[0], размер_изображения[1])) # Загружаем картинку
  img_numpy = np.array(sample)[None,...] # Преобразуем зображение в numpy-массив
  img_numpy = img_numpy/255

  number = np.random.randint(img_numpy.shape[0]) # Задаем индекс изображения в тестовом наборе
  sample = img_numpy[number]
  if sample.shape == (784,):
    sample = sample.reshape((28,28))  
  if sample.shape == (28, 28, 1):
    sample = sample.reshape((28,28))
  print('Тестовое изображение:')
  plt.imshow(sample, cmap='gray') # Выводим изображение из тестового набора с заданным индексом
  plt.axis('off') # Отключаем оси
  plt.show() 

  sample = img_numpy[number].reshape((1 + нейронка.input.shape[1:]))
  pred = нейронка.predict(sample)[0] # Распознаем изображение с помощью обученной модели
  print()
  print('Результат предсказания модели:')

  def keywithmaxval(d):
    global r
    v = list(d.values())
    k = list(d.keys())
    return str(k[v.index(max(v))])
    
  dicts = {}

  for i in range(len(классы)):
    dicts[классы[i]] = round(100*pred[i],2)
    print('Модель распознала модель ',классы[i],' на ',round(100*pred[i],2),'%',sep='')

  print('Нейронная сеть считает, что это: ', keywithmaxval(dicts))
    
def тест_модели_HR(gan_generator):
  def load_image(path):
    return np.array(Image.open(path))
  #Функция для преобразования картинки lr в sr
  def resolve(model, lr_batch):
      lr_batch = tf.cast(lr_batch, tf.float32)
      sr_batch = model(lr_batch)
      sr_batch = tf.clip_by_value(sr_batch, 0, 255)
      sr_batch = tf.round(sr_batch)
      sr_batch = tf.cast(sr_batch, tf.uint8)
      return sr_batch 
  def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]
  def resolve_and_plot(lr_image_path):
    lr = load_image(lr_image_path)    
    gan_sr = resolve_single(gan_generator, lr)    
    plt.figure(figsize=(10, 15))
    images = [lr, gan_sr]
    titles = ['Исходное изображение', 'Изображение после обработки']
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title, fontsize=10)
        plt.xticks([])
        plt.yticks([])  

  url = 'https://storage.googleapis.com/aiu_bucket/Examples.zip' # Указываем URL-файла
  output = 'Examples.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  # Скачиваем и распаковываем архив
  датасет.распаковать_архив(
      откуда = "Examples.zip",
      куда = "/content"
  )
  for file in os.listdir('demo1/'):
    resolve_and_plot('demo1/' + file)

def показать_график_обучения(**kwargs):
  keys = list(kwargs['статистика'].history.keys())
  for i in range(len(keys)//2):
    plt.figure(figsize=(12, 6)) # Создаем полотно для визуализации
    plt.plot(kwargs['статистика'].history[keys[i]], label ='Обучающая выборка') # Визуализируем график ошибки на обучающей выборке
    plt.plot(kwargs['статистика'].history['val_'+keys[i]], label ='Проверочная выборка') # Визуализируем график ошибки на проверочной выборке
    plt.legend() # Выводим подписи на графике
    if 'loss' in keys[i]:
      plt.title('График ошибки обучения модели') # Выводим название графика
    else:
      plt.title('График точности обучения модели') # Выводим название графика
    plt.show()

def загрузить_предобученную_модель():
  url = 'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1QYLIUQWWyqLvn8TCEAZiY7q8umv7lyYw' # Указываем URL-файла
  output = 'model.h5' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) 
  model = load_model('model.h5')
  return model

def создать_модель_HighResolution():
  LR_SIZE = 24
  HR_SIZE = 96
    #Коэффициенты для преобразования RGB
  DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255
  # Нормализует RGB изображения к промежутку [0, 1]
  def normalize_01(x):
    return x / 255.0
  # res_block
  def res_block(x_in, num_filters, momentum=0.8):
      x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
      x = BatchNormalization(momentum=momentum)(x)
      x = PReLU(shared_axes=[1, 2])(x)
      x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
      x = BatchNormalization(momentum=momentum)(x)
      x = Add()([x_in, x])
      return x  
  # Блок апсемплинга
  def upsample(x_in, num_filters):
      x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
      x = Lambda(pixel_shuffle(scale=2))(x)
      return PReLU(shared_axes=[1, 2])(x)   

  def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)     

  # Обратная нормализация
  def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean
  # Обратная нормализация
  def denormalize_m11(x):
      return (x + 1) * 127.5
      
  def sr_resnet(num_filters=64, num_res_blocks=16):
      x_in = Input(shape=(None, None, 3))
      x = Lambda(normalize_01)(x_in)

      x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
      x = x_1 = PReLU(shared_axes=[1, 2])(x)

      for _ in range(num_res_blocks):
          x = res_block(x, num_filters)

      x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
      x = BatchNormalization()(x)
      x = Add()([x_1, x])

      x = upsample(x, num_filters * 4)
      x = upsample(x, num_filters * 4)

      x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
      x = Lambda(denormalize_m11)(x)

      return Model(x_in, x)
  generator = sr_resnet
  return generator

#LabStory
#Авторизация
def авторизация_LabStory():
  global token
  global headers
  global user_id

  login = input('Введите логин:')
  password = getpass.getpass('Введите пароль:')
  headers = {"content-type": "application/json"} # Формирует заголовок (указываем, что тип контента json)
  post = 'http://labstory.neural-university.ru/api/login?email='+login+'&password='+password # Формирует post-запрос
  json_response = requests.post(post, headers=headers) # Отправляем post-запрос на сервер

  if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
    print('Авторизация успешно завершена')
    token = ast.literal_eval(json_response.text)['token'] # Сохраняем токен (ast.literal_eval - text to dict)
  elif json_response.status_code == 422: # Если пришел 422-ый код ответа (все хорошо)
    print('Неверный логи или пароль.')
    return
  else:
    print('Ошибка выполнения запроса')
    print('Код ошибки: ', json_response.status_code)
    print(json_response.text)
  get = 'http://labstory.neural-university.ru/api/my/profile?access_token='+token
  json_response = requests.get(get, headers=headers) # Отправляем post-запрос на сервер
  if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
    user = json_response.json()
  else:
    print('Ошибка выполнения запроса')
    print('Код ошибки: ', json_response.status_code)
    print(json_response.text)
  user_id = user['id']

# Добавление набора данных
def добавить_датасет_LabStory(dataset_dict):
  if 'token' not in globals():
    print('Для выполнения данной функции, необходимо авторизироваться.')
    return
  global датасет_id
  post = 'http://labstory.neural-university.ru/api/my/datasets?access_token='+token

  name = dataset_dict['name']
  description = dataset_dict['description']
  url = dataset_dict['url']
  data = {
      "url": url,
      "name": name,
      "description": description
  }
  json_response = requests.post(post, json=data, headers=headers)
  if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
    # вычисление id датасета путем получения списка всех датасетов и получения id последнего
    get = 'http://labstory.neural-university.ru/api/my/datasets?access_token='+token
    json_response = requests.get(get, headers=headers) # Отправляем post-запрос на сервер
    datasets = json_response.json()
    meta = datasets['meta']
    get = 'http://labstory.neural-university.ru/api/my/datasets?per_page=' + str(meta['total']) + '&access_token='+token
    json_response = requests.get(get, headers=headers)
    datasets = json_response.json()
    d = datasets['data'][-1]
    датасет_id = d["id"]
    print('Датасет успешно добавлен')
    print(f'id: {d["id"]}')
  else:
    print('Ошибка выполнения запроса')
    print('Код ошибки: ', json_response.status_code)
    print(json_response.text)
  
# Получение списка набора данных
def список_датасетов_LabStory():
  if 'token' not in globals():
    print('Для выполнения данной функции, необходимо авторизироваться.')
    return
  get = 'http://labstory.neural-university.ru/api/my/datasets?access_token='+token
  json_response = requests.get(get, headers=headers) # Отправляем post-запрос на сервер
  if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
    datasets = json_response.json()
    meta = datasets['meta']
    get = 'http://labstory.neural-university.ru/api/my/datasets?per_page='+ str(meta['total']) + '&access_token='+token
    json_response = requests.get(get, headers=headers)
    datasets = json_response.json()  
    i=1
    for d in datasets['data']:
      print('\033[1m', i, '. Датасет: ', d["name"], ' (\033[32mid ', d["id"], ')', ', \033[0m ', d["description"], sep='')
      print(' '*len(str(i)), ' Ссылка: ', d['url'])
      print(' '*len(str(i)), ' -----------------------')
      print()
      i+=1
    print()
    print('\033[1m Всего датасетов: ', meta['total'])
  else:
    print('Ошибка выполнения запроса')
    print('Код ошибки: ', json_response.status_code)
    print(json_response.text)

  
# Получение списка набора данных
def список_датасетов_LabStory():
  if 'token' not in globals():
    print('Для выполнения данной функции, необходимо авторизироваться.')
    return
  get = 'http://labstory.neural-university.ru/api/my/datasets?access_token='+token
  json_response = requests.get(get, headers=headers) # Отправляем post-запрос на сервер
  if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
    datasets = json_response.json()
    meta = datasets['meta']
    get = 'http://labstory.neural-university.ru/api/my/datasets?per_page='+ str(meta['total']) + '&access_token='+token
    json_response = requests.get(get, headers=headers)
    datasets = json_response.json()  
    i=1
    for d in datasets['data']:
      print('\033[1m', i, '. Датасет: ', d["name"], ' (\033[32mid ', d["id"], ')', ', \033[0m ', d["description"], sep='')
      print(' '*len(str(i)), ' Ссылка: ', d['url'])
      print(' '*len(str(i)), ' -----------------------')
      print()
      i+=1
    print('\033[1m Всего датасетов: ', meta['total'])
  else:
    print('Ошибка выполнения запроса')
    print('Код ошибки: ', json_response.status_code)
    print(json_response.text)
    
# Удаление набора данных
def удалить_датасет_LabStory(id):
  if 'token' not in globals():
    print('Для выполнения данной функции, необходимо авторизироваться.')
    return
  delete = f'http://labstory.neural-university.ru/api/my/datasets/{id}?access_token='+token
  json_response = requests.delete(delete)
  if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
    print('Датасет успешно удален')
  elif json_response.status_code == 500:
    print('Невозможно удалить датасет который привязан к эксперименту')
  elif json_response.status_code == 404:
    print('Невозможно удалить датасет. Датасета с таким id не существует...')
 
# Выбор датасета по id
def выбрать_датасет_LabStory(id):
  if 'token' not in globals():
    print('Для выполнения данной функции, необходимо авторизироваться.')
    return
  global датасет_id
  get = f'http://labstory.neural-university.ru/api/my/datasets/{id}?access_token='+token
  json_response = requests.get(get, headers=headers) # Отправляем post-запрос на сервер
  if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
    datasets = json_response.json()
    датасет_id = datasets['id']
    print('\033[1mВыбран датасет: ', )
    print(datasets["name"], ' (\033[32mid ', datasets["id"], ')', ', \033[0m ', datasets["description"], sep='')
    print('Ссылка: ', datasets['url'])
  elif json_response.status_code == 404:
    print('\033[1mОшибка выполнения запроса')
    print('\033[0mДатасета с id', id, '\033[0mне существует в Вашем аккаунте')
    
# Вывод информации о текущем датасете
def текущий_датасет():
  print('id', датасет_id) 

# Создание задачи
def добавить_задачу_LabStory(task_dict):
  if 'token' not in globals():
    print('Для выполнения данной функции, необходимо авторизироваться.')
    return
  global задача_id
  post = 'http://labstory.neural-university.ru/api/my/tasks?access_token='+token

  name = task_dict['name']
  description = task_dict['description']
  task = {
      "name": name,
      "description": description
  }
  json_response = requests.post(post, json=task, headers=headers)
  if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
    print('Задача успешно добавлена')
  else:
    print('Ошибка выполнения запроса')
    print('Код ошибки: ', json_response.status_code)
    print(json_response.text)

  # получаем id только что добавленной задачи через поиск id всех задач
  get = 'http://labstory.neural-university.ru/api/my/tasks?access_token='+token
  json_response = requests.get(get, headers=headers) # Отправляем post-запрос на сервер
  if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
    tasks = json_response.json()
    meta = tasks['meta']
    get = 'http://labstory.neural-university.ru/api/my/tasks?per_page=' + str(meta['total']) + '&access_token='+token
    json_response = requests.get(get, headers=headers)
    tasks = json_response.json()
    task_id = tasks['data'][-1]
    задача_id = task_id['id']
    print('id задачи: ',task_id['id'] )
  else:
    print('Ошибка выполнения запроса')
    print('Код ошибки: ', json_response.status_code)
    print(json_response.text)
  
# Получение списка задач
def список_задач_LabStory():
  if 'token' not in globals():
    print('Для выполнения данной функции, необходимо авторизироваться.')
    return
  get = 'http://labstory.neural-university.ru/api/my/tasks?access_token='+token
  json_response = requests.get(get, headers=headers) # Отправляем post-запрос на сервер
  if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
    tasks = json_response.json()
    meta = tasks['meta']
    i=1
    get = 'http://labstory.neural-university.ru/api/my/tasks?per_page=' + str(meta['total']) + '&access_token='+token
    json_response = requests.get(get, headers=headers)
    tasks = json_response.json()
    i=1
    for t in tasks['data']:
      print(f'\033[1m {i}. Задача: {t["name"]} \033[32m(id {t["id"]})\033[0m, {t["description"]}')
      print(' '*len(str(i)), '  \033[1mЭкспериментов:\033[0m ', t['experiments_count'])
      print('-----------------------')
      print()
      i+=1
    print()
    print('\033[1m Всего задач: ', meta['total'])
  else:
    print('Ошибка выполнения запроса')
    print('Код ошибки: ', json_response.status_code)
    print(json_response.text)
    
# Выбор задачи по id
def выбрать_задачу_LabStory(id):
  if 'token' not in globals():
    print('Для выполнения данной функции, необходимо авторизироваться.')
    return
  global задача_id
  get = f'http://labstory.neural-university.ru/api/my/tasks/{id}?access_token='+token
  json_response = requests.get(get, headers=headers) # Отправляем post-запрос на сервер
  if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
    task = json_response.json()
    задача_id = task['id']
    print('\033[1mВыбрана задача (id ', задача_id, '):', sep='')
    print(task['name'], ', \033[0m', task['description'], sep='')
    print('Экспериментов: ', task['experiments_count'])
  elif json_response.status_code == 404:
    print('Ошибка выполнения запроса')
    print(f'Задачи c id {id} не существует в Вашем аккаунте.')
    
def текущая_задача():
  print('id', задача_id)
  
# Сохранение эксперимента
def сохранить_эксперимент_LabStory(experiment_dict):
  global model_architecture
  if 'token' not in globals():
    print('Для выполнения данной функции, необходимо авторизироваться.')
    return
  if 'задача_id' not in globals():
    print('Не выбрана задача')
    print('Выберите или создайте задачу и повторите действие')
    return
  if 'датасет_id' not in globals():
    print('Не выбран датасет')
    print('Выберите или создайте датасет и повторите действие')  
    return
  post = 'http://labstory.neural-university.ru/api/my/experiments?access_token='+token

  model_history_list = []
  for key in experiment_dict['history'].history.keys():
    model_history_list.append(experiment_dict['history'].history[f'{key}'])  

  loss_float, metric_float, val_loss_float, val_metric_float = '', '', '', ''
  loss, metric, val_loss, val_metric = '', '', '', ''

  if experiment_dict['loss'] == min:
    loss_float = min(model_history_list[0])
    loss = [round(v, 4) for v in model_history_list[0]]
    try:
      val_loss_float = min(model_history_list[2])
      val_loss = [round(v, 4) for v in model_history_list[2]]
    except IndexError:
      val_loss_float = 0
      val_loss = [0]
  elif experiment_dict['loss'] == max:
    loss_float = max(model_history_list[0])
    loss = [round(v, 4) for v in model_history_list[0]]
    try:
      val_loss_float = max(model_history_list[2])
      val_loss = [round(v, 4) for v in model_history_list[2]]
    except IndexError:
      val_loss_float = 0
      val_loss = [0]

  if experiment_dict['metrics'] == min:
    metric_float = min(model_history_list[1])
    metric = [round(v, 4) for v in model_history_list[1]]
    try:
      val_metric_float = min(model_history_list[3])
      val_metric = [round(v, 4) for v in model_history_list[3]]
    except IndexError:
      val_metric_float = 0
      val_metric = [0]
  elif experiment_dict['metrics'] == max:
    metric_float = max(model_history_list[1])
    metric = [round(v, 4) for v in model_history_list[1]]
    try:
      val_metric_float = max(model_history_list[3])
      val_metric = [round(v, 4) for v in model_history_list[3]]
    except IndexError:
      val_metric_float = 0
      val_metric = [0]
###############################
# Поиск архитектуры модели    
###############################
  #sum_=0
  find = experiment_dict['function']
  #key_words = ['слои', 'Сверточный2D', 'Dense', 'Conv2D']
  for i in reversed(experiment_dict['cache']):  
    #for j in range(len(key_words)):
      if (find in i) and ('experiment_dict' not in i):
        model_architecture = i
        break
      #else: sum_+=1
  #print(m_arc)
  #print(sum_, len(experiment_dict['cache']))   
###############################
  try:
    model_architecture
  except NameError:
    print('Не найдено ячейки с комментарием указанным в качестве значения ключа "function" в словаре experiment_dict') 
    return
  experiment = {
    "task_id": задача_id,
    "user_id": user_id,
    "dataset_id": датасет_id,
    "name_model": experiment_dict['name'],
    "loss": loss,
    "loss_float": loss_float,
    "val_loss": val_loss,
    "val_loss_float": val_loss_float,
    "metric": metric,
    "metric_float": metric_float,
    "val_metric": val_metric,
    "val_metric_float": val_metric_float,
    "function_creating_model": model_architecture,
    "function_data_preprocessing": experiment_dict['data_processing_type'],
    "description_model": experiment_dict['description'],
    "description_data_preprocessing": experiment_dict['description_data_processing_type'],
    "comment": experiment_dict['comment'],
    "tags": experiment_dict['tags'],
  }
  json_response = requests.post(post, json=experiment, headers=headers)
  if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
# получаем id только что сохраненного эксперимента по id всех экспериментов
    get = 'http://labstory.neural-university.ru/api/my/experiments?access_token='+token
    json_response = requests.get(get, headers=headers) # Отправляем post-запрос на сервер
    if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
      experiments = json_response.json()
      meta = experiments['meta']
      get = 'http://labstory.neural-university.ru/api/my/experiments?per_page=' + str(meta['total']) + '&access_token='+token
      json_response = requests.get(get, headers=headers)
      experiments = json_response.json()
      experiment_id = experiments['data'][-1]
      эксперимент_id = experiment_id['id']
      print('Эксперимент успешно сохранен')
      print('id эксперимента:', experiment_id['id'] )
  else:
    print('Ошибка выполнения запроса')
    print('Код ошибки: ', json_response.status_code)
    print(json_response.text)
      

# Получение списка экспериметов
def список_экспериметов_LabStory():
  if 'token' not in globals():
    print('Для выполнения данной функции, необходимо авторизироваться.')
    return
  get = 'http://labstory.neural-university.ru/api/my/experiments?access_token='+token
  json_response = requests.get(get, headers=headers) # Отправляем post-запрос на сервер
  if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
    experiments = json_response.json()
    meta = experiments['meta']
    get = 'http://labstory.neural-university.ru/api/my/experiments?per_page=' + str(meta['total']) + '&access_token='+token
    json_response = requests.get(get, headers=headers)
    experiments = json_response.json()
    i=1
    print()
    for e in experiments['data']:
      print(i,'. id эксперимента: ', e['id'], sep='')
      print('-----------------------')
      i+=1
    print()
    print('\033[1mВсего экспериментов: ', len(experiments['data']))
  else:
    print('Ошибка выполнения запроса')
    print('Код ошибки: ', json_response.status_code)
    print(json_response.text)
 
# Получение эксперимета по id
def посмотреть_эксперимент_по_id_LabStory(id):
  if 'token' not in globals():
    print('Для выполнения данной функции, необходимо авторизироваться.')
    return
    
  get = f'http://labstory.neural-university.ru/api/my/experiments/{id}?access_token='+token
  json_response = requests.get(get, headers=headers) # Отправляем post-запрос на сервер
  if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
    experiment = json_response.json()
    columns = ['Название', 'Описание', 'val_metrics', 'val_loss',  'Комментарий', 'Датасет', 'Дата создания']
    data = [[experiment['name_model'],
             experiment['description_model'],
             experiment['val_metric_float'],
             experiment['val_loss_float'],
             experiment['comment'],
             experiment['dataset']['name'],
             experiment['created_at']]]     
    print(tabulate(data, headers=columns))
  else:
    print('Ошибка выполнения запроса')
    print(f'Эксперимента c id {id} не существует в Вашем аккаунте.')
    
def получить_архитектуру(id_exp):
  if 'token' not in globals():
    print('Для выполнения данной функции, необходимо авторизироваться.')
    return
    
  get = f'http://labstory.neural-university.ru/api/my/experiments/{id_exp}?access_token='+token
  json_response = requests.get(get, headers=headers) # Отправляем post-запрос на сервер
  if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
    experiment = json_response.json()
    получить_архитектуру = experiment['function_creating_model'] 
  else:
    print('Ошибка выполнения запроса')
    print(f'Эксперимента c id {id_exp} не существует в Вашем аккаунте.')
  return получить_архитектуру

def все_эксперименты_по_задаче(id):
  if 'token' not in globals():
    print('Для выполнения данной функции, необходимо авторизироваться.')

  get = f'http://labstory.neural-university.ru/api/my/tasks/{id}?access_token='+token
  json_response = requests.get(get, headers=headers) # Отправляем post-запрос на сервер
  if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
    get = 'http://labstory.neural-university.ru/api/my/experiments?access_token='+token
    json_response = requests.get(get, headers=headers) # Отправляем post-запрос на сервер
    if json_response.status_code == 200: # Если пришел 200-ый код ответа (все хорошо)
      experiments = json_response.json()
      meta = experiments['meta']
      get = 'http://labstory.neural-university.ru/api/my/experiments?per_page=' + str(meta['total']) + '&access_token='+token
      json_response = requests.get(get, headers=headers)
      experiments = json_response.json()
      num_=1
      for i in range(len(experiments['data'])):
        if experiments['data'][i]['task_id'] == id:
          print(num_, '. id эксперимента: ', experiments['data'][i]['id'], sep='')
          num_+=1
      print()
      print('-----------------------')
      print()
      print('\033[1mВсего экспериментов: ', num_-1)

  elif json_response.status_code == 404:
    print('Ошибка выполнения запроса')
    print(f'Задачи c id {id} не существует в Вашем аккаунте.')


#Демонстрация

def демонстрация_МНИСТ():
  global x_train_org
  global y_train_org
  global x_test_org
  global y_test_org
  global x_train
  global x_train
  global x_test
  global x_test
  global model
  global imgs
  global weights
  global losses
  global accuracy
  global model
  global idx

  (x_train_org, y_train_org), (x_test_org, y_test_org) = mnist.load_data()
  x_train = x_train_org.reshape((-1, 784))
  x_train = x_train/255

  x_test = x_test_org.reshape((-1, 784))
  x_test = x_test/255

  model = Sequential()
  model.add(Dense(10, use_bias=False, activation='softmax', input_dim=784))
  #model.trainable = False
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  imgs = []
  weights = [model.get_weights()[0]]
  losses = []
  accuracy = []
  idx = 0
  def on_batch_end(batch, logs):
    global idx
    if idx%60==0:
      imgs.append(model.get_weights()[0])
    idx+=1  

  def on_epoch_end(epoch, logs):
    weights.append(model.get_weights()[0])
    losses.append(logs['loss'])
    accuracy.append(logs['accuracy'])
  print('Ожидайте, идет подготовка к демонстрации')
  lc = LambdaCallback(on_batch_end=on_batch_end, on_epoch_end=on_epoch_end)
  model.fit(x_train, y_train_org, batch_size=100, epochs=10, verbose=0, callbacks=[lc])
  display.clear_output(wait=True)
  print('Демонстрация готова')
  
def показать_изменение_маски_MNIST(number):
  for n in range(len(number)):
    numeric = np.array(imgs)[...,number[n]]
    f, ax = plt.subplots(3,8,figsize=(28, 10))
    for i in range(3):
      for j in range(8):
        ax[i,j].imshow(numeric[i*8+j].reshape((28,28)), cmap='gray')
        ax[i,j].axis('off')

       

def показать_изменение_веса():
  num = 4
  global weights
  weights = np.array(weights)
  w = weights[:,np.random.randint(784),num]
  d = np.diff(w)
  import pandas as pd
  df = pd.DataFrame(columns=['Значение веса','Изменение веса','Новое значение', 'Ошибка модели', 'Точность модели'])
  df.index.name = 'Эпоха'
  for i in range(10):
    df.loc[i] = [w[:-1][i], d[i], w[1:][i], losses[i], accuracy[i]]
  return df.head(10)

def демонстрация_АВТО():
  global idx
  global imgs
  global обучающая_выборка
  global part1
  global part2
  global part3
  global part4

  путь = 'автомобили'
  коэф_разделения=0.9
  обучающая_выборка = [] # Создаем пустой список, в который будем собирать примеры изображений обучающей выборки
  y_train = [] # Создаем пустой список, в который будем собирать правильные ответы (метки классов: 0 - Феррари, 1 - Мерседес, 2 - Рено)
  x_test = [] # Создаем пустой список, в который будем собирать примеры изображений тестовой выборки
  y_test = [] # Создаем пустой список, в который будем собирать правильные ответы (метки классов: 0 - Феррари, 1 - Мерседес, 2 - Рено)

  for j, d in enumerate(sorted(os.listdir(путь))):
    files = sorted(os.listdir(путь + '/'+d))    
    count = len(files) * коэф_разделения
    for i in range(len(files)):
      sample = image.load_img(путь + '/' +d +'/'+files[i], target_size=(54, 96)) # Загружаем картинку
      img_numpy = np.array(sample) # Преобразуем зображение в numpy-массив
      if i<count:
        обучающая_выборка.append(img_numpy) # Добавляем в список x_train сформированные данные
        y_train.append(j) # Добавлеям в список y_train значение 0-го класса
      else:
        x_test.append(img_numpy) # Добавляем в список x_test сформированные данные
        y_test.append(j) # Добавлеям в список y_test значение 0-го класса
  display.clear_output(wait=True)
  x_train = np.array(обучающая_выборка) # Преобразуем к numpy-массиву
  y_train = np.array(y_train) # Преобразуем к numpy-массиву
  x_test = np.array(x_test) # Преобразуем к numpy-массиву
  y_test = np.array(y_test) # Преобразуем к numpy-массиву
  x_train = x_train/255.
  x_test = x_test/255.
  
  inp = Input(shape=(54,96,3))
  x1 = Conv2D(8, (3,3), activation='relu', padding='same') (inp)
  x2 = Conv2D(8, (3,3), activation='relu', padding='same') (x1)
  x3 = Conv2D(8, (3,3), activation='relu', padding='same') (x2)
  x4 = Conv2D(8, (3,3), activation='relu', padding='same') (x3)
  x = Flatten() (x4)
  x = Dense(10, activation='softmax')(x)

  part1 = Model(inp, x1)
  part2 = Model(inp, x2)
  part3 = Model(inp, x3)
  part4 = Model(inp, x4)
  model = Model(inp, x)
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  imgs = []
  sample = random.choice(x_train)
  idx = 0
  def on_batch_end(batch, logs):
    global idx
    if idx%25==0:
      imgs.append(part1.predict(sample[None,...]).reshape((54,96,8)))
    idx+=1
  print('Ожидайте, идет подготовка к демонстрации')
  lc = LambdaCallback(on_batch_end=on_batch_end)
  model.fit(x_train, y_train, epochs=10, batch_size=32, callbacks=[lc], verbose=0)
  display.clear_output(wait=True)
  print('Демонстрация готова')
  
def показать_изменение_маски_АВТО(num):
  f, ax = plt.subplots(3,6,figsize=(20, 8))
  for i in range(3):
    for j in range(6):
      ax[i,j].imshow(imgs[i*6+j][:,:,num].reshape((54,96)), cmap='gray')
      ax[i,j].axis('off')

def показать_маски():
  import warnings
  warnings.filterwarnings('ignore')
  sample = random.choice(обучающая_выборка)
  images1 = part1.predict(sample[None,...]).reshape((54,96,8))
  images2 = part2.predict(sample[None,...]).reshape((54,96,8))
  images3 = part3.predict(sample[None,...]).reshape((54,96,8))
  images4 = part4.predict(sample[None,...]).reshape((54,96,8))
  print('*** Оригинальное изображение ***')
  plt.imshow(sample)
  plt.axis('off')
  plt.show()
  print()

  print('*** Карты первого сверточного слоя ***')
  f, ax = plt.subplots(1,8, figsize=(40,25))
  for i in range(8):
    ax[i].imshow(images1[:,:,i], cmap='gray')
    ax[i].axis('off')
  plt.show()
  print()

  print('*** Карты второго сверточного слоя ***')
  f, ax = plt.subplots(1,8, figsize=(40,25))
  for i in range(8):
    ax[i].imshow(images2[:,:,i], cmap='gray')
    ax[i].axis('off')
  plt.show()
  print()

  print('*** Карты третьего сверточного слоя ***')
  f, ax = plt.subplots(1,8, figsize=(40,25))
  for i in range(8):
    ax[i].imshow(images3[:,:,i], cmap='gray')
    ax[i].axis('off')
  plt.show()
  print()

  print('*** Карты четвертого сверточного слоя ***')
  f, ax = plt.subplots(1,8, figsize=(40,25))
  for i in range(8):
    ax[i].imshow(images4[:,:,i], cmap='gray')
    ax[i].axis('off')
  plt.show()
