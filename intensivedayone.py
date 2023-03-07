#@title Установка модуля УИИ
from PIL import Image
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from IPython import display as ipd

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns 

import gdown
import zipfile
import os
import random
import time 
import gc

sns.set(style='darkgrid') 
seed_value = 12
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class AccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        self.times = []


    def plot_graph(self):        
        plt.figure(figsize=(20, 14))
        plt.subplot(2, 2, 1)
        plt.title('Точность', fontweight='bold')
        plt.plot(self.train_acc, label='Точность на обучащей выборке')
        plt.plot(self.val_acc, label='Точность на проверочной выборке')
        plt.xlabel('Эпоха обучения')
        plt.ylabel('Доля верных ответов')
        plt.legend()        
        plt.show()
       

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.train_acc.append(logs['accuracy'])
        self.val_acc.append(logs['val_accuracy'])
        self.train_loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])
        t = round(time.time() - self.start_time, 1)
        self.times.append(t)
        if logs['val_accuracy'] > self.accuracymax:
            self.accuracymax = logs['val_accuracy']
            self.idxmax = epoch
        print(f'Эпоха {epoch+1}'.ljust(10)+ f'Время обучения: {t}c'.ljust(25) + f'Точность на обучающей выборке: {bcolors.OKBLUE}{round(logs["accuracy"]*100,1)}%{bcolors.ENDC}'.ljust(50) +f'Точность на проверочной выборке: {bcolors.OKBLUE}{round(logs["val_accuracy"]*100,1)}%{bcolors.ENDC}')
        self.cntepochs += 1

    def on_train_begin(self, logs):
        self.idxmax = 0
        self.accuracymax = 0
        self.cntepochs = 0

    def on_train_end(self, logs):
        ipd.clear_output(wait=True)
        for i in range(self.cntepochs):
            if i == self.idxmax:
                print('\33[102m' + f'Эпоха {i+1}'.ljust(10)+ f'Время обучения: {self.times[i]}c'.ljust(25) + f'Точность на обучающей выборке: {round(self.train_acc[i]*100,1)}%'.ljust(41) +f'Точность на проверочной выборке: {round(self.val_acc[i]*100,1)}%'+ '\033[0m')
            else:
                print(f'Эпоха {i+1}'.ljust(10)+ f'Время обучения: {self.times[i]}c'.ljust(25) + f'Точность на обучающей выборке: {bcolors.OKBLUE}{round(self.train_acc[i]*100,1)}%{bcolors.ENDC}'.ljust(50) +f'Точность на проверочной выборке: {bcolors.OKBLUE}{round(self.val_acc[i]*100,1)}%{bcolors.ENDC}' )
        self.plot_graph()

class TerraDataset:
    bases = {
        'Молочная_продукция' : {
            'url': 'https://storage.yandexcloud.net/terraai/sources/milk.zip',
            'info': 'Вы скачали базу с изображениями бутылок молока. База содержит 1500 изображений трех категорий: «Parmalat», «Кубанская буренка», «Семейный формат»',
            'dir_name': 'milk_ds',
            'task_type': 'img_classification',
            'size': (96, 53),
        },
        'Пассажиры_автобуса' : {
            'url': 'https://storage.yandexcloud.net/terraai/sources/bus.zip',
            'info': 'Вы скачали базу с изображениями пассажиров автобуса. База содержит 9081 изображение двух категорий: «Входящие пассажиры», «Выходящие пасажиры»',
            'dir_name': 'passengers',
            'task_type': 'img_classification',
            'size': (128, 64),
        },
        'Возгорания' : {
            'url': 'https://storage.yandexcloud.net/terraai/sources/fire.zip',
            'info': 'Вы скачали базу с изображениями возгораний. База содержит 6438 изображение двух категорий: «Есть возгорание», «Нет возгорания»',
            'dir_name': 'fire',
            'task_type': 'img_classification',
            'size': (96, 76),
        },
        'авто' : {
            'url': 'https://storage.yandexcloud.net/aiueducation/Intensive/cars.zip',
            'info': 'Вы скачали базу с изображениями марок авто. База содержит 3427 изображений трех категорий: «Феррари», «Мерседес», «Рено»',
            'dir_name': 'car',
            'task_type': 'img_classification',
            'size': (54, 96),
        },
        'майонез' : {
            'url': 'https://storage.yandexcloud.net/aiueducation/Intensive/mayonnaise.zip',
            'info': 'Вы скачали базу с изображениями брендов майонеза. База содержит 150 изображений трех категорий: «ЕЖК», «Махеев», «Ряба»',
            'dir_name': 'mayonesse',
            'task_type': 'img_classification',
            'size': (96, 76),
        },
    }
    def __init__(self, name):
        '''
        parameters:
            name - название датасета
        '''        
        self.base = self.bases[name]
        self.sets = None
        self.classes = None

    def load(self):
        '''
        функция загрузки датасета
        '''
        
        print(f'{bcolors.BOLD}Загрузка датасета{bcolors.ENDC}',end=' ')
        
        # Загурзка датасета из облака
        fname = gdown.download(self.base['url'], None, quiet=True)

        if Path(fname).suffix == '.zip':
            # Распаковка архива
            with zipfile.ZipFile(fname, 'r') as zip_ref:
                zip_ref.extractall(self.base['dir_name'])

            # Удаление архива
            os.remove(fname)

        # Вывод информационного блока
        print(f'{bcolors.OKGREEN}Ok{bcolors.ENDC}')
        print(f'{bcolors.OKBLUE}Инфо:{bcolors.ENDC}')
        print(f'    {self.base["info"]}')
        return self.base['task_type']

    def samples(self):
        '''
        Функция визуализации примеров
        '''
        
        # Визуализация датасета изображений для задачи классификации
        if self.base['task_type'] == 'img_classification':
            # Получение списка классов (названия папок в директории)
            self.classes = sorted(os.listdir(self.base['dir_name']))

            # Построение полотная визуализации
            f, ax = plt.subplots(len(self.classes), 5, figsize=(24, len(self.classes) * 4))
            for i, class_ in enumerate(self.classes):
                # Выбор случайного изображения
                for j in range(5):
                  random_image = random.choice(
                      os.listdir(os.path.join(
                          self.base['dir_name'], 
                          class_)))
                  img = Image.open(os.path.join(
                      self.base['dir_name'],
                      class_,
                      random_image))
                  ax[i, j].imshow(img)
                  ax[i, j].axis('off')
                  ax[i, j].set_title(class_)
            plt.show()   

    def create_sets(self):
        '''
        Функция создания выборок
        '''
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        print(f'{bcolors.BOLD}Создание наборов данных для обучения модели{bcolors.ENDC}', end=' ')

        # Создание выборок для задачи классификации изображений
        if self.base['task_type'] == 'img_classification':

            # Получение списка директорий
            self.classes = sorted(os.listdir(self.base['dir_name']))
            counts = []

            # Проход по всем папкам директории (по всем классам)
            for j, d in enumerate(self.classes):

              # Получение списка всех изображений очередного класса
              files = sorted(os.listdir(os.path.join(self.base['dir_name'], d)))

              # Параметр разделения выборок
              counts.append(len(files))
              count = counts[-1] * .9

              # Проход по всем изображениям очередного класса
              for i in range(len(files)):
                  
                  # Загрузка очередного изображения
                  sample = np.array(image.load_img(os.path.join(
                      self.base['dir_name'],
                      d,
                      files[i]), target_size=self.base['size']))
                  
                  # Добавление элемента в тестовую или проверочную выборку
                  if i<count:
                    x_train.append(sample)
                    y_train.append(j)
                  else:
                    x_test.append(sample)
                    y_test.append(j)
            self.sets = (np.array(x_train)/255., np.array(y_train)), (np.array(x_test)/255., np.array(y_test))

            # Вывод финальной информации
            print(f'{bcolors.OKGREEN}Ok{bcolors.ENDC}')
            print()
            print(f'Размер созданных выборок:')
            print(f'  Обучающая выборка: {self.sets[0][0].shape}')
            print(f'  Метки обучающей выборки: {self.sets[0][1].shape}')
            print(f'  Проверочная выборка: {self.sets[1][0].shape}')
            print(f'  Метки проверочной выборки: {self.sets[1][1].shape}')
            print()
            print(f'Распределение по классам:')
            f, ax =plt.subplots(1,2, figsize=(16, 5))            
            ax[0].bar(self.classes, np.array(counts)*0.9)
            ax[0].set_title('Обучающая выборка')
            ax[1].bar(self.classes, np.array(counts)*0.1, color='g')
            ax[1].set_title('Проверочная выборка')
            plt.show()


class TerraModel:
    def __init__(self, task_type, trds):
        self.model = None
        self.task_type = task_type
        self.trds = trds
    
    @staticmethod
    def create_layer(params):
        '''
           Функция создания слоя
        '''
        activation = 'relu'
        params = params.split('-')
        
        # Добавление входного слоя
        if params[0].lower() == 'входной':
            return Input(shape=eval(params[1]))

        # Добавление полносвязного слоя
        if params[0].lower() == 'полносвязный':
            if len(params)>2:
                activation = params[2]
            return Dense(eval(params[1]), activation=activation)

        # Добавление выравнивающего слоя
        if params[0].lower() == 'выравнивающий':
            return Flatten()

        # Добавление сверточного слоя (Conv2D)
        if params[0].lower() == 'сверточный2д':
            if len(params)>3:
                activation = params[3]
            return Conv2D(eval(params[1]), eval(params[2]), activation=activation, padding='same')
            
    def create_model(self, layers):
        '''
        Функция создания нейронной сети
        parameters:
            layers - слои (текстом)
        '''
        if self.task_type=='img_classification':
            layers += '-softmax'        
        layers = layers.split()
        # Создание входного слоя
        inp = self.create_layer(f'входной-{self.trds.sets[0][0].shape[1:]}')

        # Создание первого слоя
        x = self.create_layer(layers[0]) (inp)

        # Создание остальных слоев
        for layer in layers[1:]:
            x = self.create_layer(layer) (x)            
        self.model = Model(inp, x)        

    def train_model(self, epochs, use_callback=True):
        '''
        Функция обучения нейронной сети
        parameters:
            epochs - количество эпох
        '''
        
        # Обучение модели классификации изображений
        if self.task_type=='img_classification':
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer = Adam(0.0001), metrics=['accuracy'])
            accuracy_callback = AccuracyCallback()
            callbacks = []
            if use_callback:
                callbacks = [accuracy_callback]
            history = self.model.fit(self.trds.sets[0][0], self.trds.sets[0][1],
                          batch_size = self.trds.sets[0][0].shape[0]//25,
                          validation_data=(self.trds.sets[1][0], self.trds.sets[1][1]),
                          epochs=epochs,
                          callbacks=callbacks,
                          verbose = 0)
            return history
            
    def test_model(self):
        '''
        Функция тестирования модели
        '''
        # Тестирование модели классификации изображений
        if self.task_type=='img_classification':
            for i in range(10):
                number = np.random.randint(self.trds.sets[1][0].shape[0])
                sample = self.trds.sets[1][0][number]
                print('Тестовое изображение:')
                plt.imshow(sample) # Выводим изображение из тестового набора с заданным индексом
                plt.axis('off') # Отключаем оси
                plt.show() 
                pred = self.model.predict(sample[None, ...])[0]
                max_idx = np.argmax(pred)
                print()
                print('Результат предсказания модели:')
                for i in range(len(self.trds.classes)):
                    if i == max_idx:
                        print(bcolors.BOLD, end='')
                    print(f'Модель распознала класс «{self.trds.classes[i]}» на {round(100*pred[i],1)}%{bcolors.ENDC}')
                print('---------------------------')
                print('Правильный ответ: ',end='')
                if max_idx == self.trds.sets[1][1][number]:
                    print(bcolors.OKGREEN, end='')
                else:
                    print(bcolors.FAIL, end='')
                print(self.trds.classes[self.trds.sets[1][1][number]],end=f'{bcolors.ENDC}\n')
                print('---------------------------')
                print()
                print()


class TerraIntensive:
    def __init__(self):
       self.trds = None
       self.trmodel = None
       self.task_type = None

    def load_dataset(self, ds_name):
        self.trds = TerraDataset(ds_name)
        self.task_type = self.trds.load()

    def samples(self):
        self.trds.samples()

    def create_sets(self):
        self.trds.create_sets()

    def create_model(self, layers):
        print(f'{bcolors.BOLD}Создание модели нейронной сети{bcolors.ENDC}', end=' ')
        self.trmodel = TerraModel(self.task_type, self.trds)
        self.trmodel.create_model(layers)
        print(f'{bcolors.OKGREEN}Ok{bcolors.ENDC}')

    def train_model(self, epochs):
        self.trmodel.train_model(epochs)

    def test_model(self):
        self.trmodel.test_model()

    def train_model_average(self, layers, cnt=10):
        if self.task_type == 'img_classification':
          print(f'{bcolors.BOLD}Определение среднего показателя точности модели на {cnt} запусках{bcolors.ENDC}')
          print()
          average_accuracy = []
          average_val_accuracy = []
          times=[]
          for i in range(cnt):
              start_time = time.time()
              self.trmodel.create_model(layers)
              history = self.trmodel.train_model(20, False).history
              average_accuracy.append(np.max(history['accuracy']))
              average_val_accuracy.append(np.max(history['val_accuracy']))
              t = round(time.time() - start_time, 1)
              times.append(t)
              print(f'Запуск {i+1}'.ljust(10)+ f'Время обучения: {t}c'.ljust(25) + f'Точность на обучающей выборке: {bcolors.OKBLUE}{round(average_accuracy[-1]*100,1)}%{bcolors.ENDC}'.ljust(50) +f'Точность на проверочной выборке: {bcolors.OKBLUE}{round(average_val_accuracy[-1]*100,1)}%{bcolors.ENDC}')
              gc.collect()
               
          ipd.clear_output(wait=True)
          print(f'{bcolors.BOLD}Определение среднего показателя точности модели на {cnt} запусках{bcolors.ENDC}')
          print()
          argmax_idx = np.argmax(average_val_accuracy)
          for i in range(cnt):
              if i == argmax_idx:
                  print('\33[102m' + f'Запуск {i+1}'.ljust(10)+ f'Время обучения: {times[i]}c'.ljust(25) + f'Точность на обучающей выборке: {round(average_accuracy[i]*100,1)}%'.ljust(41) +f'Точность на проверочной выборке: {round(average_val_accuracy[i]*100,1)}%'+ '\033[0m')
              else:
                  print(f'Запуск {i+1}'.ljust(10)+ f'Время обучения: {times[i]}c'.ljust(25) + f'Точность на обучающей выборке: {bcolors.OKBLUE}{round(average_accuracy[i]*100,1)}%{bcolors.ENDC}'.ljust(50) +f'Точность на проверочной выборке: {bcolors.OKBLUE}{round(average_val_accuracy[i]*100,1)}%{bcolors.ENDC}' )
          print()
          print(f'{bcolors.BOLD}Средняя точность на обучающей выборке: {bcolors.ENDC}{round(np.mean(average_accuracy[i])*100,1)}%')
          print(f'{bcolors.BOLD}Максимальная точность на обучающей выборке: {bcolors.ENDC}{round(np.max(average_accuracy[i])*100,1)}%')
          print(f'{bcolors.BOLD}Средняя точность на проверочной выборке: {round(np.mean(average_val_accuracy[i])*100,1)}%')
          print(f'{bcolors.BOLD}Максимальная точность на проверочной выборке: {round(np.max(average_val_accuracy[i])*100,1)}%')


terra_ai = TerraIntensive()