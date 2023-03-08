from PIL import Image
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Embedding, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from IPython import display as ipd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from collections import Counter
from tensorflow.keras import utils

from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns 

import dataset, text_process, модель, check_for_errors, квартиры
import gdown
import zipfile
import os
import random
import time 

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
        print(f'Эпоха {epoch+1}'.ljust(10)+ f'Время обучения: {t}c'.ljust(25) + f'Точность на обучающей выборке: {bcolors.OKBLUE}{round(logs["accuracy"]*100,2)}%{bcolors.ENDC}'.ljust(50) +f'Точность на проверочной выборке: {bcolors.OKBLUE}{round(logs["val_accuracy"]*100,2)}%{bcolors.ENDC}')
        self.cntepochs += 1

    def on_train_begin(self, logs):
        self.idxmax = 0
        self.accuracymax = 0
        self.cntepochs = 0

    def on_train_end(self, logs):
        ipd.clear_output(wait=True)
        for i in range(self.cntepochs):
            if i == self.idxmax:
                print('\33[102m' + f'Эпоха {i+1}'.ljust(10)+ f'Время обучения: {self.times[i]}c'.ljust(25) + f'Точность на обучающей выборке: {round(self.train_acc[i]*100,2)}%'.ljust(41) +f'Точность на проверочной выборке: {round(self.val_acc[i]*100,2)}%'+ '\033[0m')
            else:
                print(f'Эпоха {i+1}'.ljust(10)+ f'Время обучения: {self.times[i]}c'.ljust(25) + f'Точность на обучающей выборке: {bcolors.OKBLUE}{round(self.train_acc[i]*100,2)}%{bcolors.ENDC}'.ljust(50) +f'Точность на проверочной выборке: {bcolors.OKBLUE}{round(self.val_acc[i]*100,2)}%{bcolors.ENDC}' )
        self.plot_graph()

class TerraDataset:
    train_test_ratio = 0.1
    bases = {
        'Отзывы_тесла' : {
            'url': 'https://storage.yandexcloud.net/aiueducation/Intensive/tesla.zip',
            'info': 'Вы скачали базу с отзывами на автомобиль «ТЕСЛА». База содержит 3381 отзыв двух категорий: «Позитивные», «Негативные»',
            'dir_name': 'tesla',
            'task_type': 'text_classification',
            'classes': ['Негативные', 'Позитивные']
        },
        'Квартиры' : {
            'url': 'https://storage.yandexcloud.net/aiueducation/Intensive/moscow.csv',
            'info': 'Вы скачали информационную базу стоимости квартир в г. Москва. База содержит информацию о 62504 квартирах',
            'dir_name': 'moscow',
            'task_type': 'flat_regression',            
        },
        'Вакансии' : {
            'url': 'https://storage.yandexcloud.net/aiueducation/Intensive/hr.zip',
            'info': 'Вы скачали базу вакансий. База содержит информацию о 764 ползователях',
            'dir_name': 'вакансии',
            'task_type': 'hr_regression',
            'classes': ['Подходит', 'Не подходит']
        },
        'Симптомы' : {
            'url': 'https://storage.yandexcloud.net/aiueducation/Intensive/symptoms.zip',
            'info': "Вы скачали базу симптомов заболеваний. База содержит 10 категорий: ['Колит', 'Гепатит', 'Гастрит', 'Холицестит', 'Дуоденит', 'Энтерит', 'Язва', 'Эзофагит', 'Аппендицит', 'Панкреатит']",
            'dir_name': 'симптомы_заболеваний',
            'task_type': 'text_classification',
            'classes': ['Колит', 'Гепатит', 'Гастрит', 'Холицестит', 'Дуоденит', 'Энтерит', 'Язва', 'Эзофагит', 'Аппендицит', 'Панкреатит']
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
        print(f'{bcolors.OKBLUE}Ифно:{bcolors.ENDC}')
        print(f'    {self.base["info"]}')
        return self.base['task_type']

    def samples(self):
        '''
        Функция визуализации примеров
        '''
        
        # Визуализация датасета изображений для задачи классификации
        if self.base['task_type'] == 'text_classification':
            print(f'{bcolors.BOLD}Примеры:{bcolors.ENDC}')
            print()
            # Получение списка классов (названия папок в директории)
            self.classes = sorted(os.listdir(self.base['dir_name']))

            # Вывод примеров
            for i, class_ in enumerate(self.classes):
                print(f'{bcolors.BOLD}{bcolors.OKBLUE}{self.base["classes"][i]}:{bcolors.ENDC}')                
                with open(os.path.join(self.base['dir_name'], class_), 'r') as f:
                  texts = f.read()
                texts = texts.split('\n')
                samples = random.choices(texts, k=5)
                # Выбор случайного текста
                for j in samples:
                    print(f'   {j.lstrip()}')
                print()

        if self.base['task_type'] == 'flat_regression':
              dataset.показать_примеры(путь='квартиры', количество = 5)
        if self.base['task_type'] == 'hr_regression':
              dataset.показать_примеры(путь = 'Вакансии')
    
    def dataset_info(self):
        '''
        Функция отображает информацию по текстовому датасету
        '''
        if self.base['task_type'] == 'text_classification':
            # Получение списка классов (названия папок в директории)
            self.classes = sorted(os.listdir(self.base['dir_name']))
            # Вывод примеров
            print(f'{bcolors.BOLD}Информация по датасету:{bcolors.ENDC}')
            print()             
            for i, class_ in enumerate(self.classes):
                print(f'Класс {bcolors.OKBLUE}«{self.base["classes"][i]}»:{bcolors.ENDC}') 
                with open(os.path.join(self.base['dir_name'], class_), 'r') as f:
                  texts = f.read()
                texts = texts.split('\n')
                print(f'  Количество примеров: {len(texts)}')
                tuple_length = [len(x.split(' ')) for x in texts]                
                print(f'  Макисмальная длина отзыва (слов): {np.max(tuple_length)}')
                print(f'  Минимальная длина отзыва (слов): {np.min(tuple_length)}')
                print(f'  Средняя длина отзыва (слов): {int(np.mean(tuple_length))}')
                print()

    def create_sets(self, *params):
        if self.base['task_type'] == 'text_classification':
            kwargs =  {'путь':self.base['dir_name'], 'MWC':eval(params[0]),'xLen':eval(params[1]),'step':eval(params[2])}
            self.sets = dataset.создать_выборки(**kwargs)
            # Вывод финальной информации
                
            print()
            print(f'Размер созданных выборок для Embedding:')
            print(f'  Обучающая выборка: {self.sets[0][0][0].shape}')
            print(f'  Проверочная выборка: {self.sets[1][0][0].shape}')
            print()
            print(f'Размер созданных выборок для BagOfWords:')
            print(f'  Обучающая выборка: {self.sets[0][0][1].shape}')
            print(f'  Проверочная выборка: {self.sets[1][0][1].shape}')
            print()
            print(f'Размер созданных выборок для BagOfWords + PyMorphy:')
            print(f'  Обучающая выборка: {self.sets[0][0][2].shape}')
            print(f'  Проверочная выборка: {self.sets[1][0][2].shape}')
            print()
            print(f'Метки:')
            print(f'  Метки обучающей выборки: {self.sets[0][1].shape}')
            print(f'  Метки проверочной выборки: {self.sets[1][1].shape}')
            print()
            print(f'Распределение по классам:')
            f, ax =plt.subplots(1,2, figsize=(16, 5))            
            ax[0].bar(np.array(self.base['classes'])[list(Counter(list(np.argmax(self.sets[0][1], axis=1))).keys())], Counter(list(np.argmax(self.sets[0][1], axis=1))).values())
            ax[0].set_title('Обучающая выборка')
            ax[1].bar(np.array(self.base['classes'])[list(Counter(list(np.argmax(self.sets[1][1], axis=1))).keys())], Counter(list(np.argmax(self.sets[1][1], axis=1))).values(), color='g')
            ax[1].set_title('Проверочная выборка')
            plt.show()  
        if self.base['task_type'] == 'flat_regression':          
            self.sets = dataset.создать_выборки(путь='квартиры')
            print()
            print(f'Размер созданных выборок:')
            print('Ветвь 1')
            print(f'  Обучающая выборка: {self.sets[0][0][0].shape}')
            print(f'  Метки обучающей выборки: {self.sets[0][1].shape}')
            print(f'  Проверочная выборка: {self.sets[1][0][0].shape}')
            print(f'  Метки проверочной выборки: {self.sets[1][1].shape}')
            print('Ветвь 2')
            print(f'  Обучающая выборка: {self.sets[0][0][1].shape}')
            print(f'  Метки обучающей выборки: {self.sets[0][1].shape}')
            print(f'  Проверочная выборка: {self.sets[1][0][1].shape}')
            print(f'  Метки проверочной выборки: {self.sets[1][1].shape}')
            
            print()
        if self.base['task_type'] == 'hr_regression':
            self.sets = dataset.создать_выборки(путь='вакансии')
            print()
            print(f'Размер созданных выборок:')
            print(f'  Обучающая выборка: {self.sets[0][0].shape}')
            print(f'  Метки обучающей выборки: {self.sets[0][1].shape}')
            print(f'  Проверочная выборка: {self.sets[1][0].shape}')
            print(f'  Метки проверочной выборки: {self.sets[1][1].shape}')
            print(f'Распределение по классам:')
            f, ax =plt.subplots(1,2, figsize=(16, 5))            
            ax[0].bar(np.array(self.base['classes'])[list(Counter(list(np.argmax(self.sets[0][1], axis=1))).keys())], Counter(list(np.argmax(self.sets[0][1], axis=1))).values())
            ax[0].set_title('Обучающая выборка')
            ax[1].bar(np.array(self.base['classes'])[list(Counter(list(np.argmax(self.sets[1][1], axis=1))).keys())], Counter(list(np.argmax(self.sets[1][1], axis=1))).values(), color='g')
            ax[1].set_title('Проверочная выборка')
            plt.show()
            print()
    
class TerraModel:    
    def __init__(self, task_type, trds, type_trds):
        self.model = None
        self.task_type = task_type
        self.trds = trds
        self.type_trds = type_trds

    def create_model_combine(self, *branches):
        if self.trds.base['task_type'] == 'text_classification':
            self.model = модель.создать_составную_сеть(self.trds.sets[0][0], self.trds.sets[0][1], *branches)
        if self.trds.base['task_type'] == 'flat_regression':
            self.model = модель.создать_составную_сеть_квартиры(self.trds.sets[0][0], *branches)
    
    def create_model(self, layers, **kwargs):
      if 'model_type' in kwargs:
          if kwargs['model_type'] == 'BagOfWords':
              self.model = модель.создать_сеть(layers, self.trds.sets[0][0][1].shape[1:], параметры_модели=None, задача='классификация вакансий')
              self.type_trds = 'BagOfWords'
          elif kwargs['model_type'] == 'BagOfWords+PyMorphy':
              self.model = модель.создать_сеть(layers, self.trds.sets[0][0][1].shape[1:], параметры_модели=None, задача='классификация вакансий')
              self.type_trds = 'BagOfWords+PyMorphy'
          if kwargs['model_type'] == 'Embedding':
              self.model = модель.создать_сеть(layers, self.trds.sets[0][0][1].shape[1:], параметры_модели=None, задача='классификация вакансий')
              self.type_trds = 'Embedding'
      else:
          self.model = модель.создать_сеть(layers, self.trds.sets[0][0].shape[1:], параметры_модели=None, задача='классификация вакансий')

    def train_model(self, epochs, use_callback = True, **kwargs):
      if self.trds.base['task_type'] == 'text_classification':
        if self.type_trds == 'BagOfWords':
          self.model.compile(loss='categorical_crossentropy', optimizer = Adam(0.0001), metrics=['accuracy'])
          модель.обучение_модели(self.model, self.trds.sets[0][0][1], self.trds.sets[0][1], self.trds.sets[1][0][1], self.trds.sets[1][1], 16, epochs, 0.2, **kwargs)
        elif self.type_trds == 'BagOfWords+PyMorphy':
          self.model.compile(loss='categorical_crossentropy', optimizer = Adam(0.0001), metrics=['accuracy'])
          модель.обучение_модели(self.model, self.trds.sets[0][0][2], self.trds.sets[0][1], self.trds.sets[1][0][2], self.trds.sets[1][1], 16, epochs, 0.2, **kwargs)
        elif self.type_trds == 'Embedding':
          self.model.compile(loss='categorical_crossentropy', optimizer = Adam(0.0001), metrics=['accuracy'])
          модель.обучение_модели(self.model, self.trds.sets[0][0][0], self.trds.sets[0][1], self.trds.sets[1][0][0], self.trds.sets[1][1], 16, epochs, 0.2, **kwargs)
        else:
          модель.обучение_модели(self.model, self.trds.sets[0][0], self.trds.sets[0][1], self.trds.sets[1][0], self.trds.sets[1][1], 16, epochs, 0.2, **kwargs)
            
      if self.trds.base['task_type'] == 'flat_regression':
        модель.обучение_модели_квартиры(self.model, self.trds.sets[0][0], self.trds.sets[0][1], self.trds.sets[1][0], self.trds.sets[1][1], 256, epochs, 0.2)
      if self.trds.base['task_type'] == 'hr_regression':            
          self.model.compile(loss='sparse_categorical_crossentropy', optimizer = Adam(0.0001), metrics=['accuracy'])
          accuracy_callback = AccuracyCallback()
          callbacks = []
          if use_callback:
              callbacks = [accuracy_callback]
          y_train = np.argmax(self.trds.sets[0][1], axis=1)
          y_test = np.argmax(self.trds.sets[1][1], axis=1)
          history = self.model.fit(self.trds.sets[0][0], y_train,
                        batch_size = self.trds.sets[0][0].shape[0]//25,
                        validation_data=(self.trds.sets[1][0], y_test),
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose = 0)
          return history
          
    def test_model(self, *params):
      if self.trds.base['task_type'] == 'text_classification':
        text_process.тест_модели_симп(self.model, eval(params[1]), eval(params[2]), params[0], self.trds.base['classes'], model_type=self.type_trds)
      if self.trds.base['task_type'] == 'flat_regression':
        квартиры.тест_модели(self.model, *params)
      if self.trds.base['task_type'] == 'hr_regression':
        print(f'{bcolors.BOLD}Тестирование модели на случайном примере тестовой выборки: {bcolors.ENDC}')
        print()
        модель.тест_модели_вакансии(self.model, self.trds.sets[1][0], self.trds.sets[1][1])


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

    def dataset_info(self):
        self.trds.dataset_info()

    def create_sets(self, *params):
        self.trds.create_sets(*params)

    def create_model(self, layers, **kwargs):
        print(f'{bcolors.BOLD}Создание модели нейронной сети{bcolors.ENDC}', end=' ')
        self.trmodel = TerraModel(self.task_type, self.trds, 'hr_regression')
        self.trmodel.create_model(layers, **kwargs)
        print(f'{bcolors.OKGREEN}Ok{bcolors.ENDC}')

    def create_model_combine(self, *branches):
        print(f'{bcolors.BOLD}Создание комбинированной модели нейронной сети{bcolors.ENDC}', end=' ')
        self.trmodel = TerraModel(self.task_type, self.trds, 'Combined')
        self.trmodel.create_model_combine(*[b+'-linear' for b in branches])
        print(f'{bcolors.OKGREEN}Ok{bcolors.ENDC}')

    def train_model(self, epochs):
        self.trmodel.train_model(epochs)

    def test_model(self, *data):
        self.trmodel.test_model(*data)

    def train_model_average(self, layers, cnt=10):
        if self.task_type == 'hr_regression':
          print(f'{bcolors.BOLD}Определение среднего показателя точности модели на {cnt} запусках{bcolors.ENDC}')
          print()
          average_accuracy = []
          average_val_accuracy = []
          times=[]
          for i in range(cnt):
              start_time = time.time()
              self.trmodel.create_model(layers)
              history = self.trmodel.train_model(100, False).history
              average_accuracy.append(np.max(history['accuracy']))
              average_val_accuracy.append(np.max(history['val_accuracy']))
              t = round(time.time() - start_time, 1)
              times.append(t)
              print(f'Запуск {i+1}'.ljust(10)+ f'Время обучения: {t}c'.ljust(25) + f'Точность на обучающей выборке: {bcolors.OKBLUE}{round(average_accuracy[-1]*100,2)}%{bcolors.ENDC}'.ljust(50) +f'Точность на проверочной выборке: {bcolors.OKBLUE}{round(average_val_accuracy[-1]*100,2)}%{bcolors.ENDC}')
          
          ipd.clear_output(wait=True)
          print(f'{bcolors.BOLD}Определение среднего показателя точности модели на {cnt} запусках{bcolors.ENDC}')
          print()
          argmax_idx = np.argmax(average_val_accuracy)
          for i in range(cnt):
              if i == argmax_idx:
                  print('\33[102m' + f'Запуск {i+1}'.ljust(10)+ f'Время обучения: {times[i]}c'.ljust(25) + f'Точность на обучающей выборке: {round(average_accuracy[i]*100,2)}%'.ljust(41) +f'Точность на проверочной выборке: {round(average_val_accuracy[i]*100,2)}%'+ '\033[0m')
              else:
                  print(f'Запуск {i+1}'.ljust(10)+ f'Время обучения: {times[i]}c'.ljust(25) + f'Точность на обучающей выборке: {bcolors.OKBLUE}{round(average_accuracy[i]*100,2)}%{bcolors.ENDC}'.ljust(50) +f'Точность на проверочной выборке: {bcolors.OKBLUE}{round(average_val_accuracy[i]*100,2)}%{bcolors.ENDC}' )
          plt.figure(figsize=(14, 6))
          plt.bar(np.arange(cnt)-0.15, np.array(average_accuracy)*100, color='gold', width = 0.4, align='center', label='Обучающая выборка')
          plt.bar(np.arange(cnt)+0.15, np.array(average_val_accuracy)*100, color='lime', width = 0.4, align='center', label='Проверочная выборка')
          plt.legend()
          plt.show()
          print()
          print(f'{bcolors.BOLD}Средняя точность на обучающей выборке: {bcolors.ENDC}{round(np.mean(average_accuracy[i])*100,2)}%')
          print(f'{bcolors.BOLD}Средняя точность на проверочной выборке: {bcolors.ENDC}{round(np.mean(average_val_accuracy[i])*100,2)}%')