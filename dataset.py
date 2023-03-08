import text_process, квартиры, модель
import subprocess
from subprocess import STDOUT, check_call
import os, time, random, re,pymorphy2, shutil
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from IPython import display
from tensorflow.keras.utils import to_categorical, plot_model # Полкючаем методы .to_categorical() и .plot_model()
from tensorflow.keras import datasets # Подключаем набор датасетов
import importlib.util, sys, gdown
from tqdm.notebook import tqdm_notebook as tqdm_
from PIL import Image
import pandas as pd
import requests
import ast
import json

import zipfile
from sklearn.model_selection import train_test_split

#sda#sda
###
###                 ЗАГРУЗКА ДАННЫХ
###
def загрузить_базу(база = ''):
  справка = True
  база_1 = база
  база = база.lower()
  print('Загрузка данных')
  print('Это может занять несколько минут...')  
  if база == 'mnist':
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    if справка:
      print('Вы скачали базу рукописных цифр MNIST. \nБаза состоит из двух наборов данных: обучающего (60тыс изображений) и тестовго (10тыс изображений).')
      print('Размер каждого изображения: 28х28 пикселей')
  
  elif база == 'авто-2':
    загрузить_базу_АВТОМОБИЛИ_2()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Вы скачали базу с изображениями марок автомобилей. \nБаза состоит из двух марок: Феррари и Мерседес')
      print('Количество изображений в базе: 2249') 

  elif база == 'авто-3':
    загрузить_базу_АВТОМОБИЛИ_3()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Вы скачали базу с изображениями марок автомобилей. \nБаза состоит из трех марок: Феррари, Мерседес и Рено')
      print('Количество изображений в базе: 3429')

  elif база == 'молочная_продукция':
    загрузить_базу_МОЛОКО()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Вы скачали базу с изображениями бутылок молока')

  elif база == 'автобусы':
    загрузить_базу_АВТОБУСЫ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Вы скачали базу с изображениями входящих и выходящих пассажиров в автобус')
  
  elif база == 'tesla':
    загрузить_базу_TESLA()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Вы скачали базу отзывов владельцев автомобилями Tesla')

  elif база == 'обнаружение_возгораний':
    загрузить_базу_ОГОНЬ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Вы скачали базу для классификации огня.')

  elif база == 'майонез':
    загрузить_базу_МАЙОНЕЗ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Вы скачали базу для классификации майонеза.')

  elif база == 'трекер':
    загрузить_базу_ТРЕКЕР()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Вы скачали базу для трекинга пассажиров в автобусе.')

  elif база == 'вакансии':
    загрузить_базу_ВАКАНСИИ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Вы скачали базу вакансий.')

  elif база == 'самолеты':
    загрузить_базу_САМОЛЕТЫ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Вы скачали базу самолетов. \nБаза состоит из оригинальных изображений и соовтетствующих им размеченных сегментированных изображений.')
      print('Количество изображений в базе: 981')    

  elif база == 'самолеты_макс':
    загрузить_базу_САМОЛЕТЫ_макс()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Вы скачали базу самолетов. \nБаза состоит из оригинальных изображений и соовтетствующих им размеченных сегментированных изображений.')
      print('Количество изображений в базе: 981') 
  
  elif база == 'люди':
    загрузить_базу_Люди()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Вы скачали базу на которой изображены люди. \nБаза состоит из оригинальных изображений и соответствующих им размеченных сегментированных изображений.')


  elif база == 'кожные_заболевания':
    загрузить_базу_БОЛЕЗНИ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Вы скачали базу кожных заболеваний. \nБаза состоит из оригинальных изображений и соовтетствующих им размеченных сегментированных изображений.')
      print('Количество категорий заболеваний: 10 (Акне, Витилиго, Герпес, Дерматит, Лишай, Невус, Псориаз, Сыпь, Хлоазма, Экзема)')    
      print('Количество изображений в базе: 981')    
  
  elif база == 'повышение_размерности':
    загрузить_базу_HR()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Вы скачали базу изображений для задачи повышения размерности')
      print('База содержит изображения высокого качества и соответствующие им изображения низкого качества')

  elif база == 'обнаружение_людей':
    загрузить_базу_ЛЮДЕЙ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Загружена размеченная база изображений для обнаружения людей')      
    print()
    print('ВНИМАНИЕ!!! Были установлены дополнительные библиотеки. Необходимо перезапустить среду для продолжения работы')
    print('Выберите пункт меню Runtime/Restart runtime и нажмите «Yes»')
    print('После этого сделайте повторный запуск ячейки: import terra_ai')
  
  elif база == 'симптомы_заболеваний':
    #Загрузка базы
    загрузить_базу_ЗАБОЛЕВАНИЯ()
    #display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Загружена база симптомов заболеваний') 
    print()
    
  elif база == 'писатели':
    загрузить_базу_ПИСАТЕЛИ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Загружена база писателей')      
    print()

  elif база == 'диалоги':
    загрузить_базу_ДИАЛОГИ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Загружена база диалогов')
      print('Количество пар вопрос-ответ: 50 тысяч')
    print()
  elif база == 'договоры':
    загрузить_базу_ДОГОВОРА()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Загружена база договоров')
      print('База размечена по 6 категориям: Условия - Запреты - Стоимость - Деньги - Сроки - Неустойка')
# s6 Всё про адреса и геолокации
    print()

  elif база == 'квартиры':
    загрузить_базу_КВАРТИРЫ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Загружена база квартир')
    print()

  elif база == 'губы':
    загрузить_базу_ГУБЫ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Загружена база сегментации женских губ')
    print()

  elif база == 'трафик':
    загрузить_базу_ТРАФИК()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Загружена база трафика компании')
    print()

  elif база == 'умный_дом':
    загрузить_базу_УМНЫЙ_ДОМ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Загружена база голосовых команд для умного дома')
    print()

  elif база == 'трейдинг':
    загрузить_базу_ТРЕЙДИНГ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    print('url:', url)
    if справка:
      print('Загружены базы акций трех вариантов: полиметаллы, газпром и яндекс')
    print()
  else:
    display.clear_output(wait=True)
    assert False, f"Указанная база '{база_1}' не найдена. Возможно вы имели ввиду '{check_for_errors.check_word(база, 'dataset')}'.\n"

def загрузить_базу_конструктора():
  url = 'https://storage.googleapis.com/aiu_bucket/UI.zip' # Указываем URL-файла
  output = 'UI.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL

def загрузить_базу_ТРАФИК():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/traff.csv'
  output = 'traff.csv' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True)

def загрузить_базу_ТРЕЙДИНГ():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/shares.zip'
  output = 'shares.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) 
  if not os.path.exists('трейдинг/'): 
      распаковать_архив(откуда = "shares.zip",куда = "")
      os.rename('shares','трейдинг')
  

def загрузить_базу_АВТОМОБИЛИ_2():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/car_2.zip'
  output = 'car.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  распаковать_архив(
      откуда = "car.zip",
      куда = "авто-2"
  )

def загрузить_базу_АВТОМОБИЛИ_3():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/car.zip'
  output = 'car.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  распаковать_архив(
      откуда = "car.zip",
      куда = "авто-3"
  )

def загрузить_базу_МОЛОКО():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/milk.zip'
  output = 'milk.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  распаковать_архив(
      откуда = "milk.zip",
      куда = "молочная_продукция"

  )
def загрузить_базу_АВТОБУСЫ():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/bus.zip'
  output = 'bus.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  распаковать_архив(
      откуда = "bus.zip",
      куда = "автобусы"
  )

def загрузить_базу_ОГОНЬ():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/fire.zip'
  output = 'fire.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  if not os.path.exists('болезни'): 
      распаковать_архив(откуда = "fire.zip",куда = "")
      os.rename('Обнаружение_возгораний','обнаружение_возгораний')

def загрузить_базу_МАЙОНЕЗ():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/mayonnaise.zip'
  output = 'mayonnaise.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  распаковать_архив(
      откуда = "mayonnaise.zip",
      куда = "майонез"
  )  

def загрузить_базу_ТРЕКЕР():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/Heads.zip'
  output = 'Heads.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  распаковать_архив(
      откуда = "Heads.zip",
      куда = "трекер"
  )   
def загрузить_базу_ВАКАНСИИ():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/HR.zip'
  output = 'HR.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  распаковать_архив(
      откуда = "HR.zip",
      куда = "вакансии")

def загрузить_базу_УМНЫЙ_ДОМ():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/cHome.zip'
  output = 'cHome.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  распаковать_архив(
      откуда = "cHome.zip",
      куда = "умный_дом"
  )

def загрузить_базу_КВАРТИРЫ():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/moscow.csv'
  output = 'moscow.csv' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL

def загрузить_базу_ДИАЛОГИ():
  global url
  #обнаружение.выполнить_команду('mkdir content')
  url = 'https://storage.googleapis.com/terra_ai/DataSets/dialog.txt'
  output = 'dialog.txt' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  

def загрузить_базу_ЗАБОЛЕВАНИЯ():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/symptoms.zip'
  output = 'symptoms.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL

  # Скачиваем и распаковываем архив
  if not os.path.exists('болезни'): 
    распаковать_архив(откуда = "symptoms.zip",куда = "")
    os.rename('Болезни','симптомы_заболеваний')

def загрузить_базу_TESLA(): 
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/tesla.zip'
  output = 'tesla.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL

  # Скачиваем и распаковываем архив
  распаковать_архив(
      откуда = "tesla.zip",
      куда = "tesla"
  )

def загрузить_базу_ПИСАТЕЛИ():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/writers.zip'
  output = 'writers.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL

  # Скачиваем и распаковываем архив
  распаковать_архив(
      откуда = "writers.zip",
      куда = "content/"
  )
  
def загрузить_базу_ЛЮДЕЙ():
  global url
  url = 'https://github.com/ultralytics/yolov5/archive/master.zip'
  output = 'tmp.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL

  распаковать_архив(
      откуда = "tmp.zip",
      куда = "/content"
  )
  # Скачиваем и распаковываем архив
  url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017val.zip'
  output = 'tmp.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL

  распаковать_архив(
      откуда = "tmp.zip",
      куда = "/content"
  )

  url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip'
  output = 'tmp.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL

  распаковать_архив(
      откуда = "tmp.zip",
      куда = "/content"
  )
  обнаружение.выполнить_команду('echo y|pip uninstall albumentations > /dev/null')
  обнаружение.выполнить_команду('pip install -q --no-input -U git+https://github.com/albumentations-team/albumentations > /dev/null')
  обнаружение.выполнить_команду('pip install -q -U PyYAML')
  
  
def загрузить_базу_ДОГОВОРА():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/docs.zip'
  output = 'договоры.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL

  # Скачиваем и распаковываем архив
  распаковать_архив(
      откуда = "договоры.zip",
      куда = "договоры"
      )
 
  url = 'https://storage.googleapis.com/terra_ai/DataSets/test_doc.txt'
  output = 'договоры/test_doc.txt' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL


def загрузить_договоры():

  def readText(fileName):
    f = open(fileName, 'r', encoding='utf-8') #Открываем наш файл для чтения и считываем из него данные 
    text = f.read() #Записываем прочитанный текст в переменную 
    delSymbols = ['\n', "\t", "\ufeff", ".", "_", "-", ",", "!", "?", "–", "(", ")", "«", "»", "№", ";"]
    for dS in delSymbols: # Каждый символ в списке символов для удаления
      text = text.replace(dS, " ") # Удаляем, заменяя на пробел
    text = re.sub("[.]", " ", text) 
    text = re.sub(":", " ", text)
    text = re.sub("<", " <", text)
    text = re.sub(">", "> ", text)
    text = ' '.join(text.split()) 
    return text # Возвращаем тексты
  def text2Words(text):
    morph = pymorphy2.MorphAnalyzer() # Создаем экземпляр класса MorphAnalyzer
    words = text.split(' ') # Разделяем текст на пробелы
    words = [morph.parse(word)[0].normal_form for word in words] #Переводим каждое слово в нормалную форму  
    return words # Возвращаем слова   
  directory = 'договоры/Договора432/' # Путь к папке с договорами
  agreements = [] # Список, в который запишем все наши договоры
  for filename in os.listdir(directory): # Проходим по всем файлам в директории договоров
    try:    
        txt = readText(directory + filename) # Читаем текст договора
        if txt != '': # Если текст не пустой
          agreements.append(readText(directory + filename)) # Преобразуем файл в одну строку и добавляем в agreements
    except:
        continue
  words = [] # Здесь будут храниться все договора в виде списка слов
  curTime = time.time() # Засечем текущее время
  for i in tqdm_(range(len(agreements)), desc='Обработка догововров', ncols=1000): # Проходимся по всем договорам
    words.append(text2Words(agreements[i])) # Преобразуем очередной договор в список слов и добавляем в words
  wordsToTest = words[-10:] # Возьмем 10 текстов для финальной проверки обученной нейронной сети 
  words = words[:-10] # Для обученающей и проверочной выборок возьмем все тексты, кроме последних 10
  display.clear_output(wait=True)
  return (agreements, words, wordsToTest)  #, agreements

def загрузить_базу_САМОЛЕТЫ():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/airplane.zip'
  output = 'самолеты.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  url = 'https://storage.googleapis.com/terra_ai/DataSets/segment.zip' # Указываем URL-файла
  output = 'сегменты.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  
  # Скачиваем и распаковываем архив
  if not os.path.exists('самолеты'): 
      распаковать_архив(откуда = "самолеты.zip",куда = "самолеты/")
      os.rename('самолеты/Самолеты','самолеты/оригинал')
      
      распаковать_архив(откуда = "сегменты.zip",куда = "самолеты/")
      os.rename('самолеты/Сегменты','самолеты/сегментация')

  display.clear_output(wait=True)  
  # Обрабатываем скаченные изображения

def загрузить_базу_САМОЛЕТЫ_макс():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/airplane_max.zip' # Указываем URL-файла
  output = 'airplane_max.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL


      
      
  url = 'https://storage.googleapis.com/terra_ai/DataSets/segment_max.zip' # Указываем URL-файла
  output = 'segment_max.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  if not os.path.exists('самолеты_макс/'): 
  
      распаковать_архив(откуда = "airplane_max.zip",куда = "самолеты_макс/")
      os.rename('самолеты_макс/airplane_max','самолеты_макс/оригинал')
      
      распаковать_архив(откуда = "segment_max.zip",куда = "самолеты_макс/")
      os.rename('самолеты_макс/segment_max','самолеты_макс/сегментация')
  display.clear_output(wait=True)  
  # Обрабатываем скаченные изображения

def загрузить_базу_Люди():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/original_people.zip' # Указываем URL-файла
  output = 'original_people.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  # Скачиваем и распаковываем архив
  распаковать_архив(
      откуда = "original_people.zip",
      куда = "/content"
  )
  url = 'https://storage.googleapis.com/terra_ai/DataSets/segment_people.zip' # Указываем URL-файла
  output = 'segment_people.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  # Скачиваем и распаковываем архив
  распаковать_архив(
      откуда = "segment_people.zip",
      куда = "/content"
  )

  if not os.path.exists('люди/'): 
  
      распаковать_архив(откуда = "original_people.zip",куда = "люди/")
      os.rename('люди/original','люди/оригинал')
      
      распаковать_архив(откуда = "segment_people.zip",куда = "люди/")
      os.rename('люди/segment','люди/сегментация')
  display.clear_output(wait=True)  
  # Обрабатываем скаченные изображения
  
def загрузить_базу_ГУБЫ():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/lips.zip' # Указываем URL-файла
  output = 'lips.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  # Скачиваем и распаковываем архив
  распаковать_архив(
      откуда = "lips.zip",
      куда = "/content"
  )
  if not os.path.exists('губы'): 
      распаковать_архив(откуда = "lips.zip",куда = "")
      os.rename('Губы','губы')
  display.clear_output(wait=True)  

def загрузить_базу_БОЛЕЗНИ():
  global url
  url = 'https://storage.googleapis.com/terra_ai/DataSets/origin.zip'
  output = 'diseases.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  # Скачиваем и распаковываем архив
  распаковать_архив(
      откуда = "diseases.zip",
      куда = "diseases"
  )
  url = 'https://storage.googleapis.com/terra_ai/DataSets/segmentation.zip'
  output = 'segm.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  # Скачиваем и распаковываем архив
  распаковать_архив(
      откуда = "segm.zip",
      куда = "diseases"
  )

def загрузить_базу_HR():
  global url
  # Скачиваем и распаковываем архив в колаб по ссылке
  url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip'          
  output = 'DIV2K_valid_LR_bicubic_X4.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  распаковать_архив(
      откуда = "DIV2K_valid_LR_bicubic_X4.zip",
      куда = "повышение_размерности"
  )
  url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'
  output = 'DIV2K_valid_HR.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  распаковать_архив(
      откуда = "DIV2K_valid_HR.zip",
      куда = "повышение_размерности"
  )
  os.system("cp -r повышение_размерности/DIV2K_valid_LR_bicubic/X4 повышение_размерности")
  shutil.rmtree('/повышение_размерности/DIV2K_valid_LR_bicubic')
  os.rename('повышение_размерности/X4', 'повышение_размерности/DIV2K_valid_LR_bicubic')

def формирование_повышение_размерности():
  pathHR = 'повышение_размерности/DIV2K_valid_HR/'
  pathLR = 'повышение_размерности/DIV2K_valid_LR_bicubic/'
  imagesList = [os.listdir(pathHR), os.listdir(pathLR)] # Список из корневых каталогов изображений, их два - hr и lr
  train_hr = []
  train_lr = []

  # В цикле проходимся по каталогу с hr изображениями
  for image in sorted(imagesList[0]):
      img = Image.open(pathHR + image)
      train_hr.append(img) # Добавляем все изображения в список

  # В цикле проходимся по каталогу с lr изображениями
  for image in sorted(imagesList[1]):
      img = Image.open(pathLR + image)
      train_lr.append(img) # Добавляем все изображения в список
  return train_hr, train_lr

###
###                    ДЕМОНСТРАЦИЯ ПРИМЕРОВ
###

def показать_примеры(**kwargs):
  kwargs['путь'] =  kwargs['путь'].replace('/','')
  if 'вопросы' in kwargs:  
    count = 1
    if 'количество' in kwargs:
        count = kwargs['количество']
    questions = kwargs['вопросы']
    answers = kwargs['ответы']
    for i in range(count):
        print()
        idx = np.random.randint(len(questions)-1)
        print('Вопрос:', questions[idx])
        print('Ответ:', answers[idx])

  if 'путь' in kwargs:
    kwargs['путь'] = kwargs['путь'].lower() 
    new_data = ['молочная_продукция', 'автобусы','обнаружение_возгораний']
    
    if kwargs['путь'] == 'диалоги':
      assert os.path.exists('' + 'dialog.txt'), f'Загрузите базу для \'{kwargs["путь"]}\' для корректной работы.'  
      f = open('dialog.txt', 'r', encoding='utf-8')
      text= f.read()
      text = text.replace('"','')
      text = text.split('\n')
      question = text[::3]
      answers = text[1::3]
      for i in range(10,0,-1):
        print(question[-i], answers[-i])
    
    elif kwargs['путь'] == 'обнаружение_возгораний':
      assert os.path.exists('' + "обнаружение_возгораний"), f'Загрузите базу  для \'обнаружение_возгораний\' для корректной работы.' 
      path = '/content'
      size=(512,512)
      fire_idx = [20,30,70,80,90,
                  100,130,200,300,500,
                  610,18,19,21,26]
      not_fire_idx = [0,30,80,110,140,
                      320,390,400,450,500,
                      550,840,850,960,1010]

      fig = plt.figure(figsize=(16,8))

      fire_img = sorted(os.listdir(os.path.join(path, 'обнаружение_возгораний/картинки_огня')))
      for i, idx in enumerate(np.random.choice(fire_idx, 4, replace=False)):
          img_path = os.path.join(path, 'обнаружение_возгораний/картинки_огня', fire_img[idx])
          img = image.load_img(img_path, target_size=size)
          img = np.array(img)
          
          ax = fig.add_subplot(2, 4, i+1, xticks=[], yticks=[])
          ax.set_title('Огонь')
          plt.imshow(img)

      not_fire_img = sorted(os.listdir(os.path.join(path, 'обнаружение_возгораний/случайные_картинки')))
      for i, idx in enumerate(np.random.choice(not_fire_idx, 4, replace=False)):
          img_path = os.path.join(path, 'обнаружение_возгораний/случайные_картинки', not_fire_img[idx])
          img = image.load_img(img_path, target_size=size)
          img = np.array(img)
          
          ax = fig.add_subplot(2, 4, i+5, xticks=[], yticks=[])
          ax.set_title('Нет огня')
          plt.imshow(img)
    
    elif kwargs['путь'] == 'трейдинг':
      kwargs.pop('путь')
      traide.show_data(**kwargs)
    elif kwargs['путь'] == 'умный_дом':
       if 'файл' in kwargs:
         home.параметризация_аудио(kwargs['файл'])
       else:
         home.показать_примеры_голосовых_команд()
    
    elif kwargs['путь'] == 'молочная_продукция':
      assert os.path.exists('' + kwargs['путь']), f'Загрузите базу для \'{kwargs["путь"]}\' для корректной работы.'   
      size=(512,256)
      path = '' + kwargs['путь']
      fig = plt.figure(figsize=(10,10))

      classes = sorted(os.listdir(path))
      for i in range(len(classes)):
          images = sorted(os.listdir(os.path.join(path, classes[i])))
          img_path = os.path.join(path, classes[i], images[0])
          img = image.load_img(img_path, target_size=size)
          img = np.array(img)
          
          ax = fig.add_subplot(1, 3, i+1, xticks=[], yticks=[])
          plt.imshow(img)

    elif kwargs['путь'] == 'автобусы':
      assert os.path.exists('' + kwargs['путь']), f'Загрузите базу для \'{kwargs["путь"]}\' для корректной работы.'  
      path = '' + kwargs['путь']
      size=(256,150)
      entering_idx = [1,170,338,1200,1536,
                      1762,1830,1942,2312,2414,
                      2455,2712,3513,3657,3744,
                      3761,3792,3797,3826,3877,
                      3904,3945,3972,3993,4140]

      getting_off_idx = [184,445,653,1220,1241,
                        1283,1367,1388,1409,1453,
                        1487,1648,1906,2505,2412,
                        2292,1865]

      fig = plt.figure(figsize=(16,10))

      entering_img = sorted(os.listdir(os.path.join(path, 'entering')))
      for i, idx in enumerate(np.random.choice(entering_idx, 5, replace=False)):
          img_path = os.path.join(path, 'entering', entering_img[idx])
          img = image.load_img(img_path, target_size=size)
          img = np.array(img)
          
          ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
          ax.set_title('Входящий')
          plt.imshow(img)

      getting_off_img = sorted(os.listdir(os.path.join(path, 'getting_off')))
      for i, idx in enumerate(np.random.choice(getting_off_idx, 5, replace=False)):
          img_path = os.path.join(path, 'getting_off', getting_off_img[idx])
          img = image.load_img(img_path, target_size=size)
          img = np.array(img)
          
          ax = fig.add_subplot(2, 5, i+6, xticks=[], yticks=[])
          ax.set_title('Выходящий')
          plt.imshow(img)
    elif kwargs['путь'] == 'квартиры':
        if 'количество' in kwargs:
          квартиры.samples(kwargs['количество'])
        else:
          квартиры.samples(5)
    
    elif kwargs['путь'] == 'mnist':

        (_, _), (x_test, y_test) = datasets.mnist.load_data()
        real_img = [x_test[y_test==i][0] for i in range(10)]
        real_img_concat = np.concatenate([i for i in real_img], axis=1)
        plt.figure(figsize=(56, 2*len(real_img)), dpi=25)
        plt.imshow(real_img_concat, cmap='Greys_r')
        plt.grid(False)
        plt.axis('off')
        plt.show()

    elif kwargs['путь'] == 'симптомы_заболеваний':        
        path = f'{kwargs["путь"]}/'
        assert os.path.exists(path), f'Загрузите базу \'{kwargs["путь"]}\'.'
        text = []
        classes = []
        n = 0
        codecs_list = ['UTF-8', 'Windows-1251']

        for filename in sorted(os.listdir(path)): # Проходим по всем файлам в директории договоров
            n +=1
            for codec_s in codecs_list:
                try:
                    text.append(обработка_текста.readText(path+filename, codec_s)) # Преобразуем файл в одну строку и добавляем в agreements
                    classes.append(filename.replace(".txt", ""))
                    break
                except UnicodeDecodeError:
                    print('Не прочитался файл: ', path+currdir+'/'+filename, codec_s)
                else:
                    next 
        nClasses = len(classes) #определяем количество классов
        print('В данной базе содержатся симптомы следующих заболеваний:')
        print(classes)        
        n = np.random.randint(10)
        print()
        print('Пример симптомов случайного заболевания:')
        print('Заболевание: ', classes[n])
        print('Симптомы:')
        print('     *', text[n][:100]) # Пример первых 100 символов первого документа      
        return classes, nClasses

    elif kwargs['путь'] == 'tesla':        
        path = f'{kwargs["путь"]}/'
        assert os.path.exists(path), f'Загрузите базу \'{kwargs["путь"]}\'.'
        text = []
        classes = []
        n = 0
        codecs_list = ['UTF-8', 'Windows-1251']

        for filename in sorted(os.listdir(path)): # Проходим по всем файлам в директории договоров
            n +=1
            for codec_s in codecs_list:
                try:
                    text.append(обработка_текста.readText(path+filename, codec_s)) # Преобразуем файл в одну строку и добавляем в agreements
                    classes.append(filename.replace(".txt", ""))
                    break
                except UnicodeDecodeError:
                    print('Не прочитался файл: ', path+currdir+'/'+filename, codec_s)
                else:
                    next 
        nClasses = len(classes) #определяем количество классов
        print('В данной базе находятся положительные и негативные отзывы об автомобилях Tesla:')
        print(classes)        
        n = np.random.randint(2)
        print()
        print('Пример отзыва:')
        print('Тип отзыва: ', classes[n])
        print('Отзывы:')
        print('     *', text[n][:100]) # Пример первых 100 символов первого документа      
        return classes, nClasses

    elif kwargs['путь'] == 'писатели':        
        path = 'content/writers/'
        assert os.path.exists(path), f'Загрузите базу \'{kwargs["путь"]}\'.'
        text = []
        classes = []
        n = 0
        codecs_list = ['UTF-8', 'Windows-1251']

        for filename in os.listdir(path): # Проходим по всем файлам в директории договоров
            n +=1
            for codec_s in codecs_list:
                try:
                    text.append(обработка_текста.readText(path+filename, codec_s)) # Преобразуем файл в одну строку и добавляем в agreements
                    classes.append(filename.replace(".txt", ""))
                    break
                except UnicodeDecodeError:
                    print('Не прочитался файл: ', path+currdir+'/'+filename, codec_s)
                else:
                    next 
        nClasses = len(classes) #определяем количество классов
        print('В данной базе содержатся произведения следующих писателей:')
        print(classes)        
        n = np.random.randint(6)
        print()
        print('Пример текста случайного произведения и автора:')
        print('Автор: ', classes[n])
        print('Произведение и пример: ', text[n][:78], '\n', text[n][78:178], '\n', text[n][178:278], '\n',
                                         text[n][278:378], '\n', text[n][378:478], '\n', text[n][578:678], '\n', 
                                         text[n][778:878], '\n', text[n][978:1078]) # Пример первых 1000 символов первого документа      
        return classes, nClasses

    elif kwargs['путь'] == 'трекер':               
        path = 'трекер'
        kwargs["тип_изображений"] = kwargs["тип_изображений"].lower()
        assert os.path.exists(path), f'Загрузите базу \'{kwargs["путь"]}\'.'
        assert 'тип_изображений' in kwargs, f'Уточните тип изображений параметром \'тип_изображений\'.'
        list_of_humans = ['один человек', 'разные люди']
        
        assert kwargs['тип_изображений'] in list_of_humans, f'Тип изображения \'{kwargs["тип_изображений"]}\' не распознано.\
        Возможно вы имели ввиду \'{check_for_errors.check_word(kwargs["тип_изображений"],extra=list_of_humans)}\'.'
        
        size=(64,64)
        heads_idx = [9,39,100,130,
                    170,330,400,
                    450,500,630,680,700,
                    730,750,810,870,
                    960,1010,1040,1100,1140,
                    1210,1290,1370,1650]

        fig = plt.figure(figsize=(18,7))

        heads_img = sorted(os.listdir(os.path.join(path, 'images')))
        np.random.choice(heads_idx, 5, replace=False)
        for i, idx in enumerate(np.random.choice(heads_idx, 5, replace=False)):
            img_path = os.path.join(path, 'images', heads_img[idx])
            img = image.load_img(img_path, target_size=size)
            img = np.array(img)
            
            ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
            plt.imshow(img)

            if kwargs['тип_изображений'] == 'один человек':
                next_idx = idx + 3
            elif kwargs['тип_изображений'] == 'разные люди':
                next_idx = np.random.choice(heads_idx, 1)[0]
                while idx == next_idx:
                    next_idx = np.random.choice(heads_idx, 1)[0]
                
            img_path = os.path.join(path, 'images', heads_img[next_idx])
            img = image.load_img(img_path, target_size=size)
            img = np.array(img)

            ax = fig.add_subplot(2, 5, i+6, xticks=[], yticks=[])
            plt.imshow(img)

    elif kwargs['путь'] == 'обнаружение':
      обнаружение.показать_примеры()

    elif kwargs['путь'] == 'трафик':
      dataframe = pd.read_csv('traff.csv', header=None)
      dataframe.rename(columns={0: "Дата", 1: "Трафик"}, inplace=True)
      dataframe['Трафик'] = [float(i.replace(',','')) for i in dataframe['Трафик']]
      print(dataframe[:5])
      

    elif kwargs['путь'] == 'вакансии':
      path = 'вакансии'
      assert os.path.exists(path), f'Загрузите базу \'{kwargs["путь"]}\'.'
      num = 3
      data = pd.read_csv(path + '/data.csv')

      columns=['Пол', 'Возраст', 'Город', 'Готовность к переезду',
              'Готовность к командировкам', 'Гражданство', 'Разрешение на работу',
              'Знания языков', 'Образование', 'Дополнительное образование',
              'Зарплата', 'Время в пути до работы', 'Занятость', 
              'График', 'Опыт работы (мес)', 'Обязанности на пред.работе']

      idx = np.random.randint(0, data.shape[0], num)
      for i in idx:
          print('Пример резюме:')
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
          print('---------------------------------------------------------------')
          result = data['Этап сделки'][i]
          print('%-28s' % 'Подходит ли кандидат: ', result)
          print('---------------------------------------------------------------')
          print()
    
    elif kwargs['путь'] not in new_data:
      if not 'размер' in kwargs:
          kwargs['размер'] = (108, 192)
      func_true_false(kwargs['путь'], type_check='dataset')
      if kwargs['путь']=='майонез':
          kwargs['размер'] = (192, 108)
      
      if kwargs['размер'] == 'молочная_продукция':
          kwargs['размер'] = (512,256)
      kwargs['путь'] = ''+kwargs['путь']
      count_classes = sorted(os.listdir(kwargs['путь']))
      count = 5    
      fig, axs = plt.subplots(len(count_classes), count, figsize=(25, 15)) #Создаем полотно из 3 графиков
      for idx,num in enumerate(count_classes): #Проходим по всем классам
        car_path = os.path.join(kwargs['путь'], num)
        for i_1, curr_img in enumerate(sorted(os.listdir(car_path))[:count]):
            img_path = os.path.join(car_path,curr_img) #Выбираем случайное фото для отображения
            axs[idx, i_1].imshow(image.load_img(img_path, target_size=kwargs['размер'])) #Отображение фотографии
            axs[idx, i_1].axis('off') # отключаем оси
      plt.show() #Показываем изображения
    else:
        assert False, f'Невозможно проверить указанный путь: \'{kwargs["путь"]}\'.\
        Возможно вы имели ввиду \'{check_for_errors.check_word(kwargs["путь"], "show_paths")}\'.'  
        
  elif ('изображения' in kwargs):
    kwargs['изображения'] = kwargs['изображения'].lower()
    if 'метки' in kwargs:
      count = kwargs['метки'].max() # Задачем количество примеров
    elif 'количество' in kwargs:
      count = kwargs['количество'] # Задачем количество примеров
    else:
      count = 5
    f, axs = plt.subplots(1,count,figsize=(22,5)) # Создаем полотно для визуализации
    idx = np.random.choice(kwargs['изображения'].shape[0], count) # Получаем 5 случайных значений из диапазона от 0 до 60000 (x_train_org.shape[0])
    for i in range(count):
      axs[i].imshow(kwargs['изображения'][idx[i]], cmap='gray') # Выводим изображение из обучающей выборки в черно-белых тонах
      axs[i].axis('off')
    plt.show()
  elif ('оригиналы' in kwargs):
    kwargs['оригиналы'] = kwargs['оригиналы'].lower()
    показать_примеры_сегментации(**kwargs)
  elif ('изображения_низкого_качества' in kwargs):
    kwargs['изображения_низкого_качества'] = kwargs['изображения_низкого_качества'].lower()
    показать_примеры_HR(**kwargs)
    

def показать_примеры_HR(**kwargs):
  lr = kwargs['изображения_низкого_качества']
  hr = kwargs['изображения_высокого_качества']
  '''
  показать_примеры_изображений - функция вырезает небольшие кусочки из hr и lr изображений и выводит в масштабе на экран
  вход:
    lr, hr - списки с lr, hr изображениями соответственно
  '''
  n = 3 # Указываем кол-во отображаемых пар изображений
  fig, axs = plt.subplots(n, 2, figsize=(10, 15)) # Задаем размера вывода изображения

  for i in range(n): # В цикле попарно выводим изображения
    ind = random.randint(0, len(lr)) # Выбираем случайный индекс изображения

    area = (100, 100, 200, 200) # Задаем координаты точек для вырезания участка из изображения низкого качества
    cropped_lr = lr[ind].crop(area) # Вырезаем участок
    area = (400, 400, 800,800) # Задаем координаты точек для вырезания участка из изображения высокого качества
    cropped_hr = hr[ind].crop(area) # Вырезаем участок

    axs[i,0].axis('off')
    axs[i,0].imshow(cropped_lr) # Отображаем lr изображение
    axs[i,0].set_title('Низкое качество', fontsize=30)
    axs[i,1].axis('off')
    axs[i,1].imshow(cropped_hr) # Отображаем hr изображение
    axs[i,1].set_title('Высокое качество', fontsize=30)
  plt.show() #Показываем изображения

###
###                СОЗДАНИЕ ВЫБОРОК
###

def создать_выборки(**kwargs):
  путь = kwargs['путь']
  data_list_seg = ['самолеты','самолеты_макс','губы','люди']
  if not os.path.exists('test_sets'):
    os.mkdir('test_sets')
    
    
  if kwargs['путь'] == 'mnist':
      (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
  
  elif kwargs['путь'] == 'трекер':
    (x_train, y_train), (x_test, y_test) = img_loader(**kwargs)
      
  elif kwargs['путь'] in ['tesla','симптомы_заболеваний']:
    return text_process.создать_выборки(**kwargs)
  
  elif kwargs['путь']=='квартиры':
      return квартиры.создать_выборки()
  
  elif kwargs['путь']=='трейдинг':
      kwargs.pop('путь')
      (x_train, y_train), (x_test, y_test) = traide.получить_данные(**kwargs)
      
  elif kwargs['путь']=='умный_дом':
    kwargs.pop('путь')
    (x_train, y_train), (x_test, y_test) = home.создать_выборки(**kwargs)
  
  elif kwargs['путь']=='трафик':
    dataframe = pd.read_csv('traff.csv', header=None)
    dataframe.rename(columns={0: "Дата", 1: "Трафик"}, inplace=True)
    dataframe['Трафик'] = [float(i.replace(',','')) for i in dataframe['Трафик']]
    return dataframe
  
  elif kwargs['путь']=='вакансии':
    (x_train, y_train), (x_val, y_val) =модель.создать_выборки_вакансии()
    return (x_train, y_train), (x_val, y_val)
    
  elif kwargs['путь']=='диалоги':
    f = open('dialog.txt', 'r', encoding='utf-8')
    text= f.read()
    text = text.replace('"','')
    text = text.split('\n')
    question = text[::3]
    answers = text[1::3]
    return question[:-1], answers
    
  elif kwargs['путь'] == 'повышение_размерности':
      return формирование_повышение_размерности()
  
  elif kwargs['путь'] == 'договоры':
     return загрузить_договоры()

  elif kwargs['путь'] in data_list_seg:
      x, y = обработка_изображений(**kwargs)
      (x_train, y_train), (x_test, y_test) = сегментация.create_xy(x,y)
  else:
    путь_con = ''+kwargs['путь']
    x_train = [] # Создаем пустой список, в который будем собирать примеры изображений обучающей выборки
    y_train = [] # Создаем пустой список, в который будем собирать правильные ответы (метки классов: 0 - Феррари, 1 - Мерседес, 2 - Рено)
    x_test = [] # Создаем пустой список, в который будем собирать примеры изображений тестовой выборки
    y_test = [] # Создаем пустой список, в который будем собирать правильные ответы (метки классов: 0 - Феррари, 1 - Мерседес, 2 - Рено)
    print('Создание наборов данных для обучения модели...')
    for j, d in enumerate(sorted(os.listdir(путь_con))):
        files = sorted(os.listdir(путь_con + '/'+d))    
        count = len(files) * (1-kwargs['коэф_разделения'])
        for i in range(len(files)):
            sample = image.load_img(путь_con + '/' +d +'/'+files[i], target_size=(kwargs['размер'])) # Загружаем картинку
            img_numpy = np.array(sample) # Преобразуем зображение в numpy-массив
            if i<count:
              x_train.append(img_numpy) # Добавляем в список x_train сформированные данные
              y_train.append(j) # Добавлеям в список y_train значение 0-го класса
            else:
              x_test.append(img_numpy) # Добавляем в список x_test сформированные данные
              y_test.append(j) # Добавлеям в список y_test значение 0-го класса
    display.clear_output(wait=True)
    x_train = np.array(x_train) # Преобразуем к numpy-массиву
    y_train = np.array(y_train) # Преобразуем к numpy-массиву
    x_test = np.array(x_test) # Преобразуем к numpy-массиву
    y_test = np.array(y_test) # Преобразуем к numpy-массиву
    x_train = x_train/255.
    x_test = x_test/255.
    print('Размер сформированного массива обучающая_выборка:', x_train.shape)
    print('Размер сформированного массива метки_обучающей_выборки:', y_train.shape)
    print('Размер сформированного массива тестовая_выборка:', x_test.shape)
    print('Размер сформированного массива метки_тестовой_выборки:', y_test.shape)
    
  np.save(f'test_sets/{путь}.npy', x_test)
  np.save(f'test_sets/{путь}_метки.npy', y_test)

  return (x_train, y_train), (x_test, y_test)

def img_loader(путь='трекер', размер=(64,64), коэф_разделения=0.2):
    
    """
    Parameters:
    путь (str)              Путь к папке с датасетом
    размер (tuple)           Размер изображения

    Return:
    x_train, y_train, x_val, y_val
    """
    путь = ''+ путь
    heads = pd.read_csv('трекер/Heads.csv', index_col=0)

    data = []
    labels = []

    for i in range(heads.shape[0]):
        # указываем пути к очередной паре картинок
        first_path = os.path.join(путь, 'images', heads['First image'][i])
        second_path = os.path.join(путь, 'images', heads['Second image'][i])
        
        # загрузка и преобразование к нужному размеру первого изображения
        first_img = Image.open(first_path)
        first_img.thumbnail(размер)
        first_img = np.array(first_img)

        # загрузка и преобразование к нужному размеру второго изображения
        second_img = Image.open(second_path)
        second_img.thumbnail(размер)
        second_img = np.array(second_img)

        # склеиваем по каналам картинки в один массив
        concat_img = np.concatenate((first_img, second_img), axis=2)

        data.append(concat_img)

        label = heads['Label'][i]
        labels.append(label)
    
    data = np.array(data) / 255.0
    labels = np.array(labels)

    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size = коэф_разделения, shuffle = True)
    print('Размер сформированного массива обучающая_выборка:', x_train.shape)
    print('Размер сформированного массива метки_обучающей_выборки:', y_train.shape)
    print('Размер сформированного массива тестовая_выборка:', x_val.shape)
    print('Размер сформированного массива метки_тестовой_выборки:', y_val.shape)
    return (x_train, y_train), (x_val, y_val)

def создать_выборки_вакансии():
  global idx_train, idx_val
  dir = 'вакансии'
    
  """
  Parameters:
  dir (str)              Путь к папке с датасетом

  Return:
  x_train, y_train, x_val, y_val
  """

  data = np.load(dir + '/X_train.npy')
  labels = np.load(dir + '/Y_train.npy')

  x_train, x_val, y_train, y_val = data[:500], data[500:], labels[:500], labels[500:]
  idx_val = [i for i in range(data[500:].shape[0])]

  return (x_train, y_train), (x_val, y_val)

def предобработка_данных(**kwargs):
  if kwargs['сетка'] == 'полносвязная':
    x_train = kwargs['изображения']/255. # Нормируем изображения, приводя каждое значение пикселя к диапазону 0..1
    x_train = x_train.reshape((-1, 28*28)) # Изменяем размер изображения, разворачивая в один вектор
    print('Размер сформированных данных:', x_train[0].shape) # Выводим размер исходного изображения
    return x_train
  elif kwargs['сетка'] == 'сверточная':
    x_train = kwargs['изображения']/255. # Нормируем изображения, приводя каждое значение пикселя к диапазону 0..1
    x_train = x_train.reshape((-1, 28,28,1)) # Изменяем размер изображения, разворачивая в один вектор
    print('Размер сформированных данных:', x_train[0].shape) # Выводим размер исходного изображения
    return x_train

def обработка_изображений(**kwargs):
  return сегментация.get_images(**kwargs)

def распаковать_архив(откуда='', куда=''):
  proc = subprocess.Popen('unzip -q "' + откуда + '" -d ' + куда, shell=True, stdin=None, stdout=open(os.devnull,"wb"), stderr=STDOUT, executable="/bin/bash")
  proc.wait()

def показать_примеры_сегментации(**kwargs):
  сегментация.show_sample(**kwargs)

def загрузить_базу_ОДЕЖДА():
  (x_train_org, y_train_org), (x_test_org, y_test_org) = datasets.fashion_mnist.load_data() # Загружаем данные набора MNIST
  display.clear_output(wait=True)
  print('Данные загружены')
  print('Размер обучающей выборки:', x_train_org.shape) # Отобразим размер обучающей выборки
  print('Размер тестовой выборки:', x_test_org.shape) # Отобразим размер тестовой выборки
  return (x_train_org, y_train_org), (x_test_org, y_test_org)

def загрузить_базу_СТРОЙКА(**kwargs):
  global url
  if 'путь' in kwargs:
	  path = 'drive/MyDrive/' + kwargs['путь'] + 'AIU.zip'
  else:
	  path = 'drive/MyDrive/AIU.zip'
  print('Загрузка данных')
  print('Это может занять несколько минут...')
  распаковать_архив(
    откуда = 'drive/MyDrive/AIU.zip',
    куда = '/content'
  )
  распаковать_архив(
      откуда = "Notebooks.zip",
      куда = "/content"
  )
  x_train = np.load('xTrain_st.npy')
  x_test = np.load('xVal_st.npy')
  y_train = np.load('yTrain_st.npy')
  y_test = np.load('yVal_st.npy')  
  print('Загрузка данных (Готово)')
  return (x_train, y_train), (x_test, y_test)
  
  print('Загрузка данных...')
  urls = ['https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1WtUbopKzQw97W8DChDu0JidJYh08XQyy',
            'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=14gsGpYv13IMUKXmjQEPhPt2bVpVkcJfY',
            'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1A9IThR5f7dUIHgohDDJBeFyAZquTiYIL',
            'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1PtMhqaPXYoJKuKjLy338rBgv-PsScYPh',
            'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1qRAPeOgCZ0g9nikKmop4uWGEe9cCSm4B',
            'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1MDJiPs1Lyh-ij5dldjAu-kwWWb2uPNKi',
            'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1S7bc5yR2fHsR81aDZsIpBcCfxf-43Cek'
            '']
  for url in urls:    
    output = 'data.zip' # Указываем имя файла, в который сохраняем файл
    gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
    if os.path.exists('data.zip'):
        break  
  # Скачиваем и распаковываем архив
  распаковать_архив(
      откуда = "data.zip",
      куда = "/content"
  )
  x_train = np.load('xTrain_st.npy')
  x_test = np.load('xVal_st.npy')
  y_train = np.load('yTrain_st.npy')
  y_test = np.load('yVal_st.npy')  
  print('Загрузка данных (Готово)')
  return (x_train, y_train), (x_test, y_test)


def func_true_false(name, type_check = "show_paths"):
    name = name.replace('/','')
    curr = check_for_errors.check_word(name, type_check)
    name_check = (curr==name)
    path_check = os.path.exists(''+curr)
    if name == 'диалоги':
        path_check = os.path.exists(os.path.join('','dialog.txt'))
    elif name == 'трафик':
        path_check = os.path.exists(os.path.join('','traff.csv'))
    elif name == 'квартиры':
        path_check = os.path.exists(os.path.join('','moscow.csv'))
    elif name_check:
        if path_check:
            pass
        else:
            assert False, f'Загрузите базу \'{curr}\'.'
    else:
        if path_check:
            assert False, f'Ваше название \'{name}\' неверное. Возможно вы имели ввиду \'{curr}\'.'
        else:
            assert False, f'Перепроверьте название и загрузите вашу базу. Работайте для базы \'{curr}\'.'