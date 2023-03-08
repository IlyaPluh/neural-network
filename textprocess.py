import os, nltk, pymorphy2, io, json
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras import utils
from tensorflow.keras.models import Model, load_model
from keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from IPython import display
nltk.download('stopwords', quiet=True)
import importlib.util, sys, gdown
import requests
import ast
import json

def readText(fileName, encod):
    f = open(fileName, 'r', encoding=encod)
    text = f.read()
    text = text.replace("\n", " ")
    return text
    
def создать_выборки(путь, MWC, xLen, step):
  print('Происходит создание выборки для обучения')
  print('Это может занять несколько минут...')
  путь = путь.lower()
  path = ''+путь+'/'
  text = []
  classes = []
  n = 0
  codecs_list = ['UTF-8', 'Windows-1251']

  for filename in sorted(os.listdir(path)): # Проходим по всем файлам в директории договоров
      n +=1
      for codec_s in codecs_list:
        try:
            text.append(readText(path+filename, codec_s)) # Преобразуем файл в одну строку и добавляем в agreements
            classes.append(filename.replace(".txt", ""))
            break
        except UnicodeDecodeError:
            print('Не прочитался файл: ', path+currdir+'/'+filename, codec_s)
        else:
            next 

  stop_words = nltk.corpus.stopwords.words('russian')
  lexeme_list = ['POS', 'animacy', 'aspect', 'case', 'gender', 'involvement', 'mood', 'number', 'person', 'tense', 'transitivity', 'voice']

  words = [] # Здесь будут лежать все списки слов каждого из описаний заболеваний
  tags = []   # Здесь будут лежать все списки списков граммем для каждого слова
  tags_all = [] # Здесь будут лежать все списки граммем всех слов для тренировки токенайзера
  for i in range(len(text)):
    word, tag = text2Words(text[i])
    words.append(word)
    tags.append(tag)
  for k in tags:
    for t in k:
      tags_all.append(t)

  #################
  #Преобразовываем текстовые данные в числовые/векторные для обучения нейросетью
  #################

  # Максимальное количество слов в словаре
  # Токенизатор кераса
  tokenizer = Tokenizer(num_words=MWC, filters='!"#$%&()*+,-––—./:;<=>?@[\\]^_`{|}~\t\n\xa0', lower=True, split=' ', oov_token='unknown', char_level=False)
  # Скармливаем ему слова
  tokenizer.fit_on_texts(words) 
  items = list(tokenizer.word_index.items())


  tokenizer_json1 = tokenizer.to_json()
  with io.open('tokenizer1.json', 'w', encoding='utf-8') as f:
      f.write(json.dumps(tokenizer_json1, ensure_ascii=False))
  with open('tokenizer1.json') as f:
      data = json.load(f)
      tokenizer = tokenizer_from_json(data)

  items = list(tokenizer.word_index.items()) 
  #Выведем первые 10 слов из словаря

  # Максимальное количество слов в словаре
  maxWordsCount2 = MWC 
  # Токенизатор кераса
  tokenizer2 = Tokenizer(num_words=maxWordsCount2, filters='!"#$%&()*+,-––—./:;<=>?@[\\]^_`{|}~\t\n\xa0', lower=True, split=' ', oov_token='unknown', char_level=False)
  # Скармливаем ему слова
  tokenizer2.fit_on_texts(tags_all) 
  items2 = list(tokenizer2.word_index.items()) 

  tokenizer_json2 = tokenizer2.to_json()
  with io.open('tokenizer2.json', 'w', encoding='utf-8') as f:
      f.write(json.dumps(tokenizer_json2, ensure_ascii=False))

  with open('tokenizer2.json') as f:
      data = json.load(f)
      tokenizer2 = tokenizer_from_json(data)

  items2 = list(tokenizer2.word_index.items())

  #Преобразовываем текст в последовательность индексов согласно частотному словарю
  xTrainIndexes = tokenizer.texts_to_sequences(words) #Обучающие тексты в индексы
  #Преобразовываем тэги в последовательность индексов согласно частотному словарю
  xTrainTagsIndexes = []
  for tag in tags:  # так как теги имеют дополнительные вложенные списки поэтому итерируем в нужном для токенайзера формате
    xTrainTagsIndexes.append(tokenizer2.texts_to_sequences(tag))

  nVal = 5000   # Количество слов проверочной выборки

  trainWords = []  # Здесь будет лежать слова для обучающей выборки
  valWords = []    # Здесь будет лежать слова для проверочной выборки

  for i in range(len(xTrainIndexes)):
    trainWords.append(xTrainIndexes[i][:-nVal])
    valWords.append(xTrainIndexes[i][-nVal:])

  trainTagsWords = []  # Здесь будет лежать теги для обучающей выборки
  valTagsWords = []    # Здесь будет лежать тэги для проверочной выборки

  for i in range(len(xTrainTagsIndexes)):
    trainTagsWords.append(xTrainTagsIndexes[i][:-nVal])
    valTagsWords.append(xTrainTagsIndexes[i][-nVal:])

  # Создаем "раскусанные" выборки длины xLen, по 4000 экземпляров на каждый класс 
  (xTrain, yTrain) = createSetsMultiClassesBallanced(trainWords, xLen, step, 4000)
  #(xTrain, yTrain) = createSetsMultiClasses(trainWords, xLen, step)

  (xTrainTags, _) = createSetsMultiClassesBallanced(trainTagsWords, xLen, step, 4000)
  # Преобразовываем полученные обучающие выборки из последовательности индексов в матрицы нулей и единиц по принципу Bag of Words
  xTrain01 = tokenizer.sequences_to_matrix(xTrain.tolist(), mode="tfidf")     # Подаем xTrain в виде списка чтобы метод успешно сработал
  #xTrainTags01 = tokenizer2.sequences_to_matrix(xTrainTags)
  # Создаем "раскусанные" выборки длины xLen для проверочной
  (xVal, yVal) = createSetsMultiClasses(valWords, xLen, step)
  #(xVal, yVal) = createSetsMultiClassesBallanced(valWords, xLen, step, 600)

  (xTagsVal, _) = createSetsMultiClasses(valTagsWords, xLen, step)
  #(xTagsVal, _) = createSetsMultiClassesBallanced(valTagsWords, xLen, step, 600)

  # Преобразовываем полученные выборки из последовательности индексов в матрицы нулей и единиц по принципу Bag of Words
  xVal01 = tokenizer.sequences_to_matrix(xVal.tolist(), mode="tfidf")         # Подаем xVal в виде списка чтобы метод успешно сработал
  #xTagsVal01 = tokenizer2.sequences_to_matrix(xTagsVal.tolist())

  xTrainTags = np.reshape(xTrainTags, (xTrainTags.shape[0], -1))
  xTagsVal = np.reshape(xTagsVal, (xTagsVal.shape[0], -1))
  xTrainTags01 = tokenizer2.sequences_to_matrix(xTrainTags.tolist())
  xTagsVal01 = tokenizer2.sequences_to_matrix(xTagsVal.tolist())

  x_train = [xTrain, xTrain01, xTrainTags01]
  y_train = yTrain
  x_val = [xVal, xVal01, xTagsVal01]
  y_val = yVal
  valw = (valWords, valTagsWords, tokenizer, tokenizer2)
  display.clear_output(wait=True)
  print('Формирование выборки завершено')
  if not os.path.exists('test_sets'):
      os.mkdir('test_sets')
  np.save(f'test_sets/{путь}0.npy', x_val[0])
  np.save(f'test_sets/{путь}1.npy', x_val[1])
  np.save(f'test_sets/{путь}2.npy', x_val[2])
  np.save(f'test_sets/{путь}_метки.npy', y_val)

  return (x_train, y_train), (x_val, y_val)
  
def text2Words(text):
  stop_words = nltk.corpus.stopwords.words('russian')
  lexeme_list = ['POS', 'animacy', 'aspect', 'case', 'gender', 'involvement', 'mood', 'number', 'person', 'tense', 'transitivity', 'voice']

  text = text.replace(".", " ")
  text = text.replace("—", " ")
  text = text.replace(",", " ")
  text = text.replace("!", " ")
  text = text.replace("?", " ")
  text = text.replace("…", " ")
  text = text.replace("-", " ")
  text = text.replace("(", " ")
  text = text.replace(")", " ")
  text = text.replace(";", " ")
  text = text.replace("°c", " ")
  text = text.replace("–", " ")
  text = text.replace("и/ить", " ")
  text = text.lower()
  morph = pymorphy2.MorphAnalyzer()
  
  words = []
  tags = []
  currWord = ""
  for symbol in text[1:]:

    if (symbol != " "):
      currWord += symbol
    else:
      if (currWord != "") & (currWord not in stop_words): # 
        words.append(currWord)
        currWord = ""
      else:
        currWord = ""
  if (currWord != "") & (currWord not in stop_words): #
        words.append(currWord)
  # генератором списков проходимся по каждому сформированному слову и лемматиризируем его с помощью pymorphy до получения граммем
  #words = [morph.parse(word)[0].normal_form for word in words]
  #tags = [morph.parse(word)[0].tag for word in words] # .cyr_repr
  for word in words:
    tag_word = []
    lex_0 = morph.parse(word)[0].tag.POS  # Part of Speech, часть речи
    if lex_0 != None:
        tag_word.append(lex_0)
    else:
        tag_word.append('not')
    lex_1 = morph.parse(word)[0].tag.animacy  # одушевленность
    if lex_1 != None:
        tag_word.append(lex_1)
    else:
        tag_word.append('not')
    lex_2 = morph.parse(word)[0].tag.aspect # вид: совершенный или несовершенный
    if lex_2 != None:
        tag_word.append(lex_2)
    else:
        tag_word.append('not')
    lex_3 = morph.parse(word)[0].tag.case # падеж
    if lex_3 != None:
        tag_word.append(lex_3)  
    else:
        tag_word.append('not')     
    lex_4 = morph.parse(word)[0].tag.gender # род (мужской, женский, средний)
    if lex_4 != None:
        tag_word.append(lex_4)
    else:
        tag_word.append('not') 
    lex_5 = morph.parse(word)[0].tag.involvement  # включенность говорящего в действие
    if lex_5 != None:
        tag_word.append(lex_5)
    else:
        tag_word.append('not')
    lex_6 = morph.parse(word)[0].tag.mood # наклонение (повелительное, изъявительное)
    if lex_6 != None:
        tag_word.append(lex_6)
    else:
        tag_word.append('not')
    lex_7 = morph.parse(word)[0].tag.number # число (единственное, множественное)
    if lex_7 != None:
        tag_word.append(lex_7)
    else:
        tag_word.append('not')
    lex_8 = morph.parse(word)[0].tag.person # лицо (1, 2, 3)
    if lex_8 != None:
        tag_word.append(lex_8)
    else:
        tag_word.append('not')
    lex_9 = morph.parse(word)[0].tag.tense  # время (настоящее, прошедшее, будущее)
    if lex_9 != None:
        tag_word.append(lex_9)
    else:
        tag_word.append('not')
    lex_10 = morph.parse(word)[0].tag.transitivity  # переходность (переходный, непереходный)
    if lex_10 != None:
        tag_word.append(lex_10)
    else:
        tag_word.append('not')
    lex_11 = morph.parse(word)[0].tag.voice # залог (действительный, страдательный)
    if lex_11 != None:
        tag_word.append(lex_11)
    else:
        tag_word.append('not')
    tags.append(tag_word)
    # генератором списков проходимся по каждому сформированному слову и лемматиризируем его с помощью pymorphy
  words = [morph.parse(word)[0].normal_form for word in words]

  return words, tags
  
def createSetsMultiClassesBallanced(wordIndexes, xLen, step, classSize):

  #Для каждого из 6 классов
  #Создаём обучающую выборку из индексов
  nClasses = len(wordIndexes)
  classesXTrain = []
  for wI in wordIndexes:
    classesXTrain.append(getSetFromIndexes(wI, xLen, step))

  #Формируем один общий xTrain
  xTrain = []
  yTrain = []
  
  for t in range(nClasses):
    xT = classesXTrain[t]
    for i in range(classSize):
      xTrain.append(xT[i % len(xT)])
    
    #Формируем yTrain по номеру класса
    currY = utils.to_categorical(t, nClasses)
    for i in range(classSize):
      yTrain.append(currY)

  xTrain = np.array(xTrain)
  yTrain = np.array(yTrain)
  
  return (xTrain, yTrain)
  
def getSetFromIndexes(wordIndexes, xLen, step):
  xTrain = []
  wordsLen = len(wordIndexes)
  index = 0
  while (index + xLen <= wordsLen):
    xTrain.append(wordIndexes[index:index+xLen])
    index += step
    
  return xTrain
  
def createSetsMultiClasses(wordIndexes, xLen, step):

  #Создаём обучающую выборку из индексов
  nClasses = len(wordIndexes)
  classesXTrain = []
  for wI in wordIndexes:
    classesXTrain.append(getSetFromIndexes(wI, xLen, step))

  #Формируем один общий xTrain
  xTrain = []
  yTrain = []
  
  for t in range(nClasses):
    xT = classesXTrain[t]
    for i in range(len(xT)):
      xTrain.append(xT[i])
    
    #Формируем yTrain по номеру класса
    currY = utils.to_categorical(t, nClasses)
    for i in range(len(xT)):
      yTrain.append(currY)

  xTrain = np.array(xTrain)
  yTrain = np.array(yTrain)

  
  return (xTrain, yTrain)

def тест_модели_симп(нейронка, xLen, step, симптомы, классы, **kwargs):
  gastro_10_predict(нейронка, xLen, step, симптомы, классы, **kwargs)
  
def gastro_10_predict(нейронка, xLen, step, text_input = None, классы = None, **kwargs):
    if text_input is None:
        diagnos =  "Текстовое поле пустое"
    
    else:
        classes = классы
        nClasses = len(классы)
        stop_words = nltk.corpus.stopwords.words('russian')
        lexeme_list = ['POS', 'animacy', 'aspect', 'case', 'gender', 'involvement', 'mood', 'number', 'person', 'tense', 'transitivity', 'voice']

        # Функция очистки текста и превращение в набор слов
        def text2Words(text):
          text = text.replace(".", " ")
          text = text.replace("—", " ")
          text = text.replace(",", " ")
          text = text.replace("!", " ")
          text = text.replace("?", " ")
          text = text.replace("…", " ")
          text = text.replace("-", " ")
          text = text.replace("(", " ")
          text = text.replace(")", " ")
          text = text.replace(";", " ")
          text = text.replace("°c", " ")
          text = text.replace("–", " ")
          text = text.replace("и/ить", " ")
          text = text.lower()
          morph = pymorphy2.MorphAnalyzer()
          
          words = []
          tags = []
          currWord = ""
          for symbol in text[1:]:

            if (symbol != " "):
              currWord += symbol
            else:
              if (currWord != "") & (currWord not in stop_words): # 
                words.append(currWord)
                currWord = ""
              else:
                currWord = ""
          if (currWord != "") & (currWord not in stop_words): #
                words.append(currWord)
          # генератором списков проходимся по каждому сформированному слову и лемматиризируем его с помощью pymorphy до получения граммем
          for word in words:
            tag_word = []
            lex_0 = morph.parse(word)[0].tag.POS  # Part of Speech, часть речи
            if lex_0 != None:
                tag_word.append(lex_0)
            else:
                tag_word.append('not')
            lex_1 = morph.parse(word)[0].tag.animacy  # одушевленность
            if lex_1 != None:
                tag_word.append(lex_1)
            else:
                tag_word.append('not')
            lex_2 = morph.parse(word)[0].tag.aspect # вид: совершенный или несовершенный
            if lex_2 != None:
                tag_word.append(lex_2)
            else:
                tag_word.append('not')
            lex_3 = morph.parse(word)[0].tag.case # падеж
            if lex_3 != None:
                tag_word.append(lex_3)  
            else:
                tag_word.append('not')     
            lex_4 = morph.parse(word)[0].tag.gender # род (мужской, женский, средний)
            if lex_4 != None:
                tag_word.append(lex_4)
            else:
                tag_word.append('not') 
            lex_5 = morph.parse(word)[0].tag.involvement  # включенность говорящего в действие
            if lex_5 != None:
                tag_word.append(lex_5)
            else:
                tag_word.append('not')
            lex_6 = morph.parse(word)[0].tag.mood # наклонение (повелительное, изъявительное)
            if lex_6 != None:
                tag_word.append(lex_6)
            else:
                tag_word.append('not')
            lex_7 = morph.parse(word)[0].tag.number # число (единственное, множественное)
            if lex_7 != None:
                tag_word.append(lex_7)
            else:
                tag_word.append('not')
            lex_8 = morph.parse(word)[0].tag.person # лицо (1, 2, 3)
            if lex_8 != None:
                tag_word.append(lex_8)
            else:
                tag_word.append('not')
            lex_9 = morph.parse(word)[0].tag.tense  # время (настоящее, прошедшее, будущее)
            if lex_9 != None:
                tag_word.append(lex_9)
            else:
                tag_word.append('not')
            lex_10 = morph.parse(word)[0].tag.transitivity  # переходность (переходный, непереходный)
            if lex_10 != None:
                tag_word.append(lex_10)
            else:
                tag_word.append('not')
            lex_11 = morph.parse(word)[0].tag.voice # залог (действительный, страдательный)
            if lex_11 != None:
                tag_word.append(lex_11)
            else:
                tag_word.append('not')
            tags.append(tag_word)
            # генератором списков проходимся по каждому сформированному слову и лемматиризируем его с помощью pymorphy
          words = [morph.parse(word)[0].normal_form for word in words]

          return words, tags

        # Формирование обучающей выборки по листу индексов слов (разделение на короткие векторы)
        def getSetFromIndexes(wordIndexes, xLen, step):
          xTrain = []
          wordsLen = len(wordIndexes)
          index = 0
          while (index + xLen <= wordsLen):
            xTrain.append(wordIndexes[index:index+xLen])
            index += step
            
          return xTrain

        # Функция формирование обучающей и проверочной выборки выборки из двух листов индексов от двух классов
        def createSetsMultiClasses(wordIndexes, xLen, step):

          #Для каждого из 10 классов создаём обучающую выборку из индексов
          nClasses = len(wordIndexes)
          classesXTrain = []
          for wI in wordIndexes:
            classesXTrain.append(getSetFromIndexes(wI, xLen, step))

          #Формируем один общий xTrain
          xTrain = []
          yTrain = []
          
          for t in range(nClasses):
            xT = classesXTrain[t]
            for i in range(len(xT)):
              xTrain.append(xT[i])
            
            #Формируем yTrain по номеру класса
            currY = utils.to_categorical(t, nClasses)
            for i in range(len(xT)):
              yTrain.append(currY)

          xTrain = np.array(xTrain)
          yTrain = np.array(yTrain)
          
          return (xTrain, yTrain)

        # Загружаем текст 
        test_text = text_input

        # Здесь будут лежать все списки слов каждого из описаний заболеваний и списки списков граммем для каждого слова
        test_words = []
        test_tags = []
        words, tags = text2Words(test_text[-531:])
        test_words.append(words)
        test_tags.append(tags)

        # загружаем токенайзер слов
        with open('tokenizer1.json') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        # загружаем токенайзер граммем
        with open('tokenizer2.json') as f:
            data = json.load(f)
            tokenizer2 = tokenizer_from_json(data)

        # Преобразовываем слова и тэги в последовательность индексов
        # Делаем короткие описания симптомов  длины xLen, оставляем
      

        #Тестовые тексты в индексы
        xTestIndexes = tokenizer.texts_to_sequences(test_words)

        if len(xTestIndexes[0]) < 50:
          
          xTestIndexes = pad_sequences(xTestIndexes, maxlen=xLen , padding='post', value=0)
          #Преобразовываем тэги в последовательность индексов согласно частотному словарю
          xTestTagsIndexes = []
          for tag in test_tags:  # так как теги имеют дополнительные вложенные списки поэтому итерируем в нужном для токенайзера формате
            xTestTagsIndexes.append(tokenizer2.texts_to_sequences(tag))
          xTestTagsIndexes = pad_sequences(xTestTagsIndexes, maxlen=xLen , padding='post', value=0)
        else:
          xTestTagsIndexes = []
          for tag in test_tags:  # так как теги имеют дополнительные вложенные списки поэтому итерируем в нужном для токенайзера формате
            xTestTagsIndexes.append(tokenizer2.texts_to_sequences(tag))

        # Создаем "раскусанные" выборки длины xLen для тестовой
        xTest, _ = createSetsMultiClasses(xTestIndexes, xLen, step)
        xTestTags, _ = createSetsMultiClasses(xTestTagsIndexes, xLen, step) 

        # Преобразовываем полученные тестовые выборки слов и тэгов из последовательности индексов в матрицы нулей и единиц по принципу Bag of Words
        xTest01 = tokenizer.sequences_to_matrix(xTest.tolist(), mode="tfidf") # Подаем xTest в виде списка чтобы метод успешно сработал
        xTestTags = np.reshape(xTestTags, (xTestTags.shape[0], -1)) # Делаем решейп для токенайзера
        xTestTags01 = tokenizer2.sequences_to_matrix(xTestTags.tolist())  

        # Загружаем модель с весами
        model = нейронка

        # Вывод предсказания в виде номера документа, списка заболеваний и вероятностей того, на сколько описание соответствует тому или иному заболеванию
        sumrec = [x * 0 for x in range(nClasses)]   #создаем список суммы значений предсказанных "окон" зополненный нулями по количеству классов обучающего датасета (в нашем случае их 10)
        meanrec = []                                #создаем пустой список средних значений предсказанных "окон"
        if 'model_type' in kwargs:
          if kwargs['model_type'] == 'BagOfWords':      
            currPred = model.predict(xTest01) #предсказываем документ
            currOut = np.argmax(currPred, axis=1)       #определяем номер распознанного класса для каждого блока слов длины xLen
          elif kwargs['model_type'] == 'BagOfWords+PyMorphy':      
            currPred = model.predict(xTestTags01) #предсказываем документ
            currOut = np.argmax(currPred, axis=1)
          elif kwargs['model_type'] == 'Embedding':      
            currPred = model.predict(xTest) #предсказываем документ
            currOut = np.argmax(currPred, axis=1)
          else:                
            currPred = model.predict([xTest, xTest01, xTestTags01]) #предсказываем документ
            currOut = np.argmax(currPred, axis=1) 
        else:                
            currPred = model.predict([xTest, xTest01, xTestTags01]) #предсказываем документ
            currOut = np.argmax(currPred, axis=1) 
          
        # Проходим по каждому элементу предсказания блоков слов длины xLen и суммируем значения по каждому предсказанному классу в лист sumrec
        for i in range(len(currPred)):
          for j in range(len(currPred[i])):
            sumrec[j] += currPred[i][j]

        # Проходим по каждому элементу листа sumrec и записываем среднее значение вероятности предсказанных блоков слов длины xLen в лист meanrec по каждому классу
        for t in range(len(sumrec)):
          meanrec.append(round(sumrec[t]/len(currPred),4))
          
        # Если нужно то выводим вероятности по всем классам для данного текста
        for m in range(nClasses):
          print(classes[m], ' '* (12 - len(classes[m])),' - ', round(meanrec[m]*100, 2), '%')
          # if round(meanrec[m]*100, 2) > 30:
          #   print(classes[m], ' - ', round(meanrec[m]*100, 2), '%')
        print()
          

        #выводим на печать диагноз
        diagnos =  'Нейросеть думает, что это: ' + str(classes[int(sum(currOut)/len(currPred))]) + ' - ' + str(round(meanrec[int(sum(currOut)/len(currPred))]*100, 2)) + '%'
        print(diagnos)

def тест_модели_писатели(нейронка, xLen, step, симптомы, классы):
  predict_txt(нейронка, xLen, step, симптомы, классы)
  
def predict_txt(нейронка, xLen, step, text_input = None, классы=None):
    if text_input is None:
        diagnos =  "Текстовое поле пустое"
    
    else:       

        classes = классы
        nClasses = len(классы)
        stop_words = nltk.corpus.stopwords.words('russian')
        lexeme_list = ['POS', 'animacy', 'aspect', 'case', 'gender', 'involvement', 'mood', 'number', 'person', 'tense', 'transitivity', 'voice']

        # Функция очистки текста и превращение в набор слов
        def text2Words(text):
          text = text.replace(".", " ")
          text = text.replace("—", " ")
          text = text.replace(",", " ")
          text = text.replace("!", " ")
          text = text.replace("?", " ")
          text = text.replace("…", " ")
          text = text.replace("-", " ")
          text = text.replace("(", " ")
          text = text.replace(")", " ")
          text = text.replace(";", " ")
          text = text.replace("°c", " ")
          text = text.replace("–", " ")
          text = text.replace("и/ить", " ")
          text = text.lower()
          morph = pymorphy2.MorphAnalyzer()
          
          words = []
          tags = []
          currWord = ""
          for symbol in text[1:]:

            if (symbol != " "):
              currWord += symbol
            else:
              if (currWord != "") & (currWord not in stop_words): # 
                words.append(currWord)
                currWord = ""
              else:
                currWord = ""
          if (currWord != "") & (currWord not in stop_words): #
                words.append(currWord)
          # генератором списков проходимся по каждому сформированному слову и лемматиризируем его с помощью pymorphy до получения граммем
          for word in words:
            tag_word = []
            lex_0 = morph.parse(word)[0].tag.POS  # Part of Speech, часть речи
            if lex_0 != None:
                tag_word.append(lex_0)
            else:
                tag_word.append('not')
            lex_1 = morph.parse(word)[0].tag.animacy  # одушевленность
            if lex_1 != None:
                tag_word.append(lex_1)
            else:
                tag_word.append('not')
            lex_2 = morph.parse(word)[0].tag.aspect # вид: совершенный или несовершенный
            if lex_2 != None:
                tag_word.append(lex_2)
            else:
                tag_word.append('not')
            lex_3 = morph.parse(word)[0].tag.case # падеж
            if lex_3 != None:
                tag_word.append(lex_3)  
            else:
                tag_word.append('not')     
            lex_4 = morph.parse(word)[0].tag.gender # род (мужской, женский, средний)
            if lex_4 != None:
                tag_word.append(lex_4)
            else:
                tag_word.append('not') 
            lex_5 = morph.parse(word)[0].tag.involvement  # включенность говорящего в действие
            if lex_5 != None:
                tag_word.append(lex_5)
            else:
                tag_word.append('not')
            lex_6 = morph.parse(word)[0].tag.mood # наклонение (повелительное, изъявительное)
            if lex_6 != None:
                tag_word.append(lex_6)
            else:
                tag_word.append('not')
            lex_7 = morph.parse(word)[0].tag.number # число (единственное, множественное)
            if lex_7 != None:
                tag_word.append(lex_7)
            else:
                tag_word.append('not')
            lex_8 = morph.parse(word)[0].tag.person # лицо (1, 2, 3)
            if lex_8 != None:
                tag_word.append(lex_8)
            else:
                tag_word.append('not')
            lex_9 = morph.parse(word)[0].tag.tense  # время (настоящее, прошедшее, будущее)
            if lex_9 != None:
                tag_word.append(lex_9)
            else:
                tag_word.append('not')
            lex_10 = morph.parse(word)[0].tag.transitivity  # переходность (переходный, непереходный)
            if lex_10 != None:
                tag_word.append(lex_10)
            else:
                tag_word.append('not')
            lex_11 = morph.parse(word)[0].tag.voice # залог (действительный, страдательный)
            if lex_11 != None:
                tag_word.append(lex_11)
            else:
                tag_word.append('not')
            tags.append(tag_word)
            # генератором списков проходимся по каждому сформированному слову и лемматиризируем его с помощью pymorphy
          words = [morph.parse(word)[0].normal_form for word in words]

          return words, tags

        # Формирование обучающей выборки по листу индексов слов (разделение на короткие векторы)
        def getSetFromIndexes(wordIndexes, xLen, step):
          xTrain = []
          wordsLen = len(wordIndexes)
          index = 0
          while (index + xLen <= wordsLen):
            xTrain.append(wordIndexes[index:index+xLen])
            index += step
            
          return xTrain

        # Функция формирование обучающей и проверочной выборки выборки из двух листов индексов от двух классов
        def createSetsMultiClasses(wordIndexes, xLen, step):

          #Для каждого из 10 классов создаём обучающую выборку из индексов
          nClasses = len(wordIndexes)
          classesXTrain = []
          for wI in wordIndexes:
            classesXTrain.append(getSetFromIndexes(wI, xLen, step))

          #Формируем один общий xTrain
          xTrain = []
          yTrain = []
          
          for t in range(nClasses):
            xT = classesXTrain[t]
            for i in range(len(xT)):
              xTrain.append(xT[i])
            
            #Формируем yTrain по номеру класса
            currY = utils.to_categorical(t, nClasses)
            for i in range(len(xT)):
              yTrain.append(currY)

          xTrain = np.array(xTrain)
          yTrain = np.array(yTrain)
          
          return (xTrain, yTrain)

        # Загружаем текст 
        test_text = text_input

        # Здесь будут лежать все списки слов каждого из описаний заболеваний и списки списков граммем для каждого слова
        test_words = []
        test_tags = []
        words, tags = text2Words(test_text[-531:])
        test_words.append(words)
        test_tags.append(tags)

        # загружаем токенайзер слов
        with open('tokenizer1.json') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        # загружаем токенайзер граммем
        with open('tokenizer2.json') as f:
            data = json.load(f)
            tokenizer2 = tokenizer_from_json(data)

        # Преобразовываем слова и тэги в последовательность индексов
        # Делаем короткие описания симптомов  длины xLen, оставляем
        #xLen = 50
        #step = 1

        #Тестовые тексты в индексы
        xTestIndexes = tokenizer.texts_to_sequences(test_words)

        if len(xTestIndexes[0]) < 50:
          
          xTestIndexes = pad_sequences(xTestIndexes, maxlen=xLen , padding='post', value=0)
          #Преобразовываем тэги в последовательность индексов согласно частотному словарю
          xTestTagsIndexes = []
          for tag in test_tags:  # так как теги имеют дополнительные вложенные списки поэтому итерируем в нужном для токенайзера формате
            xTestTagsIndexes.append(tokenizer2.texts_to_sequences(tag))
          xTestTagsIndexes = pad_sequences(xTestTagsIndexes, maxlen=xLen , padding='post', value=0)
        else:
          xTestTagsIndexes = []
          for tag in test_tags:  # так как теги имеют дополнительные вложенные списки поэтому итерируем в нужном для токенайзера формате
            xTestTagsIndexes.append(tokenizer2.texts_to_sequences(tag))

        # Создаем "раскусанные" выборки длины xLen для тестовой
        xTest, _ = createSetsMultiClasses(xTestIndexes, xLen, step)
        xTestTags, _ = createSetsMultiClasses(xTestTagsIndexes, xLen, step) 

        # Преобразовываем полученные тестовые выборки слов и тэгов из последовательности индексов в матрицы нулей и единиц по принципу Bag of Words
        xTest01 = tokenizer.sequences_to_matrix(xTest.tolist(), mode="tfidf") # Подаем xTest в виде списка чтобы метод успешно сработал
        xTestTags = np.reshape(xTestTags, (xTestTags.shape[0], -1)) # Делаем решейп для токенайзера
        xTestTags01 = tokenizer2.sequences_to_matrix(xTestTags.tolist())  

        # Загружаем модель с весами
        model = нейронка

        # Вывод предсказания в виде номера документа, списка заболеваний и вероятностей того, на сколько описание соответствует тому или иному заболеванию
        sumrec = [x * 0 for x in range(nClasses)]   #создаем список суммы значений предсказанных "окон" зополненный нулями по количеству классов обучающего датасета (в нашем случае их 10)
        meanrec = []                                #создаем пустой список средних значений предсказанных "окон"
        currPred = model.predict([xTest, xTest01, xTestTags01]) #предсказываем документ
        currOut = np.argmax(currPred, axis=1)       #определяем номер распознанного класса для каждого блока слов длины xLen
          
        # Проходим по каждому элементу предсказания блоков слов длины xLen и суммируем значения по каждому предсказанному классу в лист sumrec
        for i in range(len(currPred)):
          for j in range(len(currPred[i])):
            sumrec[j] += currPred[i][j]

        # Проходим по каждому элементу листа sumrec и записываем среднее значение вероятности предсказанных блоков слов длины xLen в лист meanrec по каждому классу
        for t in range(len(sumrec)):
          meanrec.append(round(sumrec[t]/len(currPred),4))
          
        # Если нужно то выводим вероятности по всем классам для данного текста
        for m in range(nClasses):
          print(classes[m], ' '* (12 - len(classes[m])),' - ', round(meanrec[m]*100, 2), '%')
          # if round(meanrec[m]*100, 2) > 30:
          #   print(classes[m], ' - ', round(meanrec[m]*100, 2), '%')
        print()
          

        #выводим на печать диагноз
        diagnos =  'Нейросеть считает что это: ' + str(classes[int(sum(currOut)/len(currPred))]) + ' - ' + str(round(meanrec[int(sum(currOut)/len(currPred))]*100, 2)) + '%'
        print(diagnos)
        
def создать_выборки_чатбота(вопросы, ответы, количество_пар=10000):
  ######################
  # Разбираем вопросы-ответы с проставлением тегов ответам
  ######################
  # Собираем вопросы и ответы в списки
  questions = вопросы[:количество_пар] # здесь будет список вопросов
  answers = ответы[:количество_пар  ] # здесь будет список ответов  
  assert len(questions) == len(answers), 'Количество вопросов не совпадает с количеством ответов'


  # Сделаем теги-метки для начала и конца ответов
  n_answers = list()
  for i in range(len(answers)):
    n_answers.append( '<START> ' + answers[i] + ' <END>' )

  ######################
  # Подключаем керасовский токенизатор и собираем словарь индексов
  ######################
  tokenizer = Tokenizer(oov_token='unknown')
  tokenizer.fit_on_texts(questions + n_answers) # загружаем в токенизатор список вопросов-ответов для сборки словаря частотности
  vocabularyItems = list(tokenizer.word_index.items()) # список с cодержимым словаря
  vocabularySize = len(vocabularyItems)+1 # размер словаря
  
  ######################
  # Устанавливаем закодированные входные данные(вопросы)
  ######################
  tokenizedQuestions = tokenizer.texts_to_sequences(questions) # разбиваем текст вопросов на последовательности индексов
  maxLenQuestions = max([ len(x) for x in tokenizedQuestions]) # уточняем длину самого большого вопроса
  # Делаем последовательности одной длины, заполняя нулями более короткие вопросы
  paddedQuestions = pad_sequences(tokenizedQuestions, maxlen=maxLenQuestions, padding='post')

  # Предподготавливаем данные для входа в сеть
  encoderForInput = np.array(paddedQuestions) # переводим в numpy массив

  ######################
  # Устанавливаем раскодированные входные данные(ответы)
  ######################
  tokenizedAnswers = tokenizer.texts_to_sequences(n_answers) # разбиваем текст ответов на последовательности индексов
  maxLenAnswers = max([len(x) for x in tokenizedAnswers]) # уточняем длину самого большого ответа
  # Делаем последовательности одной длины, заполняя нулями более короткие ответы
  paddedAnswers = pad_sequences(tokenizedAnswers, maxlen=maxLenAnswers, padding='post')

  # Предподготавливаем данные для входа в сеть
  decoderForInput = np.array(paddedAnswers) # переводим в numpy массив

  ######################
  # Раскодированные выходные данные(ответы)
  ######################
  tokenizedAnswers = tokenizer.texts_to_sequences(n_answers) # разбиваем текст ответов на последовательности индексов
  for i in range(len(tokenizedAnswers)) : # для разбитых на последовательности ответов
    tokenizedAnswers[i] = tokenizedAnswers[i][1:] # избавляемся от тега <START>
  # Делаем последовательности одной длины, заполняя нулями более короткие ответы
  paddedAnswers = pad_sequences(tokenizedAnswers, maxlen=maxLenAnswers , padding='post')

  #oneHotAnswers = utils.to_categorical(paddedAnswers, vocabularySize) # переводим в one hot vector
  oneHotAnswers = paddedAnswers # переводим в one hot vector
  decoderForOutput = np.array(oneHotAnswers) # и сохраняем в виде массива numpy
  tokenizer_json1 = tokenizer.to_json()
  with io.open('tokenizer_cb.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json1, ensure_ascii=False))
  return (encoderForInput, decoderForInput), decoderForOutput, vocabularySize

def тест_модели_чат_бот(модель, размер_словаря, энкодер, декодер):
  with open('tokenizer_cb.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
    
  def makeInferenceModels(): 
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
  
    encoderModel = Model(encoderInputs, encoderStates) 
    
    
    decoderStateInput_h = Input(shape=(None ,)) # обозначим размерность для входного слоя с состоянием state_h
    decoderStateInput_c = Input(shape=(None ,)) # обозначим размерность для входного слоя с состоянием state_c
    decoderStatesInputs = [decoderStateInput_h, decoderStateInput_c] # возьмем оба inputs вместе и запишем в decoderStatesInputs
    
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
  
    decoderOutputs, state_h, state_c = decoderLSTM(x, initial_state=decoderStatesInputs)
    decoderStates = [state_h, state_c] # LSTM даст нам новые состояния
    decoderDense = Dense(размер_словаря, activation='softmax') 
    decoderOutputs = decoderDense(decoderOutputs) # и ответы, которые мы пропустим через полносвязный слой с софтмаксом
    decoderModel = Model([decoderInputs] + decoderStatesInputs, [decoderOutputs] + decoderStates)
    return encoderModel , decoderModel

  def strToTokens(sentence: str): # функция принимает строку на вход (предложение с вопросом)
    words = sentence.lower().split() # приводит предложение к нижнему регистру и разбирает на слова
    tokensList = list() # здесь будет последовательность токенов/индексов
    for word in words: # для каждого слова в предложении
      try:
        tokensList.append(tokenizer.word_index[word]) # определяем токенизатором индекс и добавляем в список
      except:
        print('Я не знаю такого слова:', word)
        tokensList.append(0)
    return pad_sequences([tokensList], maxlen=13 , padding='post')
    

  encModel , decModel = makeInferenceModels() # запускаем функцию для построения модели кодера и декодера
  print('Тест общения с ботом. (Для завершения наберите «Выход»)')
  while(True):
      # Получаем значения состояний, которые определит кодер в соответствии с заданным вопросом
      ques = input( 'Задайте вопрос : ' )
      if ques=='Выход':
	      break
      statesValues = encModel.predict(strToTokens(ques))
      # Создаём пустой массив размером (1, 1)
      emptyTargetSeq = np.zeros((1, 1))    
      emptyTargetSeq[0, 0] = tokenizer.word_index['start'] # положим в пустую последовательность начальное слово 'start' в виде индекса

      stopCondition = False # зададим условие, при срабатывании которого, прекратится генерация очередного слова
      decodedTranslation = '' # здесь будет собираться генерируемый ответ
      while not stopCondition : # пока не сработало стоп-условие
        # В модель декодера подадим пустую последовательность со словом 'start' и состояния предсказанные кодером по заданному вопросу.
        # декодер заменит слово 'start' предсказанным сгенерированным словом и обновит состояния
        decOutputs , h , c = decModel.predict([emptyTargetSeq] + statesValues)
        
        #argmax пробежит по вектору decOutputs'а[0,0,15104], найдет макс.значение, и вернёт нам номер индекса под которым оно лежит в массиве
        sampledWordIndex = np.argmax( decOutputs[0, 0, :]) # argmax возьмем от оси, в которой 15104 элементов. Получили индекс предсказанного слова.
        sampledWord = None # создаем переменную, в которую положим слово, преобразованное на естественный язык
        for word , index in tokenizer.word_index.items():
          if sampledWordIndex == index: # если индекс выбранного слова соответствует какому-то индексу из словаря
            decodedTranslation += ' {}'.format(word) # слово, идущее под этим индексом в словаре, добавляется в итоговый ответ 
            sampledWord = word # выбранное слово фиксируем в переменную sampledWord
        
        # Если выбранным словом оказывается 'end' либо если сгенерированный ответ превышает заданную максимальную длину ответа
        if sampledWord == 'end' or len(decodedTranslation.split()) > 13:
          stopCondition = True # то срабатывает стоп-условие и прекращаем генерацию

        emptyTargetSeq = np.zeros((1, 1)) # создаем пустой массив
        emptyTargetSeq[0, 0] = sampledWordIndex # заносим туда индекс выбранного слова
        statesValues = [h, c] # и состояния, обновленные декодером
        # и продолжаем цикл с обновленными параметрами
      
      print(decodedTranslation[:-3]) # выводим ответ сгенерированный декодером

def разметка_договора(model, корректор):
  xLen = 256
  step = 30
  embeddingSize = 300
  t = корректор

  test_file = 'Договоры/test_doc.txt' # Путь к папке с договором
  
  ################## ПОВТОР ВЫШЕ ИМЕЮЩИХСЯ ФУНКЦИЙ из лекции ###########################################
  # Чтение файла текста из файла, очитска от знаков препинания
  def readText(fileName):
    f = open(fileName, 'r') #Открываем наш файл для чтения и считываем из него данные 
    text = f.read() #Записываем прочитанный текст в переменную 
    # Определяем символы для удаления
    delSymbols = ['\n', "\t", "\ufeff", ".", "_", "-", ",", "!", "?", "–", "(", ")", "«", "»", "№", ";"]

    for dS in delSymbols: # Каждый символ в списке символов для удаления
      text = text.replace(dS, " ") # Удаляем, заменяя на пробел

    # Ищем шаблоны и символы в строке и меняем на указанную подстроку
    text = re.sub("[.]", " ", text) 
    text = re.sub(":", " ", text)
    text = re.sub("<", " <", text)
    text = re.sub(">", "> ", text)

    # Метод split разделит текст по пробелам (а их может быть несколько после удаления символов)
    # При помощи метода join запишем все разделенные слова снова в строку
    text = ' '.join(text.split()) 

    text = text.lower() # Переводим текст в нижний регистр
    return text # Возвращаем тексты

  # Преобразование исходного текста в список из слов в нормальной форме 
  def text2Words(text):
    morph = pymorphy2.MorphAnalyzer() # Создаем экземпляр класса MorphAnalyzer
    words = text.split(' ') # Разделяем текст на пробелы
    words = [morph.parse(word)[0].normal_form for word in words] #Переводим каждое слово в нормалную форму  
    return words # Возвращаем слова


  def get01XSamples(tok_agreem, tags_index):
    tags01 = [] # Список для тегов
    indexes = [] # Здесь будут лежать индексы
  
    for agreement in tok_agreem: # Проходимся по каждому договору-списку
      tag_place = [0, 0, 0, 0, 0, 0] # Создаем вектор [0,0,0,0,0,0]
      for ex in agreement: # Проходимся по каждому слову договора
          if ex in tags_index: # Смотрим, если индекс оказался нашим тегом
            place = np.argwhere(tags_index==ex) # Записываем под каким местом лежит этот тег в своем списке
            if len(place)!=0: # Проверяем, чтобы тег действительно был
              if place[0][0]<6: # Первые шесть тегов в списке - открывающие
                tag_place[place[0][0]] = 1    # Поэтому ставим 1
              else: 
                tag_place[place[0][0] - 6] = 0  # Остальные теги закрывающие, так что меняем на ноль
          else:          
            tags01.append(tag_place.copy()) # Расширяем наш список с каждой итерацией. Получаем в конце длинный список из всех тегов в одном 
            indexes.append(ex) # Докидываем индекс слова в список индексов

    return indexes, tags01

  # Получение списка слов из индексов
  def reverseIndex(clean_voc, x):
    reverse_word_map = dict(map(reversed, clean_voc.items())) # Берем пары значений всего словаря и размечаем наоборот, т.е. value:key
    words = [reverse_word_map.get(letter) for letter in x] # Вытаскиваем по каждому ключу в список
    return words # Возвращаем полученный текст


  # Формируем выборку из индексов
  def getSetFromIndexes(wordIndexes, xLen, step): 
    xBatch = [] # Лист для фрагментов текста
    wordsLen = len(wordIndexes) # Определяем длинну текста
    index = 0 # Задаем стартовый индекс
    
    while (index + xLen <= wordsLen): # Пока сумма индекса с длинной фрагмента меньше или равна числу слов в выборке
      xBatch.append(wordIndexes[index:index+xLen]) # Добавляем X в лист фразментов текста
      index += step # Сдвигаемся на step

    return xBatch # Лист для фрагментов текста

  # Создаем выборки
  def getSets(model, senI, tagI):
    xVector = [] # Здесь будет лежать embedding представление каждого из индексов
    tmp = [] # Временный список
    for text in senI: # Проходимся по каждому тексту-списку
      tmp=[]
      for word in text: # Проходимся по каждому слову в тексте-списке
        tmp.append(model[word]) 

      xVector.append(tmp)

    return np.array(xVector), np.array(tagI)
  ############################################################################################################

  # Получение договора для последующей разметки
  test_dogovor = [] # Список, в который запишем все наши договоры test_dogovor
  txt = readText(test_file) # Читаем текст договора
  if txt != '': # Если текст не пустой
    test_dogovor.append(readText(test_file)) # Преобразуем файл в одну строку и добавляем в agreements

  # покажем текст
  n = 0
  print(os.listdir(test_directory)[n], '\n', '\n') # Выведем название n-ого договора
  #test_dogovor[n] 

  #переводим в слова
  words_test_dogovor = [] # Здесь будут храниться все договора в виде списка слов
  curTime = time.time() # Засечем текущее время
  for i in range(len(test_dogovor)): # Проходимся по всем договорам
    words_test_dogovor.append(text2Words(test_dogovor[i])) # Преобразуем очередной договор в список слов и добавляем в words

  import textwrap
  print("Полученный и очищенный от лишних знаков договор", '\n')
  text_vnachale = ' '.join(words_test_dogovor[0])
  print(textwrap.fill(text_vnachale, 100), '\n', '\n')

  # Преобразовываем текст в последовательность индексов согласно частотному словарю
  test_dogovor_tok_agreem = tokenizer.texts_to_sequences(words_test_dogovor) # Обучающие тесты в индексы

  # decoded_text => test_dogovor_decoded
  test_dogovor_xData, test_dogovor_yData = get01XSamples(test_dogovor_tok_agreem,tags_index) # Распознаем теги и создаем список с ними, с индексами
  test_dogovor_decoded = reverseIndex(clean_voc, test_dogovor_xData) # Для создания списков с embedding-ами сначала преобразуем список индексов обратно в слова

  # Генерируем наборы с заданными параметрами окна
  test_dogovor_xTest = getSetFromIndexes(test_dogovor_decoded, xLen, step) # Последовательность из xLen слов
  test_dogovor_yTest = getSetFromIndexes(test_dogovor_yData, xLen, step) # Последовательность из xLen-тегов
  modelGENSIM = word2vec.Word2Vec(test_dogovor_xTest, size = embeddingSize, window = 10, min_count = 1, workers = 10, iter = 10)

  test_dogovor_GENSIM, y_test_dogovor_GENSIM = getSets(modelGENSIM, test_dogovor_xTest, test_dogovor_yTest)

  pred = model(test_dogovor_GENSIM) # получаем предикт от модели
  print('Категории по которым размечаются договоры', '\n')
  print('s1 - Условия')
  print('s2 - Запреты')
  print('s3 - Стоимость, всё про цены и деньги')
  print('s4 - Всё про сроки')
  print('s5 - Неустойка')
  print('s6 - Всё про адреса и геолокации', '\n', '\n')


  # функция реконструирует из предикта в лист векторов 0 и 1
  def reconst_predict(pred, t):
    pred_new = np.zeros_like(pred)
    mask = np.where(pred > t) # Записываем маску в при сравнении с корректоромзаданным в скобках функции
    pred_new[mask] = 1  # присваиваеи 1 где выне порога заданного корректором

    k = xLen//step                      # сколько целых шагов step в наборе xLen
    p = len(test_dogovor_yData)//xLen   # сколько целых шагов xLen в договоре test_dogovor_yData
    reconst = pred_new[0]               # берем 1й кусок xLen
    for i in range(1, p):              
      reconst = np.append(reconst, pred_new[i*k], axis = 0)   # берем через xLen еще и  еще  кусок xLen  p раз
    end = pred_new[-1][:(len(test_dogovor_yData)-p*xLen)]     # вычисляем сколько нужно в конце до длины test_dogovor_yData
    reconst_lst = np.append(reconst, end, axis = 0)           # докидываем в конце остаток

    return reconst_lst


  #списки тегов для функций ниже
  open_tags = ['<s' + str(i) + '>' for i in range(1, 7)] # Получаем список открывающих тегов
  all_tags = open_tags
  close_tags = ['</s' + str(i) + '>' for i in range(1, 7)] # Получаем список закрывающих тегов
  all_tags.extend(close_tags)

  # функция обратного преоброзавания списка из векторов 0 и1 собранного из предикта
  def get_list_with_tags(reconst_zone):
    # вставка тегов
    super_lst = []
    check = [0,0,0,0,0,0]
    for i in range(len(reconst_zone)):
      for j in range(len(reconst_zone[i])):
        if (reconst_zone[i][j] == 1 and check[j] == 0):
          super_lst.append(open_tags[j])
          check[j] = 1
        if (reconst_zone[i][j] == 1 and check[j] == 1):
          pass
        if (reconst_zone[i][j] == 0 and check[j] == 1):
          super_lst.append(close_tags[j])
          check[j] = 0
        if (reconst_zone[i][j] == 0 and check[j] == 0):
          pass
      super_lst.append(0)
    list_with_tags = super_lst   

    return list_with_tags


  # функция разметки ОДНОГО ПЕРВОГО договора тегами, под демонстрацию!!!!
  def razmetka(dogovor_lst, list_with_tags):
    dogovor = dogovor_lst[0]
    out_list=[]
    s = 0
    k = len(dogovor)/len(list_with_tags) # dogovor и list_with_tags - эти массивы обычно не совпадают
    # и нам нужно перенести разбиение на зоны в одном в  разбиение на зоны в другом
    for j in range(len(list_with_tags)):
      i = round(j*k)                    # подгонка одного к другоиу
      for tags in all_tags:
        if i < len(list_with_tags):     # проверка на длину и текущий индекс
          if list_with_tags[i] == tags:
            out_list.extend(dogovor[s:i])
            out_list.append(list_with_tags[i])
            s = i
    if s < len(dogovor): # проверка на длину и финальный индекс
      out_list.extend(dogovor[s:])      # пристыковка финальной части если осталась
          
    return out_list

  recont_list = reconst_predict(pred, t)

  list_with_tags = get_list_with_tags(recont_list)

  dogovor_s_razmetka = razmetka(words_test_dogovor,list_with_tags)

  text_razmet = ' '.join(dogovor_s_razmetka)

  print("Размеченый нейронкой договор", '\n')
  print(textwrap.fill(text_razmet, 100), '\n', '\n')


  #Получение ранее размеченного вручную договора для сравнения
  test_dogovor_1 = [] # Список, в который запишем все наши договоры test_dogovor
  for filename in os.listdir(test_directory_1): # Проходим по всем файлам в директории договоров
    txt = readText(test_directory_1 + filename) # Читаем текст договора
    if txt != '': # Если текст не пустой
      test_dogovor_1.append(readText(test_directory_1 + filename)) # Преобразуем файл в одну строку и добавляем в agreements

  #переводим в слова
  words_test_dogovor_1 = [] # Здесь будут храниться все договора в виде списка слов
  curTime = time.time() # Засечем текущее время
  for i in range(len(test_dogovor_1)): # Проходимся по всем договорам
    words_test_dogovor_1.append(text2Words(test_dogovor_1[i])) # Преобразуем очередной договор в список слов и добавляем в words

  print("Размеченный ранее вручную договор для сравнения", '\n')
  text_vnachale = ' '.join(words_test_dogovor_1[0])
  print(textwrap.fill(text_vnachale, 100))


def загрузить_предобученную_модель():
  url = 'https://storage.googleapis.com/aiu_bucket/tokenizer_best.json'
  output = 'tokenizer_best.json' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL

  with open('tokenizer_best.json') as f:
    data = json.load(f)
    токенайзер = tokenizer_from_json(data)
  # url = ''
  # output = 'model_chatbot_100epochs(rms)+50(ada).h5'
  # gdown.download(url, output, quiet=True)  
  url = 'https://storage.googleapis.com/aiu_bucket/model_chatbot_100epochs(rms)%2B50(ada).h5'
  output = 'model_chatbot_100epochs(rms)%2B50(ada).h5' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  model = load_model('model_chatbot_100epochs(rms)%2B50(ada).h5')


  def strToTokens(sentence: str): # функция принимает строку на вход (предложение с вопросом)
    words = sentence.lower().split() # приводит предложение к нижнему регистру и разбирает на слова
    tokensList = list() # здесь будет последовательность токенов/индексов
    for word in words: # для каждого слова в предложении
      try:
        tokensList.append(токенайзер.word_index[word]) # определяем токенизатором индекс и добавляем в список
      except KeyError:
        pass
    # Функция вернёт вопрос в виде последовательности индексов, ограниченной длиной самого длинного вопроса из нашей базы вопросов
    return pad_sequences([tokensList], maxlen=13, padding='post')
  ######################
  # Устанавливаем связи между слоями рабочей модели и предобученной
  ######################
  def loadInferenceModels():
    encoderInputs = model.input[0]   # входом энкодера рабочей модели будет первый инпут предобученной модели(input_1)
    encoderEmbedding = model.layers[2] # связываем эмбединг слои(model.layers[2] это embedding_1)
    encoderOutputs, state_h_enc, state_c_enc = model.layers[4].output # вытягиваем аутпуты из первого LSTM слоя обуч.модели и даем энкодеру(lstm_1)
    encoderStates = [state_h_enc, state_c_enc] # ложим забранные состояния в состояния энкодера
    encoderModel = Model(encoderInputs, encoderStates) # формируем модель

    decoderInputs = model.input[1]   # входом декодера рабочей модели будет второй инпут предобученной модели(input_2)
    decoderStateInput_h = Input(shape=(200 ,)) # обозначим размерность для входного слоя с состоянием state_h
    decoderStateInput_c = Input(shape=(200 ,)) # обозначим размерность для входного слоя с состоянием state_c

    decoderStatesInputs = [decoderStateInput_h, decoderStateInput_c] # возьмем оба inputs вместе и запишем в decoderStatesInputs

    decoderEmbedding = model.layers[3] # связываем эмбединг слои(model.layers[3] это embedding_2)
    decoderLSTM = model.layers[5] # связываем LSTM слои(model.layers[5] это lstm_2)
    decoderOutputs, state_h, state_c = decoderLSTM(decoderEmbedding.output, initial_state=decoderStatesInputs)
    decoderStates = [state_h, state_c] # LSTM даст нам новые состояния

    decoderDense = model.layers[6] # связываем полносвязные слои(model.layers[6] это dense_1)
    decoderOutputs = decoderDense(decoderOutputs) # выход с LSTM мы пропустим через полносвязный слой с софтмаксом

      # Определим модель декодера, на входе далее будут раскодированные ответы (decoderForInputs) и состояния
      # на выходе предсказываемый ответ и новые состояния
    decoderModel = Model([decoderInputs] + decoderStatesInputs, [decoderOutputs] + decoderStates)
    return encoderModel , decoderModel

  ######################
  # Устанавливаем окончательные настройки и запускаем рабочую модель над предобученной
  ######################

  encModel , decModel = loadInferenceModels() # запускаем функцию для построения модели кодера и декодера
  
  display.clear_output(wait=True)
  print()
  print('Тест общения с ботом. (Для завершения наберите «Выход»)')
  while(True):
    # Получаем значения состояний, которые определит кодер в соответствии с заданным вопросом
    ques = input( 'Задайте вопрос : ' )
    if ques=='Выход':
	    break
    statesValues = encModel.predict(strToTokens(ques))
    # Создаём пустой массив размером (1, 1)
    emptyTargetSeq = np.zeros((1, 1))    
    emptyTargetSeq[0, 0] = токенайзер.word_index['start'] # положим в пустую последовательность начальное слово 'start' в виде индекса

    stopCondition = False # зададим условие, при срабатывании которого, прекратится генерация очередного слова
    decodedTranslation = '' # здесь будет собираться генерируемый ответ
    while not stopCondition : # пока не сработало стоп-условие
      # В модель декодера подадим пустую последовательность со словом 'start' и состояния предсказанные кодером по заданному вопросу.
      # декодер заменит слово 'start' предсказанным сгенерированным словом и обновит состояния
      decOutputs , h , c = decModel.predict([emptyTargetSeq] + statesValues)
      
      #argmax пробежит по вектору decOutputs'а[0,0,15104], найдет макс.значение, и вернёт нам номер индекса под которым оно лежит в массиве
      sampledWordIndex = np.argmax( decOutputs[0, 0, :]) # argmax возьмем от оси, в которой 15104 элементов. Получили индекс предсказанного слова.
      sampledWord = None # создаем переменную, в которую положим слово, преобразованное на естественный язык
      for word , index in токенайзер.word_index.items():
        if sampledWordIndex == index: # если индекс выбранного слова соответствует какому-то индексу из словаря
          decodedTranslation += ' {}'.format(word) # слово, идущее под этим индексом в словаре, добавляется в итоговый ответ 
          sampledWord = word # выбранное слово фиксируем в переменную sampledWord
      
      # Если выбранным словом оказывается 'end' либо если сгенерированный ответ превышает заданную максимальную длину ответа
      if sampledWord == 'end' or len(decodedTranslation.split()) > 13:
        stopCondition = True # то срабатывает стоп-условие и прекращаем генерацию

      emptyTargetSeq = np.zeros((1, 1)) # создаем пустой массив
      emptyTargetSeq[0, 0] = sampledWordIndex # заносим туда индекс выбранного слова
      statesValues = [h, c] # и состояния, обновленные декодером
      # и продолжаем цикл с обновленными параметрами
    
    print(decodedTranslation[:-3]) # выводим ответ сгенерированный декодером

def func_true_false(name, type_check = "show_paths"):
    name = name.replace('/','')
    curr = check_for_errors.check_word(name, type_check)
    name_check = (curr==name)
    path_check = os.path.exists(''+curr)
    if name_check:
        if path_check:
            pass
        else:
            assert False, f'Загрузите базу для \'{curr}\'.'
    else:
        if path_check:
            assert False, f'Ваше название \'{name}\' неверное. Возможно вы имели ввиду \'{curr}\'.'
        else:
            assert False, f'Перепроверьте название и загрузите вашу базу. Работайте для базы \'{curr}\'.'