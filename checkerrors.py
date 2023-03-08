import Levenshtein
import numpy as np
import re
from IPython import display
from IPython.display import clear_output
from tqdm.notebook import tqdm_notebook
import time
import os
import requests
import datetime
import time
import requests
from termcolor import colored
display.clear_output(wait=True)



datasets = ['mnist','авто-2','авто-3', 'молочная_продукция','автобусы', 'tesla', 'обнаружение_возгораний', 'майонез', 'трекер',\
             'вакансии', 'самолеты', 'самолеты_макс', 'люди', 'кожные_заболевания', 'повышение_размерности', 'обнаружение_людей',\
             'симптомы_заболеваний', 'писатели', 'диалоги', 'договоры', 'квартиры', 'губы', 'трафик', 'умный_дом', 'трейдинг']

all_layers = ['Полносвязный','Повтор','Эмбеддинг','Сверточный2D','Сверточный1D','Выравнивающий',\
          'Нормализация','Нормализация_01','Нормализация_11','Денормализация','Активация',\
          'ЛСТМ','МаксПуллинг2D','МаксПуллинг1D','Дропаут','PReLU','LeakyReLU']

all_task = ['классификация изображений', 'классификация вакансий','временной ряд','аудио',\
             'сегментация изображений','сегментация текста']
             
all_trade = ['газпром', 'яндекс', 'полиметаллы']

all_home =['кондиционер','свет','телевизор','фон']

all_paths = ['молочная_продукция', 'автобусы', 'майонез', 'обнаружение_возгораний',
           'авто-2','авто-3','молочная_продукция','умный_дом', 'договоры', 'tesla',\
           'симптомы_заболеваний','писатели', 'трекер', 'вакансии', 'mnist']

def check_word(arg, word_type=None, extra=None):
    arg = arg.upper()
    if extra:
        distances = [Levenshtein.distance(arg.upper(), i.upper()) for i in extra]
        return f'{extra[np.argmin(distances)]}' 
    if word_type=='dataset':
        distances = [Levenshtein.distance(arg.upper(), i.upper()) for i in datasets]
        return f'{datasets[np.argmin(distances)]}'
    elif word_type=='layer':
        distances = [Levenshtein.distance(arg.upper(), i.upper()) for i in all_layers]
        return f'{all_layers[np.argmin(distances)]}'
    elif word_type=='task':
        distances = [Levenshtein.distance(arg.upper(), i.upper()) for i in all_task]
        return f'{all_task[np.argmin(distances)]}' 
    elif word_type=='trade':
        distances = [Levenshtein.distance(arg.upper(), i.upper()) for i in all_trade]
        return f'{all_trade[np.argmin(distances)]}'
    elif word_type=='home':
        distances = [Levenshtein.distance(arg.upper(), i.upper()) for i in all_home]
        return f'{all_home[np.argmin(distances)]}'
    elif word_type=='show_paths':
        distances = [Levenshtein.distance(arg.upper(), i.upper()) for i in all_paths]
        return f'{all_paths[np.argmin(distances)]}' 


def decorator_check(func):
  def inner_func(arg, **kwargs):
    try:
      return func(arg, **kwargs)
    except ValueError:
      assert False, f'У вас присутсвует строка \'{arg}\', проверьте наличие отсупа или пробела в этом месте.'
  return inner_func

def test_homework(arg):
    check_words = []
    if arg==1:
        paths = [os.path.exists('car.zip'), os.path.exists('mayonnaise.zip')]
    elif arg==2:
        paths = [os.path.exists('HR.zip'), os.path.exists('tesla.zip'), os.path.exists('symptoms.zip')]
    elif arg==3:
        paths = [os.path.exists('cHome.zip'), os.path.exists('shares.zip'), os.path.exists('lips.zip')]
    else:
        assert False,'Введите число от 1 до 3, где число отвечает за номер выполненого домашнего задания.'
    for i in range(1,len(paths)+1):
        if paths[i-1]:
            pass
        else:
            check_words.append(f'Вы ещё не работали с задачей-{i}\n')
    for i in tqdm_notebook(range(len(paths) - len(check_words)), desc='HW', total = len(paths)):
        time.sleep(0.5)

    for i in check_words:
        print(i)

    if len(check_words)==0:
        return True
    return False

def send_work(dict_arg_homework = {1:1441, 2:1442, 3:1446}, LP = 204886, online='ваш учебный день', home='интенсив'):
    numbers = list(dict_arg_homework)
    numbers_1 = ''
    for i in numbers[:-1]:
        numbers_1+=f'{i} '
    education_number = int(input(f'Введите {online} - {numbers_1}или {numbers[-1]}: '))
    display.clear_output()
    if home=='интенсив':
        result = test_homework(education_number)
    else:
        pass
    if result:
        username = input('Введите ваш email: ')
        response = requests.get('https://skochnev.dev.neural-university.com/api/v2/users',
                                params ={'username': f'{username}'})
        if response.json()['hydra:member']:
            id_user = response.json()['hydra:member'][0]['id']
            time_now = str(datetime.datetime.fromtimestamp(time.time()+10800))
            params = {
            "homework": f"/api/v2/homework/{dict_arg_homework[education_number]}",
            "description": "Ваша работа зачтена. Успехов!",
            "filing_at": str(time_now).replace(' ','T')+'Z',
            "student": f"/api/v2/users/{id_user}",
            "canComment": True,
            "vision": 0,
            "points": 1,
            "status": 4,
            "learningProgramm": f"/api/v2/learning_programms/{LP}",
            'complexity_level':1
            }
            response = requests.post(
                'https://skochnev.dev.neural-university.com/api/v2/homework_results',
                json=params)
            print('Вы сдали!')
        else:
            assert False, 'Данный email не был найден в базе.'
    

def func_dict(kwargs, dict_vals):
    for i in dict_vals:
        if i in kwargs:
            dict_vals[i] = func_is_digit(kwargs[i])
        else:
            pass
    return dict_vals
    
    
def func_is_digit(arg):
    if isinstance(arg, tuple):
        return arg
    elif isinstance(arg,str): 
        if arg.isdigit():
            return int(arg)
        try:
            return float(arg)
        except ValueError:
            return arg
    return arg 


def decorator_check_hw(func):
  def inner_func(**kwargs):
    try:
      return func(**kwargs)
    except ValueError:
      assert False, f'Убедитесь, что все ваши нейронки имеют разные названия. В случае данной ошибки сбросьте ноутбук и запустите ячейки по новой с разными названиями нейронок.'
    except KeyError as kr:
        assert False, f'Нейронки {kr} нет. Перепроверьте название.' 
    except FileNotFoundError as err:
        assert False, f'Загрузите выборки для базы  \'{str(err)[57:-5]}\'.'
  return inner_func



@decorator_check_hw
def check_hw(**kwargs):
    all_mistakes = []
    for i in range(len(kwargs['datasets'])):
        if isinstance(kwargs[f'нейронка_{i+1}'].input_shape, tuple):
            len_shape = 1
        elif isinstance(kwargs[f'нейронка_{i+1}'].input_shape, list):
            len_shape = len(kwargs[f'нейронка_{i+1}'].input_shape)
        if len_shape == 1:
            x_test = np.load(f'test_sets/{kwargs["datasets"][i]}.npy')
        elif len_shape == 3:
            x_test = []
            for num in range(3):
                curr_x_test = np.load(f'test_sets/{kwargs["datasets"][i]}{num}.npy')
                x_test.append(curr_x_test)
        
        y_test = np.load(f'test_sets/{kwargs["datasets"][i]}_метки.npy')
        accur = kwargs[f'нейронка_{i+1}'].evaluate(x_test, y_test, verbose=0)[1]
        if accur>kwargs["accuracies"][i]:
            pass
        else:
            all_mistakes.append(f'Точность {i+1}-ой задачи - {round(accur*100,2)}%, когда требовалось минимально {100*kwargs["accuracies"][i]}%')
    return all_mistakes

def test_hw(**kwargs):
    accuracies = [[0.6, 0.85],[0.1,0.1,0.1],[0.1,0.1,0.1]]
    datasets = [['авто-3', 'майонез'], ['вакансии', 'tesla', 'симптомы_заболеваний'],['умный_дом','трейдинг', 'губы']]
    temporary_var = {'datasets':datasets[kwargs['lesson']],'accuracies':accuracies[kwargs['lesson']]}
    kwargs.update(temporary_var)
    result = check_hw(**kwargs)
    clear_output()
    for i in tqdm_notebook(range(len(accuracies[kwargs['lesson']]) - len(result)), desc='HW', total = len(accuracies[kwargs['lesson']])):
        time.sleep(0.5)
    for i in result:
        print(i)

    if len(result)==0:
        return True
    return False
    




def post_homework(username, hw_id):

  # Зачет ДЗ
  homework_id = hw_id
  curator_id = 93854
  r = requests.post('https://aiu-hw-provider.herokuapp.com/api/', data={'email': f'{username}', 'homework_id': f'{homework_id}'})
  return r.json()['homework_assignment']
    

def status_message(res):
    if res['answer'] == -1:
        raise NameError('Указанный пользователь не найден.')
    elif res['answer'] == -2:
        print('Домашняя работа уже принята.')
    elif res['answer'] == -3:
        raise Exception('Непредвиденная ошибка, обратитесь к куратору.')
    elif res['answer'] == 1:
        print('Всё верно! Работа принята.')


def test_hw_1(**kwargs):
    homework_id = 1633    
    kwargs.update({'lesson':0})
    if test_hw(**kwargs):
        res = post_homework(kwargs['логин'], homework_id)
        status_message(res)
    
def test_hw_2(**kwargs):
    homework_id = 1634    
    kwargs.update({'lesson':1})
    if test_hw(**kwargs):
        res = post_homework(kwargs['логин'], homework_id)
        status_message(res)
    
def test_hw_3(**kwargs):
    homework_id = 1635    
    kwargs.update({'lesson':2})
    if test_hw(**kwargs):
        res = post_homework(kwargs['логин'], homework_id)
        status_message(res)




