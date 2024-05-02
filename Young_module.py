import pandas as pd
import numpy as np
from sklearn import  linear_model
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import math
import os

def find_coefs(Xold,yold):
    """

    :param Xold: массив перемещений
    :param yold: массив сил
    :return: возвращает наклон графика, квадратичное отклонение, массив предсказанных перемещений
    """
    global accuracy
    global length
    X=Xold
    y=yold
    armax = y.argmax()
    Xmax = X[armax]
    X = X[0:armax]
    y = y[0:armax]

    mse = 10
    m = 10
    while mse ** 0.5 / m > accuracy and X.max() > length * Xmax:
        # Обучаем модель линейной регрессии
        model = linear_model.LinearRegression()
        model.fit(X, y)

        # Получаем коэффициенты прямой
        m = model.coef_[0]
        c = model.intercept_

        # Вычисляем прогнозные значения y на основе модели
        y_pred = model.predict(X)

        # Вычисляем среднеквадратичную ошибку
        mse = mean_squared_error(y, y_pred)
        X = X[0:int(len(y) * 0.9)]
        y = y[0:int(len(y) * 0.9)]

    return m,mse,model.predict(X)

accuracy=0.05 # минимальная относительная погрешность
length=0.6 # минимальная доля до максимума


#создание дириктории для записи результатов
if not os.path.isdir("./results"):
    os.mkdir('results')
else:
    directory_count=0
    for root, dirs, files in os.walk('.'):  # начинаем обход всех директорий начиная с текущей
        for dir in dirs:
            if 'results' in dir:
                directory_count += 1
    os.rename('results',f'results{directory_count}')
    os.mkdir('results')

"""
Чтение характеристик образцов из файла header.txt
Структура файла: 
каждое испытание - это одна строчка
Структура строчка:
number type S L include comments
number - номер испытания
type - тип образца (продольный, поперечный и тд)
S - площадь поперечного сечения образца
L - длина образца
include(bool) - определяет, включается ли данный образец в вчисление средних значений модуля Юнга
comments - словнесные комментарии к испытанию
"""
data={}
with open('data/header.txt','r',encoding='UTF-8') as f:
    for line in f.readlines():
        match line.split():
            case [number,type,S,l,include,*comments]:
                data[int(number)]={'type':type,'include':bool(include),'S':float(S),'l':float(l),'comments':' '.join(comments)}

                info=pd.read_csv(f"data/{int(number)}.Stop.csv",sep=';',encoding='cp1251',decimal=',',dtype=np.float64)
                data[int(number)]['u']=info['Положение(ElectroPuls:Position) (mm)']-info['Положение(ElectroPuls:Position) (mm)'][0]
                data[int(number)]['f'] = info['Нагрузка(ElectroPuls:Нагрузка) (kgf)']-info['Нагрузка(ElectroPuls:Нагрузка) (kgf)'][0]
            case _:
                raise Exception('wrong header file')


#нахождение модулей Юнга и погрешностей
for var in data:
    m,mse,y_pred=find_coefs(data[var]['u'].values.reshape(-1,1),data[var]['f'].values)
    E= m * data[var]['l'] / data[var]['S']* 9.8
    error=mse**0.5 * data[var]['l'] / data[var]['S']* 9.8

    data[var]['E']=round(E, -int(math.floor(math.log10(abs(error)))+1))
    data[var]['error']=round(error, -int(math.floor(math.log10(abs(error)))))
    data[var]['pred']=y_pred


E_type={}
error_type={}
count_type={}

#запись результатов и отрисовка графиков
count_all=0
E_all=0
E_all_error=0
with open('results/results.txt','w') as f:
    for var in data:
        E_all+=data[var]['E']*data[var]['include']
        E_all_error+=data[var]['error']*data[var]['include']
        count_all+=1*data[var]['include']

        if data[var]['type'] in E_type.keys():
            E_type[data[var]['type']]+=data[var]['E']*data[var]['include']
            error_type[data[var]['type']]+=data[var]['error']**2*data[var]['include']
            count_type[data[var]['type']]+=1*data[var]['include']
        else:
            E_type[data[var]['type']] = data[var]['E'] * data[var]['include']
            error_type[data[var]['type']] = data[var]['error']**2 * data[var]['include']
            count_type[data[var]['type']] = 1 * data[var]['include']
        f.write(f'Эксперимент{var} E = {data[var]['E']} error = {data[var]['error']} comments: {data[var]['comments']} \n')

        plt.plot(data[var]['u'].values,data[var]['f'].values,label='experiment data')
        plt.plot(data[var]['u'].values[0:len(data[var]['pred'])], data[var]['pred'],label='predicted')
        plt.grid(True)
        plt.savefig(f'results/graphic.png', format='png')
        plt.close()
    for var in E_type:
        error=math.sqrt(error_type[var]/count_type[var])
        error=round(error, -int(math.floor(math.log10(abs(error)))))
        E=round(E_type[var]/count_type[var], -int(math.floor(math.log10(abs(error)))))
        f.write(f'{var} E = {E} error = {error} \n')

    error = math.sqrt(E_all_error / count_all)
    error = round(error, -int(math.floor(math.log10(abs(error)))))
    E = round(E_all / count_all, -int(math.floor(math.log10(abs(error)))))
    f.write(f'Средний по всем E = {E} error = {error}')







