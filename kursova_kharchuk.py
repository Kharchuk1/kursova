 # курсова роботи
import math 
import numpy as np
import matplotlib.pyplot as plt

calls_of_func=0
left_gold_section=0.382
right_gold_section=0.618

# функція Розенброка
def func_rosen(x):
    global calls_of_func
    calls_of_func += 1
    return (1-x[0])**2+100*(x[1]-x[0]**2)**2

# похідна за методом центральних різниць
def derivative_center(x):
    f_deriv_1 = (func_rosen([x[0] + derivative_step, x[1]]) - func_rosen([x[0] - derivative_step, x[1]])) / (2 * derivative_step)
    f_deriv_2 = (func_rosen([x[0], x[1] + derivative_step]) - func_rosen([x[0], x[1] - derivative_step])) / (2 * derivative_step)
    
    return [f_deriv_1, f_deriv_2]

# похідна за методом правої кінцевої різниці
def derivative_right(x, f):
    f_deriv_1 = (func_rosen([x[0] + derivative_step, x[1]]) - f) / (2 * derivative_step)
    f_deriv_2 = (func_rosen([x[0], x[1] + derivative_step]) - f) / (2 * derivative_step)
    
    return [f_deriv_1, f_deriv_2]


# похідна за методом лівої кінцевої різниці
def derivative_left(x, f):
    f_deriv_1 = (func_rosen([x[0] + derivative_step, x[1]]) - func_rosen(x)) / (derivative_step)
    f_deriv_2 = (func_rosen([x[0], x[1] + derivative_step]) - func_rosen(x)) / (derivative_step)
    return [f_deriv_1, f_deriv_2]


# градієнт
def gradient(der):
    grad = np.array([])
    grad = np.append(grad, der) 
    return grad

# норма функції (а ще критерій закінчення 2)
def norma(mas):
    result = 0
    for i in mas:
        result += i ** 2
    return math.sqrt(result)

# print(f'\nnorma(grad) = {norma(gradient(x0))}')

# критерій закінчення 1
def kriteriy_1(x0, x1, f0, f1):
    res=[]
    for i in range(len(x1)):
        res.append(x1[i]-x0[i])
    res1=norma(res)/norma(x0)
    res2=abs(f0-f1)
    result=np.array([res1,res2])
    return result

def calculate_route(A, grad):
    S = -np.dot(A, grad)
    return S

# підрахунок наступної точки
def calculate_x(x0, lam, s):
    lam_S = np.dot(lam, s)
    result = x0 + lam_S
    return result

# алгоритм Свена
def svenn(x0, alpha, lmb, s, f0):
    delta = alpha*(norma(x0)/norma(s))

    if f0 < func_rosen(calculate_x(x0, lmb + delta, s)): 
        delta = -delta
    x1 = lmb + delta
    f1 = func_rosen(calculate_x(x0, x1, s))
    while f1 < f0:
        delta *= 2
        lmb = x1
        x1 = lmb + delta
        f0 = f1
        f1 = func_rosen(calculate_x(x0, x1, s))
    a = lmb + delta / 2 
    b = lmb - delta / 2
    f0 = func_rosen(calculate_x(x0, lmb, s))
    f1 = func_rosen(calculate_x(x0, b, s))

    if f0 < f1:
        if a < b:
            return [a, b]
        else:
            return [b, a]
    elif f1 < f0:
        if lmb < x1:
            return [lmb, x1]
        else:
            return [x1, lmb]
    else:
        if lmb < b:
            return [lmb, b]
        else:
            return [b, lmb]


# метод золотого перерізу
def gold_section(a, b, eps, x0, s):
    l = b - a
    x1 = a + l * left_gold_section
    x2 = a + l * right_gold_section
    while l > eps:
        if func_rosen(calculate_x(x0, x1, s)) < func_rosen(calculate_x(x0, x2, s)):
            b = x2
            x2 = x1
            l = b - a
            x1 = a + l * left_gold_section
        else:
            a = x1
            x1 = x2
            l = b - a
            x2 = a + l * right_gold_section
    return [a, b]


# дск-пауела
def dscPowell(x0, a, b, s, eps):
    xmin = (a + b) / 2

    f1 = func_rosen(calculate_x(x0, a, s))
    f2 = func_rosen(calculate_x(x0, xmin, s))
    f3 = func_rosen(calculate_x(x0, b, s))
    xApprox = xmin + ((b - xmin) * (f1 - f3)) / (2 * (f1 - 2 * f2 + f3))

    while (abs(xmin - xApprox) > eps or abs(func_rosen(calculate_x(x0, xmin, s)) - func_rosen(calculate_x(x0, xApprox, s)))> eps):
        if xApprox < xmin:
            b = xmin
        else:
            a = xmin
        xmin = xApprox
        funcRes = [
            func_rosen(calculate_x(x0, a, s)),
            func_rosen(calculate_x(x0, xmin, s)),
            func_rosen(calculate_x(x0, b, s)),
        ]
        a1 = (funcRes[1] - funcRes[0]) / (xmin - a)
        a2 = ((funcRes[2] - funcRes[0]) / (b - a) - a1) / (b - xmin)
        xApprox = (a + xmin) / 2 - a1 / (2 * a2)
    return xmin

# lam=dscPowell(x0, 0.1, 0, s, 0.00001)
# print(f'lambda={lam}')
# print(f'calculate_x = {calculate_x(x0, lam, A)}')

def calculate_A(A, x0, x1, grad0, grad1):
    I=np.eye(len(x0))

    deltag = np.array(np.subtract(grad1, grad0))[np.newaxis] # Різниця g
    deltax = np.subtract(x1, x0) # Різниця x
    
    deltax = np.array(deltax)[np.newaxis] # Розширюємо deltax
    deltaxT = np.array(deltax).T # Транспонуємо detlax
    deltagT = np.array(deltag).T# Транспонуємо detltag

    deltax, deltaxT = deltaxT, deltax
    deltag, deltagT = deltagT, deltag

    # Шукаємо перший піддріб 
    first = np.dot(deltax, deltagT) # Перемножуємо
    second = np.dot(deltagT, deltax)
    res1 = first / second
    temp1 = I - res1

    # Шукаємо другий піддріб
    first = np.dot(deltag, deltaxT) # Перемножуємо
    second = np.dot(deltagT, deltax)
    res2 = first / second
    temp2 = I - res2

    # Шукаємо перший дріб 
    first = np.dot(temp1, A)
    resOne = np.dot(first, temp2)

    # Шукаємо другий дріб 
    first = np.dot(deltax, deltaxT) # Перемножуємо
    second = np.dot(deltagT, deltax)
    resTwo = first / second
    return resOne + resTwo # Вертаємо кортеж з результатами

# print(f'calculate_A={calculate_A(A, x0, x1)}')

def summary(calls_of_func, iteration, restart, xmin, fmin):
    print('Кінець програми')
    print(f'Було виконано {iteration} ітерацій')
    print(f'Було {restart} рестартів')
    print(f'Точка мінімуму {xmin}')
    print(f'Мінімум функції {fmin}')
    print(f'Функцію було викликано {calls_of_func} разів')

def print_iteration_results(x0, iteration, calls_of_func):
    print(f'Нова точка: {x0}')
    print(f'Ітерація: {iteration}')
    print(f'виклик: {calls_of_func}')

def draw(x_list, y_list):
    plt.figure(figsize=(16,8))
    plt.plot(x_list, y_list, 'bo-')
    plt.title('Траєкторія пошуку точки', size=18)
    plt.grid()
    plt.show()


# метод бройдена
def bfgs():
    A = np.eye(2)

    restart = 0
    iteration = 0

    x_list=[]
    y_list=[]

    x_0 = float(input('Введіть x0:'))
    x_1 = float(input('Введіть y0:'))

    x0 = np.array((x_0, x_1))

    alpha_sven = float(input('Введіть альфу для алгоритма Свена:'))
    lam0 = float(input('Введіть початкову лямбду для алгоритма Свена:'))
    e_mop = float(input('Введіть точність для МОП:'))
    e_bfgs = float(input('Введіть точність для BFGS:'))

    print(x0)
    print("------------------------------------")
    while True:
        x_list.append(x0[0])
        y_list.append(x0[1])

        f0 = func_rosen(x0)

        # der = derivative_left(x0, f0)
        der = derivative_center(x0)
        grad0 = gradient(der)
        s = calculate_route(A, grad0)

        # if norma(grad0) <= eps1:  # Умова 1 закінчення пошуку
        #    summary(calls_of_func, iteration, restart, x0, func_rosen(x0))
        #    draw(x_list, y_list)
        #    break

        sven = svenn(x0, alpha_sven, lam0, s, f0)
        a = sven[0]
        b = sven[1]

        # lam = dscPowell(x0, a, b, s, eps2)
        gold = gold_section(a, b, e_mop, x0, s)
        lam = (gold[0] + gold[1])/2
        
        if lam <= e_bfgs:
            A = np.eye(len(x0))
            restart += 1
            print('Restart')

        x1 = calculate_x(x0, lam, s)

        # der1 = derivative_left(x1, func_rosen(x1))
        der1 = derivative_center(x1)
        grad1 = gradient(der1)

        A = calculate_A(A, x0, x1, grad0, grad1) 

        if kriteriy_1(x0, x1, f0, func_rosen(x1))[0] <= e_bfgs or kriteriy_1(x0, x1, f0, func_rosen(x1))[1] <= e_bfgs: # Умова 2 закінчення пошуку
            print(summary(calls_of_func, iteration, restart, x1, func_rosen(x1)))
            draw(x_list, y_list)
            break
                       
        x0 = x1

        print_iteration_results(x0, iteration, calls_of_func)
        print("\n")

        iteration += 1

if __name__ == '__main__':
    derivative_step = float(input('Введіть крок для похідних:'))
    bfgs()
