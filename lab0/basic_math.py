import numpy as np
from scipy.optimize import minimize_scalar


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError()
    res_mat = [[0 for j in range(len(matrix_a))] for i in range(len(matrix_b[0]))] #np.empty((len(matrix_a),len(matrix_b[0])), dtype="float32")
    
    for i in range(len(matrix_a)):
      for i1 in range(len(matrix_b[0])):
        res_mat[i][i1] = int(sum(matrix_a[i][i2]*matrix_b[i2][i1] for i2 in range(len(matrix_a[0]))))

    return res_mat



def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """

    a_1_arr = np.array([float(i) for i in  a_1.split()])
    a_2_arr = np.array([float(i) for i in  a_2.split()])

    func_1 = lambda x: a_1_arr[0] * np.clip(x, -1e10, 1e10) ** 2 + a_1_arr[1] * np.clip(x, -1e10, 1e10) + a_1_arr[2]
    func_2 = lambda x: a_2_arr[0] * np.clip(x, -1e10, 1e10) ** 2 + a_2_arr[1] * np.clip(x, -1e10, 1e10) + a_2_arr[2]

    
    min_1 = minimize_scalar(func_1).x
    min_2 =  minimize_scalar(func_2).x

    a =  a_1_arr[0] - a_2_arr[0]
    b =  a_1_arr[1] - a_2_arr[1]
    c = a_1_arr[2] - a_2_arr[2]

    res = []

    if a == 0 and b == 0:
        if c == 0:
            return None  # Бесконечно много решений
        else:
            return []  # Нет решений
    elif a == 0:
        x = -c / b
        res.append((x, func_1(x)))
    else:
        D = b ** 2 - 4 * a * c
        if D > 0:
            x1 = (-b + D**0.5) / (2 * a)
            x2 = (-b - D**0.5) / (2 * a)
            res.append((x1, func_1(x1)))
            res.append((x2, func_1(x2)))
        elif D == 0:
            x = -b / (2 * a)
            res.append((x, func_1(x)))

    return res if res else []


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    x_e = sum(x) / len (x)
    sigma = (sum((x_i - x_e) ** 2 for x_i in x) / len(x)) ** 0.5

    m_3 = sum((x_i - x_e) ** 3 for x_i in x) / len(x)
    A_3 = m_3 / (sigma ** 3)

    return round(A_3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    x_e = sum(x) / len (x)
    sigma = (sum((x_i - x_e) ** 2 for x_i in x) / len(x)) ** 0.5

    m_4 = sum((x_i - x_e) ** 4 for x_i in x) / len(x)
    E_4 = m_4 / (sigma ** 4) -3

    return round(E_4, 2)
