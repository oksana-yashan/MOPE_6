import random
import numpy as np
import math
from functools import reduce
from itertools import compress
import scipy
from scipy.stats import f,t


x1min, x2min, x3min = -25, -20, -20
x1max, x2max, x3max = 75, 40, -15

x_min = (x1min + x2min + x3min) / 3  # mean of x1min, x2min, x3min
x_max = (x1max + x2max + x3max) / 3
x0_i = [np.average(x1min+x1max),np.average(x2min+x2max),np.average(x3min+x3max)]

truth = [True]*11
y_min = round(200 + x_min)
y_max = round(200 + x_max)
m = 3
N = 8



# from Lab3 without interaction
counter = 0
with_interaction = False
check = True



while (check):  # from Lab3
    matrix = np.ndarray(shape=(4, 7), dtype=float)
    matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0] = 1, 1, 1, 1
    matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1] = -1, -1, 1, 1
    matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2] = -1, 1, -1, 1
    matrix[0][3], matrix[1][3], matrix[2][3], matrix[3][3] = -1, 1, 1, -1

    matrix_n = np.ndarray(shape=(4, 6), dtype=float)
    matrix_n[0][0], matrix_n[1][0], matrix_n[2][0], matrix_n[3][0] = x1min, x1min, x1max, x1max
    matrix_n[0][1], matrix_n[1][1], matrix_n[2][1], matrix_n[3][1] = x2min, x2max, x1min, x2max
    matrix_n[0][2], matrix_n[1][2], matrix_n[2][2], matrix_n[3][2] = x3min, x3max, x3max, x3min

    mY_list = []
    for i in range(4):
        for j in range(3, 6):
            r = np.random.randint(y_min, y_max)
            matrix_n[i][j], matrix[i][j + 1] = r, r
        mY_list.append(((matrix_n[i][3] + matrix_n[i][4] + matrix_n[i][5]) / m).__round__(4))
    mx1 = np.sum(matrix_n, axis=0)[0] / 4
    mx2 = np.sum(matrix_n, axis=0)[1] / 4
    mx3 = np.sum(matrix_n, axis=0)[2] / 4
    my = (sum(mY_list) / len(mY_list)).__round__(2)

    a1 = (matrix_n[0][0] * mY_list[0] + matrix_n[1][0] * mY_list[1] + matrix_n[2][0] * mY_list[2] + matrix_n[3][0] *
          mY_list[3]) / 4
    a2 = (matrix_n[0][1] * mY_list[0] + matrix_n[1][1] * mY_list[1] + matrix_n[2][1] * mY_list[2] + matrix_n[3][1] *
          mY_list[3]) / 4
    a3 = (matrix_n[0][2] * mY_list[0] + matrix_n[1][2] * mY_list[1] + matrix_n[2][2] * mY_list[2] + matrix_n[3][2] *
          mY_list[3]) / 4

    a11 = (matrix_n[0][0] ** 2 + matrix_n[1][0] ** 2 + matrix_n[2][0] ** 2 + matrix_n[3][0] ** 2) / 4
    a22 = (matrix_n[0][1] ** 2 + matrix_n[1][1] ** 2 + matrix_n[2][1] ** 2 + matrix_n[3][1] ** 2) / 4
    a33 = (matrix_n[0][2] ** 2 + matrix_n[1][2] ** 2 + matrix_n[2][2] ** 2 + matrix_n[3][2] ** 2) / 4

    a12 = a21 = (matrix_n[0][0] * matrix_n[0][1] + matrix_n[1][0] * matrix_n[1][1] + matrix_n[2][0] * matrix_n[2][1] +
                 matrix_n[3][0] * matrix_n[3][1]) / 4
    a13 = a31 = (matrix_n[0][0] * matrix_n[0][2] + matrix_n[1][0] * matrix_n[1][2] + matrix_n[2][0] * matrix_n[2][2] +
                 matrix_n[3][0] * matrix_n[3][2]) / 4
    a23 = a32 = (matrix_n[0][1] * matrix_n[0][2] + matrix_n[1][1] * matrix_n[1][2] + matrix_n[2][1] * matrix_n[2][2] +
                 matrix_n[3][1] * matrix_n[3][2]) / 4

    b0 = np.linalg.det(
        np.array([[my, mx1, mx2, mx3], [a1, a11, a12, a13], [a2, a12, a22, a32], [a3, a13, a23, a33]])) / np.linalg.det(
        np.array([[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a32], [mx3, a13, a23, a33]]))
    b1 = np.linalg.det(
        np.array([[1, my, mx2, mx3], [mx1, a1, a12, a13], [mx2, a2, a22, a32], [mx3, a3, a23, a33]])) / np.linalg.det(
        np.array([[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a32], [mx3, a13, a23, a33]]))
    b2 = np.linalg.det(
        np.array([[1, mx1, my, mx3], [mx1, a11, a1, a13], [mx2, a12, a2, a32], [mx3, a13, a3, a33]])) / np.linalg.det(
        np.array([[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a32], [mx3, a13, a23, a33]]))
    b3 = np.linalg.det(
        np.array([[1, mx1, mx2, my], [mx1, a11, a12, a1], [mx2, a12, a22, a2], [mx3, a13, a23, a3]])) / np.linalg.det(
        np.array([[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a32], [mx3, a13, a23, a33]]))
    print("    Матриця планування")
    print("   x1      x2     x3      y1      y2      y3     ")
    for i in range(3):
        for j in range(6):
            print("{:>6.1f}".format(matrix_n[i][j]), end="  ")
        print("\t")
    print("\n", "y = %.2f + %.2f * x1 + %.2f * x2+ %.2f * x3" % (b0, b1, b2, b3))
    print("\nПеревірка:")
    print((b0 + b1 * matrix_n[0][0] + b2 * matrix_n[0][1] + b3 * matrix_n[0][2]).__round__(3), "   ",
          (b0 + b1 * matrix_n[1][0] + b2 * matrix_n[1][1] + b3 * matrix_n[1][2]).__round__(3), "   ",
          (b0 + b1 * matrix_n[2][0] + b2 * matrix_n[2][1] + b3 * matrix_n[2][2]).__round__(3), "   ",
          (b0 + b1 * matrix_n[3][0] + b2 * matrix_n[3][1] + b3 * +matrix_n[3][2]).__round__(3))

    print(mY_list)

    # Перевірка за Кохреном:
    s2_y1 = ((matrix[0][4] - mY_list[0]) ** 2 + (matrix[0][5] - mY_list[0]) ** 2 + (matrix[0][6] - mY_list[0]) ** 2) / 3
    s2_y2 = ((matrix[1][4] - mY_list[1]) ** 2 + (matrix[1][5] - mY_list[1]) ** 2 + (matrix[1][6] - mY_list[1]) ** 2) / 3
    s2_y3 = ((matrix[2][4] - mY_list[2]) ** 2 + (matrix[2][5] - mY_list[2]) ** 2 + (matrix[2][6] - mY_list[2]) ** 2) / 3
    s2_y4 = ((matrix[3][4] - mY_list[3]) ** 2 + (matrix[3][5] - mY_list[3]) ** 2 + (matrix[3][6] - mY_list[3]) ** 2) / 3

    f1 = m - 1
    f2 = N
    p = 0.95
    q = 1 - p
    Gp = max(s2_y1, s2_y2, s2_y3, s2_y4) / (s2_y1 + s2_y2 + s2_y3 + s2_y4)

    def cohren_value(f2, f1, q):
        f2 += 1
        partResult1 = q / (f2 - 1)
        params = [partResult1, f1, (f2 - 1 - 1) * f2]
        fisher = scipy.stats.f.isf(*params)
        result = fisher / (fisher + (f2 - 1 - 1))
        return result.__round__(3)


    Gt = cohren_value(f2, f1, q)
    if (Gp < Gt):
        print(" Отже, дисперсія однорідна")
        check = False

    else:
        print(" Дисперсія неоднорідна -->  m+1")
        m += 1

# Критерій Стьюдента
s2_b = (s2_y1 + s2_y2 + s2_y3 + s2_y4) / 4
s2_bb = s2_b / (4 * m)
s_bb = np.sqrt(s2_bb)

bb0 = (mY_list[0] * matrix[0][0] + mY_list[1] * matrix[1][0] + mY_list[2] * matrix[2][0] + mY_list[3] * matrix[3][
    0]) / N
bb1 = (mY_list[0] * matrix[0][1] + mY_list[1] * matrix[1][1] + mY_list[2] * matrix[2][1] + mY_list[3] * matrix[3][
    1]) / N
bb2 = (mY_list[0] * matrix[0][2] + mY_list[1] * matrix[1][2] + mY_list[2] * matrix[2][2] + mY_list[3] * matrix[3][
    2]) / N
bb3 = (mY_list[0] * matrix[0][3] + mY_list[1] * matrix[1][3] + mY_list[2] * matrix[2][3] + mY_list[3] * matrix[3][
    3]) / N

t = [abs(bb0) / s_bb, abs(bb1) / s_bb, abs(bb2) / s_bb, abs(bb3) / s_bb]

f3 = (m - 1) * N  # t_t = 2.306  # для значення f3 = 8, t табличне = 2,306
t_t = scipy.stats.t.ppf((1 + (1 - q)) / 2, f3)
print("\nt табличне:", t_t)

if t[0] < t_t:
    b0 = 0
    print("t0<t_t; b0=0")
if t[1] < t_t:
    b1 = 0
    print("t1<t_t; b1=0")
if t[2] < t_t:
    b2 = 0
    print("t2<t_t; b2=0")
if t[3] < t_t:
    b3 = 0
    print("t3<t_t; b3=0")

print("\n", "y = %.2f + %.2f * x1 + %.2f * x2+ %.2f * x3" % (b0, b1, b2, b3))
y1_exp = b0 + b1 * matrix_n[0][0] + b2 * matrix_n[0][1] + b3 * matrix_n[0][2]
y2_exp = b0 + b1 * matrix_n[1][0] + b2 * matrix_n[1][1] + b3 * matrix_n[1][2]
y3_exp = b0 + b1 * matrix_n[2][0] + b2 * matrix_n[2][1] + b3 * matrix_n[2][2]
y4_exp = b0 + b1 * matrix_n[3][0] + b2 * matrix_n[3][1] + b3 * matrix_n[3][2]

print(f"y1_exp = {b0:.2f}{b1:+.2f}*x11{b2:+.2f}*x12{b3:+.2f}*x13 "
      f"= {y1_exp:.2f}")
print(f"y2_exp = {b0:.2f}{b1:+.2f}*x21{b2:+.2f}*x22{b3:+.2f}*x23"
      f" = {y2_exp:.2f}")
print(f"y3_exp = {b0:.2f}{b1:+.2f}*x31{b2:+.2f}*x32{b3:+.2f}*x33 "
      f"= {y3_exp:.2f}")
print(f"y4_exp = {b0:.2f}{b1:+.2f}*x41{b2:+.2f}*x42{b3:+.2f}*x43"
      f" = {y4_exp:.2f}")

# Критерій Фішера
d = 2
f4 = N - d
s2_ad = ((y1_exp - mY_list[0]) ** 2 + (y2_exp - mY_list[1]) ** 2 + (y3_exp - mY_list[2]) ** 2 + (
            y4_exp - mY_list[3]) ** 2) / (m / N - d)

Fp = s2_ad / s2_b
Ft = scipy.stats.f.ppf(1 - q, f4, f3)
print("\nFp:", Fp)
print("Ft:", Ft)
if Fp < Ft:
    print("Рівняння регресії не адекватно оригіналу при q = 0,05",'\n\n')
    with_interaction = True
    print("Рівняння регресії з врахуванням ефекту взаємодії:")
else:
    print("Рівняння регресії адекватно оригіналу при q = 0,05")





# Equation with intersection(Lab4)
while (with_interaction):
    m = 3
    N = 8
    x0 = [1 for i in range(N)]

    #                 x1  x2   x3  x12 x13 x23 x123
    norm_x_table = [[-1, -1, -1, +1, +1, +1, -1],
                     [-1, +1, +1, -1, -1, +1, -1],
                     [+1, -1, +1, -1, +1, -1, -1],
                     [+1, +1, -1, +1, -1, -1, -1],

                     [-1, -1, +1, +1, -1, -1, +1],
                     [-1, +1, -1, -1, +1, -1, +1],
                     [+1, -1, -1, -1, -1, +1, +1],
                     [+1, +1, +1, +1, +1, +1, +1]]

#                   1     2      3       12              13              23               123
    x_table = [[x1min, x2min, x3min, x1min * x2min, x1min * x3min, x2min * x3min, x1min * x2min * x3min],
                [x1min, x2max, x3max, x1min * x2max, x1min * x3max, x2max * x3max, x1min * x2max * x3max],
                [x1max, x2min, x3max, x1max * x2min, x1max * x3max, x2min * x3max, x1max * x2min * x3max],
                [x1max, x2max, x3min, x1max * x2max, x1max * x3min, x2max * x3min, x1max * x2max * x3min],

                [x1min, x2min, x3max, x1min * x2min, x1min * x3max, x2min * x3max, x1min * x2min * x3max],
                [x1min, x2max, x3min, x1min * x2max, x1min * x3min, x2max * x3min, x1min * x2max * x3min],
                [x1max, x2min, x3min, x1max * x2min, x1max * x3min, x2min * x3min, x1max * x2min * x3min],
                [x1max, x2max, x3max, x1max * x2max, x1max * x3max, x2max * x3max, x1max * x2max * x3max]]

    y_arr = [[random.randint(y_min, y_max) for j in range(m)] for i in range(N)]  # i rows and j columns
    print(y_arr)

    # arrays with x1(i), x2(i),x3(i)
    x1i = np.array([x_table[i][0] for i in range(8)])
    x2i = np.array([x_table[i][1] for i in range(8)])
    x3i = np.array([x_table[i][2] for i in range(8)])
    yi = np.array([np.average(i) for i in y_arr])  # average for each i row in y_arr


    def m_ij(*arrays):
        return np.average(reduce(lambda el_1, el_2: el_1 + el_2, arrays))  # reduce: sums all el in given arrays


    coefs = [[N, m_ij(x1i), m_ij(x2i), m_ij(x3i), m_ij(x1i * x2i), m_ij(x1i * x3i), m_ij(x2i * x3i), m_ij(x1i * x2i * x3i)],
             [m_ij(x1i), m_ij(x1i ** 2), m_ij(x1i * x2i), m_ij(x1i * x3i), m_ij(x1i ** 2 * x2i), m_ij(x1i ** 2 * x3i),
              m_ij(x1i * x2i * x3i), m_ij(x1i ** 2 * x2i * x3i)],
             [m_ij(x2i), m_ij(x1i * x2i), m_ij(x2i ** 2), m_ij(x2i * x3i), m_ij(x1i * x2i ** 2), m_ij(x1i * x2i * x3i),
              m_ij(x2i ** 2 * x3i), m_ij(x1i * x2i ** 2 * x3i)],
             [m_ij(x3i), m_ij(x1i * x3i), m_ij(x2i * x3i), m_ij(x3i ** 2), m_ij(x1i * x2i * x3i), m_ij(x1i * x3i ** 2),
              m_ij(x2i * x3i ** 2), m_ij(x1i * x2i * x3i ** 2)],

             [m_ij(x1i * x2i), m_ij(x1i ** 2 * x2i), m_ij(x1i * x2i ** 2), m_ij(x1i * x2i * x3i), m_ij(x1i ** 2 * x2i ** 2),
              m_ij(x1i ** 2 * x2i * x3i), m_ij(x1i * x2i ** 2 * x3i), m_ij(x1i ** 2 * x2i ** 2 * x3i)],
             [m_ij(x1i * x3i), m_ij(x1i ** 2 * x3i), m_ij(x1i * x2i * x3i), m_ij(x1i * x3i ** 2),
              m_ij(x1i ** 2 * x2i * x3i), m_ij(x1i ** 2 * x3i ** 2), m_ij(x1i * x2i * x3i ** 2),
              m_ij(x1i ** 2 * x2i * x3i ** 2)],
             [m_ij(x2i * x3i), m_ij(x1i * x2i * x3i), m_ij(x2i ** 2 * x3i), m_ij(x2i * x3i ** 2),
              m_ij(x1i * x2i ** 2 * x3i), m_ij(x1i * x2i * x3i ** 2), m_ij(x2i ** 2 * x3i ** 2),
              m_ij(x1i * x2i ** 2 * x3i ** 2)],

             [m_ij(x1i * x2i * x3i), m_ij(x1i ** 2 * x2i * x3i), m_ij(x1i * x2i ** 2 * x3i), m_ij(x1i * x2i * x3i ** 2),
              m_ij(x1i ** 2 * x2i ** 2 * x3i), m_ij(x1i ** 2 * x2i * x3i ** 2), m_ij(x1i * x2i ** 2 * x3i ** 2),
              m_ij(x1i ** 2 * x2i ** 2 * x3i ** 2)]]

    free_vals = [m_ij(yi), m_ij(yi * x1i), m_ij(yi * x2i), m_ij(yi * x3i), m_ij(yi * x1i * x2i), m_ij(yi * x1i * x3i),
                 m_ij(yi * x2i * x3i), m_ij(yi * x1i * x2i * x3i)]

    # solution of system of equations(array)
    b_i = np.linalg.solve(coefs, free_vals)

    # just arrays from table
    nat_x1 = np.array([norm_x_table[i][0] for i in range(8)])
    nat_x2 = np.array([norm_x_table[i][1] for i in range(8)])
    nat_x3 = np.array([norm_x_table[i][2] for i in range(8)])


    norm_b_i = [m_ij(yi * 1), m_ij(yi * nat_x1), m_ij(yi * nat_x2), m_ij(yi * nat_x3),
                m_ij(yi * nat_x1 * nat_x2), m_ij(yi * nat_x1 * nat_x3), m_ij(yi * nat_x2 * nat_x3),
                m_ij(yi * nat_x1 * nat_x2 * nat_x3)]



    # main functions


    def theor_y(x_table, b_coef, importance):
        x_table = [list(compress(row, importance)) for row in x_table]  # update: if importance 0 - get rid of x(ij)
        b_coef = list(compress(b_coef, importance))  # update: if importance 0 - get rid of b
        print("X_table :\n", x_table)
        print("b-coeffs :\n",b_coef)
        y_vals = np.array([sum(map(lambda x, b: x * b, row, b_coef)) for row in x_table])
        return y_vals


    def student_criteria1(m, N, y_table, norm_x_table):
        print("\nЗа критерієм Стьюдента: m = {}, N = {} ".format(m, N))
        avg_variation = np.average(list(map(np.var, y_table)))  # var = mean(abs(y - y.mean())**2) in numpy
        y_avrg = np.array(list(map(np.average, y_table)))
        variation_beta_s = avg_variation / N / m
        deviation_beta_s = math.sqrt(variation_beta_s)

        x_i = np.array([[el[i] for el in norm_x_table] for i in range(len(norm_x_table))])
        coef_beta_s = np.array([round(np.average(y_avrg * x_i[i]), 3) for i in range(len(x_i))])
        print("Оцінки коефіцієнтів β(s): " + ", ".join(list(map(str, coef_beta_s))))
        t_i = np.array([abs(coef_beta_s[i]) / deviation_beta_s for i in range(len(coef_beta_s))])
        print("Коефіцієнти t:         " + ", ".join(list(map(lambda i: "{:.2f}".format(i), t_i))))
        f3 = (m - 1) * N
        q = 0.05

        t = get_student(f3, q)
        importance = [True if el>t else False for el in list(t_i)]
        print("f3 = {}; q = {}; tтабл = {}".format(f3, q, t))
        beta_i = ["β0", "β1", "β2", "β3", "β12", "β13", "β23", "β123"]
        updated_importance = [" - значимий" if i else " - незначимий" for i in importance]
        to_print = map(lambda x: x[0] + " " + x[1], zip(beta_i, updated_importance))
        x_i_names = list(compress(["", " x1", " x2", " x3", " x12", " x13", " x23", " x123"],
                                  importance))  # if importance 0 - get rid of it
        betas_to_print = list(compress(coef_beta_s, importance))
        print(*to_print, sep="; ")
        equation = " ".join(["".join(i) for i in zip(list(map(lambda x: "{:+.2f}".format(x), betas_to_print)), x_i_names)])
        print("Рівняння регресії без незначимих членів: y = " + equation)
        return importance


    def get_student(f3, q):
        return (abs(scipy.stats.t.ppf(q / 2, f3))).__round__(3)

    def get_fisher(f3, f4, q):
        return (abs(f.isf(q, f4, f3))).__round__(3)

    def cochran_criteria(m, N, y_table):
        print("За критерієм Кохрена: m = {}, N = {} ".format(m, N))
        y_variations = [np.var(i) for i in y_table]
        max_y_variation = max(y_variations)
        gp = max_y_variation / sum(y_variations)
        f1 = m - 1
        f2 = N
        p = 0.95
        q = 1 - p
        gt = cohren_value(f1, f2, q)
        print("Gp = {}; Gt = {}; f1 = {}; f2 = {}; q = {:.2f}".format(gp, gt, f1, f2, q))
        if gp < gt:
            print("Gp < Gt => дисперсії рівномірні")
            return True
        else:
            print("Gp > Gt => дисперсії нерівномірні")
            return False



    while not cochran_criteria(m, 4, y_arr):
        m += 1
        y_table = [[random.randint(y_min, y_max) for column in range(m)] for row in range(N)]
    print("Матриця планування:")
    labels = "   x1    x2    x3    x12    x13    x23   x123     y1    y2    y3"
    rows_table = [list(x_table[i]) + list(y_arr[i]) for i in range(N)]
    print(labels)
    print("\n".join([" ".join(map(lambda j: "{:<6}".format(j), rows_table[i])) for i in range(len(rows_table))]), "\n")

    norm_x_table_with_x0 = [[+1] + row for row in norm_x_table]
    importance = student_criteria1(m, N, y_arr, norm_x_table_with_x0)  # shows should each b(ij)*x(i) be in our main equation


    def equation(x1, x2, x3, coefs, importance=truth):
        factors_arr = [1, x1, x2, x3, x1 * x2, x1 * x3, x2 * x3, x1 * x2 * x3, x1 ** 2, x2 ** 2, x3 ** 2]
        updated = compress(zip(coefs, factors_arr), importance)
        return sum([el[0] * el[1] for el in updated])

    def fisher_criteria(m, N, d, x_table, y_table, b_coefficients, importance):
        print("\nЗа критерієм Фішера: m = {}, N = {} ".format(m, N))
        f3 = (m - 1) * N
        f4 = N - d
        q = 0.05

        if(N != 15):
            theoret_y = theor_y(x_table, b_coefficients, importance)
            y_averages = np.array(list(map(np.average, y_table)))

        else:
            theoret_y = np.array([equation(row[0], row[1], row[2], b_coefficients) for row in x_table])
            y_averages = np.array(list(map(lambda el: np.average(el), y_table)))

        theor_values_to_print = list(zip(map(lambda x: "x1 = {0[0]}, x2 = {0[1]}, x3 = {0[2]}".format(x), x_table), theoret_y))
        print("Теоретичні y:")
        print("\n".join([f"{el[0]}: y = {el[1]}  " for el in theor_values_to_print]))

        s_ad = m / (N - d) * (sum((theoret_y - y_averages) ** 2))
        y_variations = np.array(list(map(np.var, y_table)))
        s_v = np.average(y_variations)
        f_p = float(s_ad / s_v)
        f_t = get_fisher(f3, f4, q)
        print("Fp = {}, Ft = {}".format(f_p, f_t))
        print("Fp < Ft --> модель адекватна" if f_p < f_t else "Fp > Ft --> неадекватна")
        return True if f_p < f_t else False


    x_table_with_x0 = [[+1] + row for row in x_table]
    fisher_with_interaction = fisher_criteria(m, N, 1, x_table_with_x0, y_arr, b_i, importance)
    print(" (при врахуванні взаємодії)\n\n")





#with quadratic terms(LAB6)
    if fisher_with_interaction == False:
        m = 3
        N = 15
        x0_i = [(x1max + x1min) / 2, (x2max + x2min) / 2, (x3max + x3min) / 2]
        delta_x_i = [(x1min - x0_i[0]), (x2min - x0_i[1]), (x3min - x0_i[2])]
        l = 1.215
        truth = [True] * 11

        nat_x_table = [[x1min, x2min, x3min],
                       [x1min, x2max, x3max],
                       [x1max, x2min, x3max],
                       [x1max, x2max, x3min],

                       [x1min, x2min, x3max],
                       [x1min, x2max, x3min],
                       [x1max, x2min, x3min],
                       [x1max, x2max, x3max],

                       [-l * delta_x_i[0] + x0_i[0], x0_i[1], x0_i[2]],
                       [l * delta_x_i[0] + x0_i[0], x0_i[1], x0_i[2]],
                       [x0_i[0], -l * delta_x_i[1] + x0_i[1], x0_i[2]],
                       [x0_i[0], l * delta_x_i[1] + x0_i[1], x0_i[2]],
                       [x0_i[0], x0_i[1], -l * delta_x_i[2] + x0_i[2]],
                       [x0_i[0], x0_i[1], l * delta_x_i[2] + x0_i[2]],
                       [x0_i[0], x0_i[1], x0_i[2]]]

        x_table = [[-1, -1, -1],
                 [-1, +1, +1],
                 [+1, -1, +1],
                 [+1, +1, -1],
                 [-1, -1, +1],
                 [-1, +1, -1],
                 [+1, -1, -1],
                 [+1, +1, +1],
                 [-1.73, 0, 0],
                 [+1.73, 0, 0],
                 [0, -1.73, 0],
                 [0, +1.73, 0],
                 [0, 0, -1.73],
                 [0, 0, +1.73]]


        def generate_x_table(raw_array):
            raw_list = [row + [row[0] * row[1], row[0] * row[2], row[1] * row[2], row[0] * row[1] * row[2]] +
                        [row[0] ** 2, row[1] ** 2, row[2] ** 2] for row in raw_array]
            print(raw_list)
            return list(map(lambda row: list(map(lambda el: round(el, 3), row)), raw_list))


        def generate_y(m, factors_table):
            return [[round(function(row[0], row[1], row[2]) + random.randint(0, 10) - 5, 3) for i in range(m)]
                    for row in factors_table]


        def function(x1, x2, x3):
            coefs = [1.4, 4.0, 9.3, 7.9, 7.7, 1.0, 0.7, 2.9, 0.4, 3.9, 1.0]
            return equation(x1, x2, x3, coefs)

        natural_plan = generate_x_table(nat_x_table)
        y_arr = generate_y(m, nat_x_table)

        while not cochran_criteria(m, N, y_arr):
            m += 1
            y_arr = generate_y(m,natural_plan)
        yi = np.array([np.average(i) for i in y_arr])  # average for each i_th row in y_arr
        print("y_arr",y_arr)
        print(yi)

        mat = np.array(nat_x_table)
        mat = np.append(mat, [[mat[i][0] * mat[i][1]] for i in range(N)], 1)
        mat = np.append(mat, [[mat[i][0] * mat[i][2]] for i in range(N)], 1)
        mat = np.append(mat, [[mat[i][1] * mat[i][2]] for i in range(N)], 1)
        mat = np.append(mat, [[mat[i][0] * mat[i][1] * mat[i][2]] for i in range(N)], 1)
        mat = np.append(mat, [[mat[i][0] ** 2] for i in range(N)], 1)
        mat = np.append(mat, [[mat[i][1] ** 2] for i in range(N)], 1)
        mat = np.append(mat, [[mat[i][2] ** 2] for i in range(N)], 1)
        np.set_printoptions(linewidth=150,suppress=True)
        x = np.transpose(mat)
        print(x)

        mx1 = sum(x[0]) / len(x[0])
        mx2 = sum(x[1]) / len(x[1])
        mx3 = sum(x[2]) / len(x[2])
        mx4 = sum(x[3]) / len(x[3])
        mx5 = sum(x[4]) / len(x[4])
        mx6 = sum(x[5]) / len(x[5])
        mx7 = sum(x[6]) / len(x[6])
        mx8 = sum(x[7]) / len(x[7])
        mx9 = sum(x[8]) / len(x[8])
        mx10 = sum(x[9]) / len(x[9])

        a11 = sum([x[0][i] * x[0][i] for i in range(N)]) / len(x[0])
        a12 = a21 = sum([x[0][i] * x[1][i] for i in range(N)]) / len(x[0])
        a13 = a31 = sum([x[0][i] * x[2][i] for i in range(N)]) / len(x[0])
        a14 = a41 = sum([x[0][i] * x[3][i] for i in range(N)]) / len(x[0])
        a15 = a51 = sum([x[0][i] * x[4][i] for i in range(N)]) / len(x[0])
        a16 = a61 = sum([x[0][i] * x[5][i] for i in range(N)]) / len(x[0])
        a17 = a71 = sum([x[0][i] * x[6][i] for i in range(N)]) / len(x[0])
        a18 = a81 = sum([x[0][i] * x[7][i] for i in range(N)]) / len(x[0])
        a19 = a91 = sum([x[0][i] * x[8][i] for i in range(N)]) / len(x[0])
        a110 = a101 = sum([x[0][i] * x[9][i] for i in range(N)]) / len(x[0])

        a22 = sum([x[1][i] * x[1][i] for i in range(N)]) / len(x[0])
        a23 = a32 = sum([x[1][i] * x[2][i] for i in range(N)]) / len(x[0])
        a24 = a42 = sum([x[1][i] * x[3][i] for i in range(N)]) / len(x[0])
        a25 = a52 = sum([x[1][i] * x[4][i] for i in range(N)]) / len(x[0])
        a26 = a62 = sum([x[1][i] * x[5][i] for i in range(N)]) / len(x[0])
        a27 = a72 = sum([x[1][i] * x[6][i] for i in range(N)]) / len(x[0])
        a28 = a82 = sum([x[1][i] * x[7][i] for i in range(N)]) / len(x[0])
        a29 = a92 = sum([x[1][i] * x[8][i] for i in range(N)]) / len(x[0])
        a210 = a102 = sum([x[1][i] * x[9][i] for i in range(N)]) / len(x[0])

        a33 = sum([x[2][i] * x[2][i] for i in range(N)]) / len(x[0])
        a34 = a43 = sum([x[2][i] * x[3][i] for i in range(N)]) / len(x[0])
        a35 = a53 = sum([x[2][i] * x[4][i] for i in range(N)]) / len(x[0])
        a36 = a63 = sum([x[2][i] * x[5][i] for i in range(N)]) / len(x[0])
        a37 = a73 = sum([x[2][i] * x[6][i] for i in range(N)]) / len(x[0])
        a38 = a83 = sum([x[2][i] * x[7][i] for i in range(N)]) / len(x[0])
        a39 = a93 = sum([x[2][i] * x[8][i] for i in range(N)]) / len(x[0])
        a310 = a103 = sum([x[2][i] * x[9][i] for i in range(N)]) / len(x[0])

        a44 = sum([x[3][i] * x[3][i] for i in range(N)]) / len(x[0])
        a45 = a54 = sum([x[3][i] * x[4][i] for i in range(N)]) / len(x[0])
        a46 = a64 = sum([x[3][i] * x[5][i] for i in range(N)]) / len(x[0])
        a47 = a74 = sum([x[3][i] * x[6][i] for i in range(N)]) / len(x[0])
        a48 = a84 = sum([x[3][i] * x[7][i] for i in range(N)]) / len(x[0])
        a49 = a94 = sum([x[3][i] * x[8][i] for i in range(N)]) / len(x[0])
        a410 = a104 = sum([x[3][i] * x[9][i] for i in range(N)]) / len(x[0])

        a55 = sum([x[4][i] * x[4][i] for i in range(N)]) / len(x[0])
        a56 = a65 = sum([x[4][i] * x[5][i] for i in range(N)]) / len(x[0])
        a57 = a75 = sum([x[4][i] * x[6][i] for i in range(N)]) / len(x[0])
        a58 = a85 = sum([x[4][i] * x[7][i] for i in range(N)]) / len(x[0])
        a59 = a95 = sum([x[4][i] * x[8][i] for i in range(N)]) / len(x[0])
        a510 = a105 = sum([x[4][i] * x[9][i] for i in range(N)]) / len(x[0])

        a66 = sum([x[5][i] * x[5][i] for i in range(N)]) / len(x[0])
        a67 = a76 = sum([x[5][i] * x[6][i] for i in range(N)]) / len(x[0])
        a68 = a86 = sum([x[5][i] * x[7][i] for i in range(N)]) / len(x[0])
        a69 = a96 = sum([x[5][i] * x[8][i] for i in range(N)]) / len(x[0])
        a610 = a106 = sum([x[5][i] * x[9][i] for i in range(N)]) / len(x[0])

        a77 = sum([x[6][i] * x[6][i] for i in range(N)]) / len(x[0])
        a78 = a87 = sum([x[6][i] * x[7][i] for i in range(N)]) / len(x[0])
        a79 = a97 = sum([x[6][i] * x[8][i] for i in range(N)]) / len(x[0])
        a710 = a107 = sum([x[6][i] * x[9][i] for i in range(N)]) / len(x[0])

        a88 = sum([x[7][i] * x[7][i] for i in range(N)]) / len(x[0])
        a89 = a98 = sum([x[7][i] * x[8][i] for i in range(N)]) / len(x[0])
        a810 = a108 = sum([x[7][i] * x[9][i] for i in range(N)]) / len(x[0])

        a99 = sum([x[8][i] * x[8][i] for i in range(N)]) / len(x[0])
        a910 = a109 = sum([x[8][i] * x[9][i] for i in range(N)]) / len(x[0])

        a1010 = sum([x[9][i] * x[9][i] for i in range(N)]) / len(x[0])

        coefficients = [[1, mx1, mx2, mx3, mx4, mx5, mx6, mx7, mx8, mx9, mx10],
                   [mx1, a11, a21, a31, a41, a51, a61, a71, a81, a91, a101],
                   [mx2, a12, a22, a32, a42, a52, a62, a72, a82, a92, a102],
                   [mx3, a13, a23, a33, a43, a53, a63, a73, a83, a93, a103],
                   [mx4, a14, a24, a34, a44, a54, a64, a74, a84, a94, a104],
                   [mx5, a15, a25, a35, a45, a55, a65, a75, a85, a95, a105],
                   [mx6, a16, a26, a36, a46, a56, a66, a76, a86, a96, a106],
                   [mx7, a17, a27, a37, a47, a57, a67, a77, a87, a97, a107],
                   [mx8, a18, a28, a38, a48, a58, a68, a78, a88, a98, a108],
                   [mx9, a19, a29, a39, a49, a59, a69, a79, a89, a99, a109],
                   [mx10, a110, a210, a310, a410, a510, a610, a710, a810, a910, a1010]]

        free_values = [0 for i in range(11)]
        free_values[0] = sum(yi) / len(yi)
        for i in range(1, 11):
            free_values[i] = sum([x[i - 1][j] * yi[j] for j in range(N)]) / N

        beta_coefficients = np.linalg.solve(coefficients, free_values)
        np.set_printoptions(suppress=True, linewidth=200)
        print(beta_coefficients)


        def student_criteria2(m, N, y_table, beta_coefficients):
            print("\nПеревірка значимості коефіцієнтів регресії за критерієм Стьюдента: ".format(m, N))
            average_variation = np.average(list(map(np.var, y_table)))
            print(average_variation)

            variation_beta_s = average_variation/N/m
            print(variation_beta_s)
            standard_deviation_beta_s = math.sqrt(variation_beta_s)
            t_i = np.array([(abs(beta_coefficients[i])/standard_deviation_beta_s) for i in range(len(beta_coefficients))])
            f3 = (m - 1) * N
            q = 0.05

            t = get_student(f3, q)
            importance = [True if el>t else False for el in list(t_i)]

            print("\n"+"Оцінки коефіцієнтів βs: " + ", ".join(list(map(lambda x: str(round(float(x), 3)), beta_coefficients))))
            print("Коефіцієнти ts:         " + ", ".join(list(map(lambda i: "{:.2f}".format(i), t_i))))
            print("f3 = {}; q = {}; tтабл = {}".format(f3, q, t))
            beta_i = ["β0", "β1", "β2", "β3", "β12", "β13", "β23", "β123", "β11", "β22", "β33"]
            importance_to_print = ["важливий" if i else "неважливий" for i in importance]
            to_print = map(lambda x: x[0] + " " + x[1], zip(beta_i, importance_to_print))
            x_i_names = list(compress(["", "x1", "x2", "x3", "x12", "x13", "x23", "x123", "x1^2", "x2^2", "x3^2"], importance))
            betas_to_print = list(compress(beta_coefficients, importance))
            print(*to_print, sep="; ")
            equation = " ".join(["".join(i) for i in zip(list(map(lambda x: "{:+.2f}".format(x), betas_to_print)), x_i_names)])
            print("Рівняння регресії без незначимих членів: y = " + equation)
            return importance


        def fisher_criteria2(m, N, d, x_table, y_table, b_coefficients, importance):
            def get_fisher_value(f3, f4, q):
                return round(abs(f.isf(q, f4, f3)))

            f3 = (m - 1) * N
            f4 = N - d
            q = 0.05
            theoretical_y = np.array([equation(row[0], row[1], row[2], b_coefficients) for row in x_table])
            average_y = np.array(list(map(lambda el: np.average(el), y_table)))
            s_ad = m / (N - d) * sum((theoretical_y - average_y) ** 2)
            y_variations = np.array(list(map(np.var, y_table)))
            s_v = np.average(y_variations)
            f_p = float(s_ad / s_v)
            f_t = get_fisher_value(f3, f4, q)
            theoretical_values_to_print = list(zip(map(lambda x: "x1 = {0[1]:<10} x2 = {0[2]:<10} x3 = {0[3]:<10}"
                                                       .format(x), x_table), theoretical_y))
            print(
                "\nПеревірка адекватності моделі за критерієм Фішера: m = {}, N = {} для таблиці y_table".format(m, N))
            print("Теоретичні значення y для різних комбінацій факторів:")
            print("\n".join(["{arr[0]}: y = {arr[1]}".format(arr=el) for el in theoretical_values_to_print]))
            print("Fp = {}, Ft = {}".format(f_p, f_t))
            print("Fp < Ft => модель адекватна" if f_p < f_t else "Fp > Ft => модель неадекватна")
            return True if f_p < f_t else False

        importance = student_criteria2(m, N, y_arr, beta_coefficients)
        fisher_last = fisher_criteria(m, N, len(list(filter(None, importance))), natural_plan, y_arr, beta_coefficients, importance)
        print(" (при врахуванні квадратичних членів)\n")
        yi_to_print = ["y_{} = {}".format(i, yi[i]) for i in range(len(yi))]
        for i in yi_to_print: print(i)

    with_interaction = False
