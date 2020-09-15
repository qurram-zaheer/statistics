import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import time

mining_data = pd.read_csv("mining_data.csv", header=None)
classification_data = pd.read_csv("classification_data.csv", header=None)

mining_array = mining_data.values
classification_array = classification_data.values
n = len(mining_data.columns)
N = len(mining_data)
m = 2
c_max = 12
J_arr = []
iter_array = []
starttime = time.time()
for c in range(1, c_max):
    ################# Initializing partition matrix
    U = []
    for i in range(N):
        col = np.random.dirichlet(np.ones(c))
        U.append(col)

    error = 999
    count = 0
    while error > 0.01:
        count += 1
        ################ Calculating V vectors
        V = []
        for i in range(c):
            temp = []
            for k in range(n):
                numerator_sum = 0
                denominator_sum = 0
                for j in range(N):
                    numerator_sum += U[j][i] ** m * mining_array[j][k]
                    denominator_sum += U[j][i] ** m
                temp.append(numerator_sum / denominator_sum)
            V.append(temp)

        ################ Updating partition matrix
        U_prev = copy.deepcopy(U)
        flags_array = []
        for i in range(c):
            for j in range(N):
                denominator_sum = 0
                dist = 0
                for k in range(n):
                    dist += (mining_array[j][k] - V[i][k]) ** 2
                if dist == 0:
                    U[j][i] = 1
                    flags_array.append(j)
                else:
                    dist_array = []
                    for t in range(c):
                        dist2 = 0
                        for k in range(n):
                            dist2 += (mining_array[j][k] - V[t][k]) ** 2
                        dist_array.append(dist2)
                    for t in range(c):
                        denominator_sum += dist / dist_array[t]
                    U[j][i] = 1 / denominator_sum

        #################### Setting partition matrix values to 0 if 1 is detected
        for j in flags_array:
            for i in range(c):
                if U[j][i] != 1:
                    U[j][i] = 0

        A = np.array(U)
        B = np.array(U_prev)
        error_norm = np.sqrt(abs(A ** 2 - B ** 2))
        error = np.amax(error_norm)

    try:
        belonging_sets = np.argmax(A, axis=1)
    except ValueError:
        print("A is undefined")

    mining_data[2] = belonging_sets[:]

    LABEL_COLOR_MAP = {
        0: '#964ade',
        1: '#798c5b',
        2: '#c21cb7',
        3: '#6c4c34',
        4: '#842cab',
        5: '#01b279',
        6: '#8eff89',
        7: '#60de33',
        8: '#08832f',
        9: '#e63ceb',
        10: '#6924da'
    }

    centroid_arr = np.array(V)
    label_color = [LABEL_COLOR_MAP[l] for l in belonging_sets]
    plt.scatter(mining_array[:, 0], mining_array[:, 1], c=label_color)
    plt.scatter(centroid_arr[:, 0], centroid_arr[:, 1], c='red')
    plt.show()

    J = 0
    for i in range(c):
        for j in range(N):
            dist = 0
            for k in range(n):
                dist += (mining_array[j][k] - V[i][k]) ** 2
            J += U[j][i] ** m * dist
    J_arr.append(J)
    iter_array.append(count)

print("time taken: ", time.time() - starttime)
c_arr = [i for i in range(1, 12)]

R_min = 999999
opt_index = 999
for i in range(3, 9):
    R = abs((J_arr[i] - J_arr[i + 1]) / (J_arr[i - 1] - J_arr[i]))
    if R < R_min:
        R_min = R
        opt_index = i

plt.plot(c_arr, J_arr, label='J vs c')

plt.legend()
plt.show()
plt.plot(c_arr, iter_array, label='Iterations vs c')
plt.legend()
plt.show()
cluster_plot_x = np.arange(-15, 15, 0.1)
cluster_plot_y = np.arange(-15, 15, 0.1)
