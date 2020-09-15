import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

V = [[-0.03756566897615467, -5.722100749595026], [-0.032434104385768964, 2.129758831785378],
     [-6.441698584317882, 0.009571614371669075], [-0.028982395556735163, 6.057715778213567],
     [-4.9414665689115385, 4.858195621972518], [-5.019565545882845, -4.903593836368199],
     [-0.04044499476866669, -0.003236937261967156], [5.807843196962095, -0.05202326990575397],
     [-0.025513151593125765, -2.512503982875313], [5.183961466806978, -4.9637653330430425],
     [5.0837753345921435, 4.937375102832942]]


def set_finder(x, y):
    final_array = []
    cluster_x = []
    cluster_y = []
    for i in x:
        for j in y:
            min_dist = 999999
            dist = 0
            temp = 999999
            centroid_index = 0
            for centroid in V:
                dist += (i - centroid[0]) ** 2 + (j - centroid[1]) ** 2
                if dist < min_dist:
                    min_dist = dist
                    temp = centroid_index
                centroid_index += 1
                dist = 0
            final_array.append(temp)
            cluster_x.append(i)
            cluster_y.append(j)
    return cluster_x, cluster_y, final_array


x_grid = np.arange(-15, 15, 0.1)
y_grid = np.arange(-15, 15, 0.1)
cluster_plot_x, cluster_plot_y, belonging_sets = set_finder(x_grid, y_grid)
final_DF = pd.DataFrame()
final_DF[0] = cluster_plot_x[:]
final_DF[1] = cluster_plot_y[:]
final_DF[2] = belonging_sets[:]

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
label_color = [LABEL_COLOR_MAP[l] for l in belonging_sets]
plt.scatter(cluster_plot_x, cluster_plot_y, c=label_color)
plt.show()