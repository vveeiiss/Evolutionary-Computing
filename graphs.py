import matplotlib.pyplot as plt
import numpy as np

#enter name of experiment
experiment_name = 'test'


#fill in the right files at the right place

group1_algo1_means = np.loadtxt('fitness_mean_generalist_enemy[2, 4, 6, 8].txt')
group1_algo1_maxes = np.loadtxt('fitness_max_generalist_enemy[2, 4, 6, 8].txt')

group1_algo2_means = np.loadtxt('fitness_mean_generalist_enemy_with_mean[1, 3, 5, 7].txt')
group1_algo2_maxes = np.loadtxt('fitness_max_generalist_enemy_with_mean[1, 3, 5, 7].txt')

group2_algo1_means = np.loadtxt('fitness_mean_generalist_enemy[2, 4, 6, 8].txt')
group2_algo1_maxes = np.loadtxt('fitness_max_generalist_enemy[2, 4, 6, 8].txt')

group2_algo2_means = np.loadtxt('fitness_mean_generalist_enemy_with_mean[1, 3, 5, 7].txt')
group2_algo2_maxes = np.loadtxt('fitness_max_generalist_enemy_with_mean[1, 3, 5, 7].txt')


data = {'enemy_group_1': {'algorithm_1': {'fitness_mean_mean': np.mean(group1_algo1_means, axis = 1),
                                            'fitness_mean_max': np.mean(group1_algo1_maxes, axis = 1),
                                            'fitness_std_mean': np.std(group1_algo1_means, axis = 1),
                                            'fitness_std_max': np.std(group1_algo1_maxes, axis= 1),
                                            'color': 'blue'},
                          'algorithm_2': {'fitness_mean_mean': np.mean(group1_algo2_means, axis = 1),
                                            'fitness_mean_max': np.mean(group1_algo2_maxes, axis = 1),
                                            'fitness_std_mean': np.std(group1_algo2_means, axis = 1),
                                            'fitness_std_max': np.std(group1_algo2_maxes, axis = 1),
                                            'color': 'red'}
                          },
        'enemy_group_2': {'algorithm_1': {'fitness_mean_mean': np.mean(group2_algo1_means, axis = 1),
                                            'fitness_mean_max': np.mean(group2_algo1_maxes, axis = 1),
                                            'fitness_std_mean': np.std(group2_algo1_means, axis = 1),
                                            'fitness_std_max': np.std(group2_algo1_maxes, axis = 1),
                                            'color': 'blue'},
                           'algorithm_2': {'fitness_mean_mean': np.mean(group2_algo2_means, axis = 1),
                                            'fitness_mean_max': np.mean(group2_algo2_maxes, axis = 1),
                                            'fitness_std_mean': np.std(group2_algo2_means, axis = 1),
                                            'fitness_std_max': np.std(group2_algo2_maxes, axis = 1),
                                            'color': 'red'}
                          }
        }


for enemy_index, group in enumerate(data):
    plt.title(group, fontsize=25)
    plt.xlabel("Generation", fontsize=25)
    plt.ylabel("Fitness", fontsize=25)
    for algo in data[group]:
        plt.plot(data[group][algo]['fitness_mean_mean'], color=data[group][algo]['color'], label='{} {} {}'.format(group, algo, 'mean'))
        plt.plot(data[group][algo]['fitness_mean_max'], color=data[group][algo]['color'], linestyle="dashed", label= '{} {} {}'.format(group, algo, 'max'))

        num = range(len(data[group][algo]['fitness_mean_mean']))
        max_std_diff_up = np.array(np.add(data[group][algo]['fitness_mean_max'], data[group][algo]['fitness_std_max'])).tolist()
        max_std_diff_down = np.array(np.subtract(data[group][algo]['fitness_mean_max'], data[group][algo]['fitness_std_max'])).tolist()
        plt.fill_between(num, max_std_diff_up, max_std_diff_down, color=data[group][algo]['color'], alpha=0.5)

        max_std_diff_up = np.array(np.add(data[group][algo]['fitness_mean_mean'], data[group][algo]['fitness_std_mean'])).tolist()
        max_std_diff_down = np.array(np.subtract(data[group][algo]['fitness_mean_mean'], data[group][algo]['fitness_std_mean'])).tolist()
        plt.fill_between(num, max_std_diff_up, max_std_diff_down, color=data[group][algo]['color'], alpha=0.5)

    plt.legend()
    filename = experiment_name + "_Lineplot.png"
    print("Saving line plot to file    :", filename)
    plt.savefig(filename)
    plt.show()
