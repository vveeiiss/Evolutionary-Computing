import numpy as np

from environment import Environment  # Import framework
from demo_controller import player_controller
import matplotlib.pyplot as plt

experiment_name = 'Final_'

algo1_group1_name = ('', 'algo1_group1')
algo1_group2_name = ('', 'algo1_group2')
algo2_group1_name = ('', 'algo2_group1')
algo2_group2_name = ('', 'algo2_group2')


def run_genomes(env, genomes, enemies):
    fitnesses = np.zeros([len(genomes), len(enemies)])
    avg_fitnesses = []

    for e, enemy in enumerate(enemies):
        # Update the enemy
        env.update_parameter('enemies',[enemy])

        for g, genome in enumerate(genomes):
            fitness = 0
            for repeat in range(5):
                f, _, _, _ = env.play(pcont=np.array(genome))
                fitness += f
            fitnesses[g, e] = np.mean(fitness)
            #_, pl, el, _ = env.play(pcont=np.array(genome))
            #fitnesses[g, e] = pl - el

    for g, genome in enumerate(genomes):
        # genome.fitness = ( np.mean(fitnesses[i,:]) + np.min(fitnesses[i,:]) ) / 2
        #avg_fitnesses.append(np.sum(fitnesses[g,:]))
        avg_fitnesses.append(np.mean(fitnesses[g, :]))
    return avg_fitnesses



env = Environment(experiment_name=experiment_name,
                  player_controller=player_controller(10),
                  enemies=[1],
                  multiplemode="no")

files = [algo1_group1_name, algo1_group2_name, algo2_group1_name, algo2_group2_name]
best_solutions = np.loadtxt()
mean_fitness = []
labels = []

for solution in files:
    mean_fitness.append(run_genomes(env, [solution[1]], [1, 2, 3, 4, 5, 6, 7, 8]))
    labels.append(solution[0])

import matplotlib.pyplot as plt

title_data = "Enemies: "
plt.title(title_data, fontsize=25)
plt.ylabel("Best fitness", fontsize=25)
plt.boxplot(mean_fitness, labels = labels)
filename = 'final' + "_Boxplot.png"
print("Saving boxplot to file      :", filename)
plt.savefig(filename)
plt.show()