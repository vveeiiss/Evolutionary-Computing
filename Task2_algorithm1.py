################################
# EvoMan FrameWork - V1.0 2016 #
################################
# Evolutionary Computing
# TASK : 2
# Group: 25
################################

import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# np.random.seed(12345)

sys.path.insert(0, 'evoman')
from environment import Environment  # Import framework
from demo_controller import player_controller


POP_LOW_VALUE = -1
POP_HIGH_VALUE = 1
POP_SIZE = 30
N_RUNS = 10
N_GENERATIONS = 100
SURVIVE_FRACTION = 0.3

N_CHILDREN = int(POP_SIZE - int(SURVIVE_FRACTION * POP_SIZE))
if N_CHILDREN % 2 != 0:
	N_CHILDREN -= 1

N_SURVIVED = POP_SIZE - N_CHILDREN
N_TESTS_BEST = 5
N_HIDDEN_NEURONS = 10

ALPHA = 0.05
MUTATION_RATE = 0.10
MUTATION_INFLUENCE = 1.00
ENEMIES_CHOSEN = [1, 2, 3]

# group 1 -> [1, 2, 3], group 2 -> [4, 5, 6, 7, 8]


EXPERIMENT_NAME = "generalist_enemy{}".format(ENEMIES_CHOSEN)

def normalize(x):
  return [(val - min(x)) / (max(x)-min(x)) for val in x]

def calculate_parent_probabilities(pop_fitness):
    norm_fitness = normalize(pop_fitness)
    return norm_fitness/np.sum(norm_fitness)

# Selection: Roulette wheel, probability of selection is relative to the fitness
def select_parents(pop_fitness):
    probabilities = calculate_parent_probabilities(pop_fitness)
    par_pairs = {}

    for child in range(int(N_CHILDREN/2)):
        # Choose 2 parents without replacement, with p=probabilities
        par_pairs[child] = np.random.choice(range(POP_SIZE), size=2, replace=False, p=probabilities)

    return par_pairs

# Create offspring via crossover
def create_children(pop, par_pairs):
    children = []

    for i in range(len(par_pairs)):
        par_genomes = [ [pop[par_pairs[i][0]], pop[par_pairs[i][1]]] for i in range(int(N_CHILDREN/2))]
        n_weights = len(par_genomes[0][0])

        ## Play head/tails between parents, per weight
        order = np.random.choice([0, 1], n_weights)
        reverse_order = [0 if z == 1 else 1 for z in order]

        ## First child gets the order, second child reversed order between parents
        child_1 = [par_genomes[i][order[n]][n] for n in range(n_weights)]
        child_2 = [par_genomes[i][reverse_order[n]][n] for n in range(n_weights)]

        children.append(child_1)
        children.append(child_2)

    return children

# mutate offspring
def mutate_children(children, last_growth):
    for c, child in enumerate(children):
        for w, _ in enumerate(child):
            if np.random.random() < MUTATION_RATE + (1 - MUTATION_RATE) * (np.sqrt(last_growth)/(4+np.sqrt(last_growth))):
                children[c][w] += np.random.normal(0, (MUTATION_INFLUENCE + np.log(last_growth)))
    return children


def run_genomes(env, genomes, enemies):
    fitnesses = np.zeros([len(genomes), len(enemies)])
    avg_fitnesses = []

    for e, enemy in enumerate(enemies):
        # Update the enemy
        env.update_parameter('enemies',[enemy])

        for g, genome in enumerate(genomes):
            fitnesses[g, e], _, _, _ = env.play(pcont=np.array(genome))
            #_, pl, el, _ = env.play(pcont=np.array(genome))
            #fitnesses[g, e] = pl - el

    for g, genome in enumerate(genomes):
        # genome.fitness = ( np.mean(fitnesses[i,:]) + np.min(fitnesses[i,:]) ) / 2
        #avg_fitnesses.append(np.sum(fitnesses[g,:]))
        avg_fitnesses.append(np.mean(fitnesses[g, :]))
    return avg_fitnesses


def apply_algorithm(env, enemies, init_pop, n_best):
    # NN = 20 inputs (sensors) , 10 hidden neurons, 5 outputs (action space)
    n_neurons = int((env.get_num_sensors() + 1) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5)

    missing_pop = POP_SIZE - init_pop.shape[0]

    generation_stats_names = ["fitness_mean",
                               "fitness_max"]

    g_stats = {n : {g : [] for g in range(N_GENERATIONS)} for n in generation_stats_names}


    run_stats_names = [ "genomes", "fitness" ]
    r_stats = { n: [] for n in run_stats_names}

    for r in range(N_RUNS):
        ## Reset population to
        random_pop = np.zeros((missing_pop, init_pop.shape[1]))
        population = np.concatenate([init_pop, random_pop], axis=0)

        children = population
        population_fitness = []
        last_growth = 1                 #new line
        for g in range(N_GENERATIONS):
            print("Run        :", r+1)
            print("Generation :", g+1)
            children_fitness = run_genomes(env, children, enemies)
            if g > 0:
                if max(children_fitness) >= max(population_fitness):
                    last_growth = 1
                else:
                    last_growth += 1

            population_fitness.extend(children_fitness)

            for f, fit in enumerate(population_fitness):
                print("Genome {} fitness = {}".format(f, fit))

            g_stats["fitness_mean"][g].append(np.mean(population_fitness))
            g_stats["fitness_max"][g].append(np.max(population_fitness))

            # Parent selection
            parent_pairs = select_parents(population_fitness)

            # Survived (best performing)

            survived_num = [x for _, x in reversed(sorted(zip(population_fitness, np.arange(POP_SIZE))))][:N_SURVIVED]
            survived_population = [population[i,:] for i in survived_num]
            survived_fitness = [x for x in reversed(sorted(population_fitness))][:N_SURVIVED]

            # Create children via crossover
            children = create_children(population, parent_pairs)

            # Mutation
            mutated_children = mutate_children(children, last_growth)

            # Create new population (age based)

            population[0: N_SURVIVED, :] = survived_population
            population[N_SURVIVED:, :] = mutated_children

            population_fitness = survived_fitness

        ## add n_best best performing genomes from this run
        #r_stats["fitness"].extend(survived_fitness[:n_best])
        #r_stats["genomes"].extend(survived_population[:n_best])
        r_stats["fitness"].extend(survived_fitness[0])
        r_stats["genomes"].extend(survived_population[0])

    return g_stats, r_stats


def run_specialist(enemies = ENEMIES_CHOSEN):
    # Start time
    enemy = 'all'

    start_time = time.time()

    nonrandom_population = np.random.normal(0, 1, (POP_SIZE, 265))

    headless = True

    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Setup storage folder
    if not os.path.exists(EXPERIMENT_NAME):
        os.makedirs(EXPERIMENT_NAME)

    environment = Environment(experiment_name=EXPERIMENT_NAME,
                          player_controller=player_controller(N_HIDDEN_NEURONS),
                          enemies=[1], multiplemode="no")

    ## Run the algorithm, return all statistics
    n_best_per_run = 10

    generation_stats, run_stats = apply_algorithm(environment, enemies, nonrandom_population, n_best_per_run)

    # save generation stats in text file
    np.savetxt('fitness_mean_{}.txt'.format(EXPERIMENT_NAME), np.array([generation_stats['fitness_mean'][generation] for generation in range(N_GENERATIONS)]))
    np.savetxt('fitness_max_{}.txt'.format(EXPERIMENT_NAME), np.array([generation_stats['fitness_max'][generation] for generation in range(N_GENERATIONS)]))

    ## Test all 7 best chromosones per run 10 times
    averages = []

    ## Change back to this value after code works
    n_tests = 10

    for chromosone in run_stats["genomes"]:
        fitnesses = []
        for test in range(n_tests):
            fitness, _, _, _ = environment.play(pcont = np.array(chromosone))
            fitnesses.append(np.mean(fitness))
        averages.append(np.mean(fitnesses))

    # Save half of the best performing over the 10 runs
    best_performing = [x for _, x in reversed(sorted(zip(averages, np.arange(len(run_stats["fitness"])))))][: int((n_best_per_run*N_RUNS)/2)]
    best_genomes = [run_stats["genomes"][i] for i in best_performing]
    best_array = np.array(best_genomes)

    # Save numpy array as text file
    np.savetxt(EXPERIMENT_NAME +'_best_solutions_enemy_{}.txt'.format(enemy), best_array)

    # Print elapsed time
    elapsed_time = time.time()-start_time
    hours = int(elapsed_time / 3600)
    mins = int((elapsed_time % 3600) / 60)
    secs = int((elapsed_time % 3600) % 60)
    print('Elapsed time (hh:mm:ss) {:02}:{:02}:{:02}'.format(hours, mins, secs))


if __name__ == "__main__":
    run_specialist(ENEMIES_CHOSEN)
