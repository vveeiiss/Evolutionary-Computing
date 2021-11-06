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
# import graphs

POP_LOW_VALUE = -1
POP_HIGH_VALUE = 1
POP_SIZE = 30
N_RUNS = 5
N_GENERATIONS = 10
SURVIVE_FRACTION = 0.6

N_CHILDREN = int(POP_SIZE - SURVIVE_FRACTION * POP_SIZE)
if N_CHILDREN % 2 != 0:
	N_CHILDREN -= 1

N_SURVIVED = POP_SIZE - N_CHILDREN
N_TESTS_BEST = 5
N_HIDDEN_NEURONS = 10

ALPHA = 0.05
MUTATION_RATE = 0.05
MUTATION_INFLUENCE = 1.00
CROSSOVER_RATE = 0.7
# ENEMIES_CHOSEN = [1, 3, 5, 7]
# ENEMIES_CHOSEN = [2]
ENEMIES_CHOSEN = [1,2,3,4,5,6,7,8]
NO_IMPROVEMENT_CRITERION = 50

EXPERIMENT_NAME = "generalist_{}".format(ENEMIES_CHOSEN)

def normalize(x):
  return [(val - min(x)) / (max(x)-min(x)) for val in x]

def calculate_parent_probabilities(pop_fitness):
    norm_fitness = normalize(pop_fitness)
    return norm_fitness/np.sum(norm_fitness)

def select_parents(pop_fitness):
    exponent = 0.8

    # Selection: Roulette wheel, probability of selection is relative to the fitness
    rank_nums = [x+1 for _, x in reversed(sorted(zip(pop_fitness, np.arange(POP_SIZE))))]

    # weights = [1-(1/len(rank_nums) * n) for n in rank_nums]
    weights = [exponent**n for n in rank_nums]

    # probabilities = calculate_parent_probabilities(pop_fitness)
    probabilities = calculate_parent_probabilities(weights)

    par_pairs = {}

    for child in range(int(N_CHILDREN/2)):
        # Choose 2 parents without replacement, with p=probabilities
        if np.random.random() < CROSSOVER_RATE:
            # Probability of crossover, child gets two parents
            par_pairs[child] = np.random.choice(range(POP_SIZE), size=2, replace=False, p=probabilities)
        else:
            # Child comes from a parent and its copy
            par_pairs[child] = np.array([np.random.choice(range(POP_SIZE), replace=False, p=probabilities) for _ in range(2)])


    return par_pairs

# Create offspring via crossover
def create_children(pop, par_pairs):
    children = []

    for i in range(len(par_pairs)):
        par_genomes = [ [pop[par_pairs[i][0]], pop[par_pairs[i][1]]] for i in range(int(N_CHILDREN/2))]
        n_weights = len(par_genomes[0][0])

        ## Play head/tails between parents, per weight
        order = np.random.choice([0, 1], size=n_weights)
        reverse_order = [0 if z == 1 else 1 for z in order]

        ## First child gets the order, second child reversed order between parents
        child_1 = [par_genomes[i][order[n]][n] for n in range(n_weights)]
        child_2 = [par_genomes[i][reverse_order[n]][n] for n in range(n_weights)]

        children.extend([child_1, child_2])

    return children

# mutate offspring
def mutate_children(children, n_no_improvement):
    for c, child in enumerate(children):
        for w, _ in enumerate(child):
            if np.random.random() < MUTATION_RATE * (1 + n_no_improvement/2):
                children[c][w] += np.random.normal(0, MUTATION_INFLUENCE)
    return children


def run_genomes(env, genomes, enemies):
    fitnesses = np.zeros([len(genomes), len(enemies)])
    avg_fitnesses = []

    for e, enemy in enumerate(enemies):
        # Update the enemy
        env.update_parameter('enemies',[enemy])
        for g, genome in enumerate(genomes):
            _, p_life, e_life, time = env.play(pcont=np.array(genome))
            fitnesses[g, e] = calculate_fitness(p_life, e_life, time)

    for g, genome in enumerate(genomes):
        genome_fitness = np.sum(fitnesses[g, :])
        avg_fitnesses.append(genome_fitness)

    return avg_fitnesses


def create_init_population(enemies, solutions):
    return np.random.uniform(POP_LOW_VALUE, POP_HIGH_VALUE, size=solutions.shape)

    perc_best_solutions = 1.0
    num_best_solutions = int(perc_best_solutions * POP_SIZE)

    population = []

    # for enemy in ENEMIES_CHOSEN:
    #     #     # choices = np.random.choice(np.arange(solutions[enemy].shape[0]), size=int(num_best_solutions/len(enemies)), replace=False)
    #     #     # population.extend( [ solutions[enemy][c,:] for c in choices] )
    #     #     population.extend( solutions[enemy] )


    # population = np.array(population)

    # check if all solutions are unique
    solutions = np.vstack({tuple(row) for row in solutions})

    population = solutions[0:min(num_best_solutions, len(solutions)),:]


    num_missing = POP_SIZE - len(population[:,])
    if num_missing > 0:
        random_pop = np.zeros((num_missing, population.shape[1]))
        population = np.concatenate([population, random_pop], axis = 0)

    return population

def apply_algorithm(env, enemies, best_sol_per_enemy, n_best):
    # NN = 20 inputs (sensors) , 10 hidden neurons, 5 outputs (action space)
    n_neurons = int((env.get_num_sensors() + 1) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5)

    generation_stats_names = [ "fitness_mean", "fitness_max"]
    #g_stats = { (g, n): [] for g in range(N_GENERATIONS) for n in generation_stats_names }
    g_stats = {n : {g : [] for g in range(N_GENERATIONS)} for n in generation_stats_names}

    run_stats_names = [ "genomes", "fitness" ]
    r_stats = { n: [] for n in run_stats_names}

    best_fitness_overall = -np.inf
    best_chromosone_overall = 0
    no_improvement_generations = -1


    # population = np.random.uniform(POP_LOW_VALUE, POP_HIGH_VALUE, size=best_chromosone_overall.shape)
    population =  np.array([])
    # population = create_init_population(enemies, best_sol_per_enemy)
    population_fitness = []
    for r in range(N_RUNS):
        if population.size == 0:
            population = np.random.uniform(POP_LOW_VALUE, POP_HIGH_VALUE, size=(POP_SIZE, n_neurons))
        children = population
        for g in range(N_GENERATIONS):
            print("Run        :", r+1)
            print("Generation :", g+1)
            children_fitness = run_genomes(env, children, enemies)
            population_fitness = np.append(population_fitness, children_fitness)


            # for f, fit in enumerate(population_fitness):
            #     print("Genome {} fitness = {}".format(f, fit))

            g_stats["fitness_mean"][g].append(np.mean(population_fitness))
            g_stats["fitness_max"][g].append(np.max(population_fitness))


            # Parent selection
            parent_pairs = select_parents(population_fitness)

            # Order on population fitness
            ordered_num = [x for _, x in reversed(sorted(zip(population_fitness, np.arange(POP_SIZE))))]

            population = np.array([population[i,:] for i in ordered_num])
            population_fitness = np.array([population_fitness[i] for i in ordered_num])

            # print n best fitnesses of this generation
            for best in range(n_best):
                print("Genome #{}\tfitness = {}".format(best+1, population_fitness[best]))

            # Survived (best performing)
            survived_population = population[:N_SURVIVED,:]
            survived_fitness = population_fitness[:N_SURVIVED]

            # Create children via crossover
            children = create_children(population, parent_pairs)

            # Mutation
            mutated_children = mutate_children(children, no_improvement_generations)

            # Create new population (age based)
            population[0: N_SURVIVED, :] = survived_population
            population[N_SURVIVED:, :] = mutated_children

            population_fitness = survived_fitness

            population = np.vstack({tuple(row) for row in population})

            if len(population) < POP_SIZE:
                missing = POP_SIZE - len(population)
                population = np.append(population, np.random.uniform(POP_LOW_VALUE, POP_HIGH_VALUE, size=(missing, population.shape[1]), axis=0))

            if survived_fitness[0] > best_fitness_overall:
                # New best solution
                best_fitness_overall = population_fitness[0]
                best_chromosone_overall = np.array(population[0, :])
                no_improvement_generations = 0
                np.savetxt(EXPERIMENT_NAME+ '/cp_f{:02}_r{:01}_g{:01}'.format(best_fitness_overall, r, g), best_chromosone_overall)

            else:
                # No new best solution
                no_improvement_generations += 1

            if no_improvement_generations == NO_IMPROVEMENT_CRITERION:
                population_fitness = []
                population = np.array([])
                break

    ## add n_best best performing genomes from this run
    r_stats["fitness"].extend(survived_fitness[0])
    r_stats["genomes"].extend(survived_population[0])

    # r_stats["fitness"].extend(population_fitness[:50])
    # r_stats["genomes"].extend(population[0:50,:])

    return best_chromosone_overall, g_stats, r_stats

def calculate_fitness(player_life, enemy_life, t):
    fitness = player_life - enemy_life - np.log(t)
    return fitness

def run_specialist(enemies = ENEMIES_CHOSEN):
    # Start time
    start_time = time.time()

    best_solutions = {}

    # for e, enemy in enumerate(enemies):
    #     best_solutions[enemy] = np.loadtxt("Solutions/best_solutions_enemy_{}.txt".format(e+1))

    best_solutions = np.loadtxt("best_50_solutions_enemies_[1, 2, 3, 4, 5, 6, 7, 8].txt")

    headless = True

    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Setup storage folder
    if not os.path.exists(EXPERIMENT_NAME):
        os.makedirs(EXPERIMENT_NAME)

    environment = Environment(experiment_name=EXPERIMENT_NAME,
                          player_controller=player_controller(N_HIDDEN_NEURONS),
                          enemies=[1], multiplemode="no", randomini="yes")

    ## Run the algorithm, return all statistics
    n_best_per_run = 10

    best_chromosone, generation_stats, run_stats = apply_algorithm(environment, enemies, best_solutions, n_best_per_run)

    # save generation stats in text file
    np.savetxt(EXPERIMENT_NAME+ '/fitness_mean.txt', np.array([generation_stats['fitness_mean'][generation] for generation in range(N_GENERATIONS)]))
    np.savetxt(EXPERIMENT_NAME+'/fitness_max.txt', np.array([generation_stats['fitness_max'][generation] for generation in range(N_GENERATIONS)]))

    ## Test all best chromosones per run 10 times
    avg_fitnesses = []

    ## Change back to this value after code works
    n_tests = 10

    for chromosone in run_stats["genomes"]:
        test_fitness = []
        for test in range(n_tests):
            sum_fitnesses = 0
            for enemy in enemies:
                environment.update_parameter('enemies',[enemy])

                _, p_life, e_life, t = environment.play(pcont = np.array(chromosone))
                sum_fitnesses += calculate_fitness(p_life, e_life, t)
            test_fitness.append(np.mean(sum_fitnesses))
        avg_fitnesses.append(np.mean(test_fitness))

    # Save half of the best performing over the 10 runs
    # best_performing = [x for _, x in reversed(sorted(zip(avg_fitnesses, np.arange(len(run_stats["fitness"])))))][: int((n_best_per_run*N_RUNS)/2)]

    # Save 10 of the best performing over the 10 runs
    best_performing = [x for _, x in reversed(sorted(zip(avg_fitnesses, np.arange(len(run_stats["fitness"])))))][: 10]


    best_genomes = [run_stats["genomes"][i] for i in best_performing]

    best_array = np.array(best_genomes)

    # Save numpy array as text file
    np.savetxt(EXPERIMENT_NAME +'/best_sol_generalist_{}.txt'.format(enemies), best_array)

    # Print elapsed time
    elapsed_time = time.time()-start_time
    hours = int(elapsed_time / 3600)
    mins = int((elapsed_time % 3600) / 60)
    secs = int((elapsed_time % 3600) % 60)
    print('Elapsed time (hh:mm:ss) {:02}:{:02}:{:02}'.format(hours, mins, secs))


if __name__ == "__main__":
    run_specialist(ENEMIES_CHOSEN)
