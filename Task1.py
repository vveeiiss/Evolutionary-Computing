################################
# EvoMan FrameWork - V1.0 2016 #
################################
# Evolutionary Computing
# TASK : 1
# Group: 25
################################

import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import random
np.random.seed(12345)

sys.path.insert(0, 'evoman')
from environment import Environment  # Import framework
from demo_controller import player_controller

# Algorithm parameters:
pop_low_value = -1  # Default=-1, Minimum weight value
pop_high_value = 1  # Default=1, Maximum weight value
pop_size = 10       # Default=35, Population size
n_runs = 2         # Default=10 (requirement), Number of runs
n_generations = 2  # Default=50, Number of generations per run
n_parents = int(0.45 * pop_size)  # Must be 45%, Number of parents selected, 50% of population
children_size = pop_size - n_parents  # Children size
n_tests_best = 5    # Default=5 (requirement), test X times final best solutions
algorithms_n_hidden_neurons = [10, 40]  # Default [10, 40] always 2 algorithms, difference is number of neurons
algo_color = ["red", "green"]  # Plot color for each algorithm
enemies_chosen_name = ["Flashman", "AirMan", "WoodMan", "HeatMan", "Metalman", "CrashMan", "BubbleMan", "QuickMan"]
# Choose enemy:
# "Flashman", "AirMan", "WoodMan", "HeatMan", "Metalman", "CrashMan", "BubbleMan, "QuickMan"
#     1           2         3           4           5          6           7           8
enemies_chosen = [1, 2, 3]  # Default [1, 4, 7] (requirement = 3 enemies), enemy as a number: range 1-8
alpha = 0.05 # Check against p-value outcome statistic test

mutation_influence = 0.10

mutation_rate = 0.10
experiment_name = 'train (mutation rate = %s)'%mutation_rate  # folder to store training info

## 0.05 - 0.1 - 0.2 - 0.3 - 0.4 - 0.5 - 0.75 - 1.25 - 1.5 - 1.75 - 2


## 0.1 to 2

## 0.1 / 0.2 / 0.5 / 1 / 1.5 / 2


# Select best parents from population
def select_parent(population, population_fitness, n_parents):
    # NOTE: Only take parents that have a fitness > some level?
    parents = []
    parents_fitness = []
    parents_size = len(population_fitness)
    sort = np.argsort(population_fitness)[::-1][:parents_size]
    for bestoff in range(0, n_parents):
        parents.append(population[sort[bestoff]])
        parents_fitness.append(population_fitness[sort[bestoff]])

    return parents, parents_fitness

# Create offspring via crossover
def create_children(parents, n_neurons):
    children = np.zeros([children_size, n_neurons])
    # Take cross over at 25% of weights
    # NOTE: OR make this value bigger/smaller ?
    crossover_point = np.uint32(n_neurons/4)
    for index in range(children_size):
        # Index of the first parent to mate.
        index_first_parent = index % len(parents)
        # Index of the second parent to mate.
        index_second_parent = (index+1) % len(parents)
        # The child will have first part of its genes from the first parent.
        parent_array = np.array(parents)
        children[index][0:crossover_point] = parent_array[index_first_parent, 0:crossover_point]
        # The child will have its last part  of it genes from the second parent.
        children[index][crossover_point:] = parent_array[index_second_parent, crossover_point:]
    return children

# mutate offspring
def mutate_children(children):
    # NOTE: Only take some children instead of all ?
    for c, child in enumerate(children):
        for w in range(len(child)):
            if np.random.random() < mutation_rate:
                children[c][w] += np.random.normal(0, 1) * mutation_influence
                if children[c][w] < pop_low_value:
                    children[c][w] = pop_low_value
                elif children[c][w] > pop_high_value:
                    children[c][w] = pop_high_value
    return children


def write_best_solutions(solutions_dict):
    with open(experiment_name+'/best_solutions.txt', 'w') as f:
        f.write("Population size: \t%s\n"%pop_size)
        f.write("Number of parents: \t%s\n"%n_parents)
        f.write("Number of generations: \t%s\n"%n_generations)
        f.write("Number of runs: \t%s\n"%n_runs)
        f.write("Mutation rate: \t\t%s\n"%mutation_rate)
        f.write("Mutation influence: \t%s\n"%mutation_influence)
        f.write("\n\n\n")

        for number in algorithms_n_hidden_neurons:
            f.write("________________________________________________________________________________\n")
            f.write("________________________________________________________________________________\n")
            f.write("Number of Hidden Neurons: \t %s\n\n"%number)
            for enemy_id in enemies_chosen:
                f.write("\n")
                f.write("Enemy: \t %s \n"%enemies_chosen_name[enemy_id-1])

                current_solution = solutions_dict[number]

                f.write("Average fitness (5 runs): \t %s \n  "%current_solution[0])
                f.write("Weights: \t\t [ ")
                for a, item in enumerate(current_solution[1]):
                    f.write(str(item))
                    if a != len(current_solution[1])-1:
                        f.write(" , ")
                f.write(" ]")
                f.write("\n")
            f.write("\n \n")

def start_training():
    # Do not use visuals (headless)
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Setup storage folder
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Start time
    start_time = time.time()

    # Do evolution
    algo_enemies_max_plot = []
    algo_enemies_mean_plot = []
    algo_enemies_best_plot = []

    best_solutions = {}

    for algo_index, n_hidden_neurons in enumerate(algorithms_n_hidden_neurons):
        print()
        print("Training with algorithm", algo_index, " Number of hidden (1 layer) neurons=", n_hidden_neurons)

        enemies_max_plot = []
        enemies_mean_plot = []
        enemies_best_plot = []

        enemy_best_average_fitness = -np.inf

    
        # Create evoman environment:
        # Individual evolution mode (AI player against static enemy)
        env = Environment(experiment_name=experiment_name,
                            player_controller=player_controller(n_hidden_neurons), multiplemode= "yes",
                            enemies=enemies_chosen)
        # writes all variables related to game state into log
        env.state_to_log()

        # NN= 20 inputs (sensors) , 10 hidden neurons, 5 outputs (action space)
        n_neurons = int((env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5)

        enemy_max_plot = []
        enemy_mean_plot = []
        enemy_best_run = []

        enemy_best_weights = []
        enemy_best_average_fitness = -np.inf

        for run in range(n_runs):
            run_max_plot = []
            run_mean_plot = []
            run_best_enemy = []

            population = np.random.uniform(low=pop_low_value, high=pop_high_value, size=(pop_size, n_neurons))
            print(n_neurons)
            best_chromosone = []
            best_fitness = -1 * sys.maxsize
            parents_fitness = []
            children = population

            for generation in range(n_generations):
                print("Max Average Fitness Found [5 runs]: ", enemy_best_average_fitness)
                print()
                print("EA         :", algo_index+1)
                print("Enemy      :", enemies_chosen_name[0], enemies_chosen_name[1], enemies_chosen_name[2])
                print("Run        :", run+1)
                print("Generation :", generation+1)

                # Calculate fitness for every chromosone in this generation
                population_fitness = parents_fitness

                for chromosone in children:
                    fitness, _, _, _ = env.play(pcont=chromosone)
                    population_fitness.append(fitness)
                    # save best chromosone with best fitness
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_chromosone = chromosone


                run_max_plot.append(np.max(population_fitness))
                run_mean_plot.append(np.mean(population_fitness))

                # Parent selection
                parents, parents_fitness = select_parent(population, population_fitness, n_parents)
                # Create children via crossover
                children = create_children(parents, n_neurons)
                # Mutation
                mutated_children = mutate_children(children)
                # Create new population (age based)
                population[0: len(parents), :] = parents
                population[len(parents):, :] = mutated_children

                #print("Population shape: ", population.shape)

            enemy_max_plot.append(run_max_plot)
            enemy_mean_plot.append(run_mean_plot)

            # Test best 5 times
            best_five_fitness = []
            for test in range(n_tests_best):
                fitness, _, _, _ = env.play(pcont=best_chromosone)
                best_five_fitness.append(fitness)
            # Save average of best 5 times
            best_average_fitness = np.mean(best_five_fitness)
            enemy_best_run.append(best_average_fitness)

            if best_average_fitness > enemy_best_average_fitness:
                enemy_best_average_fitness = best_average_fitness
                enemy_best_weights = best_chromosone

        best_solutions[n_hidden_neurons] = [enemy_best_average_fitness, enemy_best_weights]

        enemies_max_plot.append(enemy_max_plot)
        enemies_mean_plot.append(enemy_mean_plot)
        enemies_best_plot.append(enemy_best_run)

    algo_enemies_max_plot.append(enemies_max_plot)
    algo_enemies_mean_plot.append(enemies_mean_plot)
    algo_enemies_best_plot.append(enemies_best_plot)

    write_best_solutions(best_solutions)

    # Calculate mean and max per generation per run per enemy
    algo_enemy_max_plot = []
    algo_enemy_max_std_plot = []
    algo_enemy_mean_plot = []
    algo_enemy_mean_std_plot = []
    for algo_index in range(len(algorithms_n_hidden_neurons)):
        enemy_max_plot = []
        enemy_max_std_plot = []
        enemy_mean_plot = []
        enemy_mean_std_plot = []
        # Compute max.std of all run's per generation of ech enemy
        for enemy_max in algo_enemies_max_plot[algo_index]:
            enemy_max_average = []
            enemy_max_std_average = []
            for enemy_max_gen in range(len(enemy_max[0])):
                enemy_max_run_gen = []
                for enemy_max_run in enemy_max:
                    enemy_max_run_gen.append(enemy_max_run[enemy_max_gen])
                enemy_max_average.append(np.max(enemy_max_run_gen))
                enemy_max_std_average.append(np.std(enemy_max_run_gen))
            enemy_max_plot.append(enemy_max_average)
            enemy_max_std_plot.append(enemy_max_std_average)
        algo_enemy_max_plot.append(enemy_max_plot)
        algo_enemy_max_std_plot.append(enemy_max_std_plot)

        # Compute mean/std of all run's per generation of ech enemy
        for enemy_mean in algo_enemies_mean_plot[algo_index]:
            enemy_mean_average = []
            enemy_mean_std_average = []
            for enemy_mean_gen in range(len(enemy_mean[0])):
                enemy_mean_run_gen = []
                for enemy_mean_run in enemy_mean:
                    enemy_mean_run_gen.append(enemy_mean_run[enemy_mean_gen])
                enemy_mean_average.append(np.mean(enemy_mean_run_gen))
                enemy_mean_std_average.append(np.std(enemy_mean_run_gen))
            enemy_mean_plot.append(enemy_mean_average)
            enemy_mean_std_plot.append(enemy_mean_std_average)
        algo_enemy_mean_plot.append(enemy_mean_plot)
        algo_enemy_mean_std_plot.append(enemy_mean_std_plot)

    # Plot fitness average max/std, mean/std over runs per generation of each enemy
    for enemy_index, enemy_id in enumerate(enemies_chosen):
        plt.title("Enemy: " + enemies_chosen_name[enemy_id-1], fontsize=25)
        plt.xlabel("Generation", fontsize=25)
        plt.ylabel("Fitness", fontsize=25)
        for algo_index in range(len(algorithms_n_hidden_neurons)):
            plt.plot(algo_enemy_max_plot[algo_index][enemy_index], color=algo_color[algo_index], linestyle="dashed", label="Max " + "EA" + str(algo_index))
            plt.plot(algo_enemy_mean_plot[algo_index][enemy_index], color=algo_color[algo_index], label="Mean " + "EA" + str(algo_index))
            max_values = algo_enemy_max_plot[algo_index][enemy_index]
            max_std_diff_up = np.array(np.add(max_values, algo_enemy_max_std_plot[algo_index][enemy_index])).tolist()
            max_std_diff_down = np.array(np.subtract(max_values, algo_enemy_max_std_plot[algo_index][enemy_index])).tolist()
            plt.fill_between(range(n_generations), max_std_diff_up, max_std_diff_down, color=algo_color[algo_index], alpha=0.5)

            mean_values = algo_enemy_mean_plot[algo_index][enemy_index]
            mean_std_diff_up = np.array(np.add(mean_values, algo_enemy_mean_std_plot[algo_index][enemy_index])).tolist()
            mean_std_diff_down = np.array(np.subtract(mean_values, algo_enemy_mean_std_plot[algo_index][enemy_index])).tolist()
            plt.fill_between(range(n_generations), mean_std_diff_up, mean_std_diff_down, color=algo_color[algo_index],
                             alpha=0.5)
        plt.ylim((0,100))
        plt.legend()
        filename = experiment_name + "/Lineplot_" + enemies_chosen_name[enemy_id-1] + ".png"
        print("Saving line plot to file    :", filename)
        plt.savefig(filename)
        plt.show()

    # Boxplot best (tested 5 times) of each run
    title_data = "Enemies: "
    x_label = []
    data = []
    for enemy_index, enemy_id in enumerate(enemies_chosen):
        if enemy_index == 0:
            title_data += enemies_chosen_name[enemy_id-1]
        else:
            title_data += ", " + enemies_chosen_name[enemy_id - 1]
        for algo_index in range(len(algorithms_n_hidden_neurons)):
            data.append(algo_enemies_best_plot[algo_index][enemy_index])
            x_label.append("EA"+str(algo_index+1))
    plt.title(title_data)
    plt.ylabel("Best fitness")
    plt.ylim((0,100))

    plt.boxplot(data, labels=x_label)
    filename = experiment_name + "/Boxplot.png"
    print("Saving boxplot to file      :", filename)
    plt.savefig(filename)
    plt.show()

    # Save Boxplot data
    filename = experiment_name + '/Boxplot_data.txt'
    print("Saving boxplot data to file :", filename)
    np.savetxt(filename, data)

    # Statistical tests
    # Check normality
    print()
    if n_runs > 2:
        enemy_algo_normality = []
        for enemy_index, enemy_id in enumerate(enemies_chosen):
            algo_normality = []
            for algo_index in range(len(algorithms_n_hidden_neurons)):
                best_values = algo_enemies_best_plot[algo_index][enemy_index]
                _, p_value = stats.shapiro(best_values)
                if p_value > alpha:
                    print("EA", algo_index, " Enemy=", enemies_chosen_name[enemy_id-1], ", Best is normal distributed")
                    algo_normality.append("Normal")
                else:
                    print("EA", algo_index, " Enemy=", enemies_chosen_name[enemy_id - 1], ", Best is NOT normal distributed")
                    algo_normality.append("Not normal")
            enemy_algo_normality.append(algo_normality)
    else:
        print("Cannot perform statistics on number of runs smaller than 3.")

    # Check significance between the Algorithms per enemy
    if len(algorithms_n_hidden_neurons) > 1 and n_runs > 2:
        for enemy_index, enemy_id in enumerate(enemies_chosen):
            all_normal = 0
            best_values = []
            for algo_index in range(len(algorithms_n_hidden_neurons)):
                if enemy_algo_normality[enemy_index][algo_index] == "Normal":
                    all_normal += 1
            # Check all algorithms for normality
            best_values_1 = algo_enemies_best_plot[0][enemy_index]
            best_values_2 = algo_enemies_best_plot[1][enemy_index]
            if all_normal == len(algorithms_n_hidden_neurons):
                # Normal
                stat_ttest, p_value = stats.ttest_ind(best_values_1, best_values_2)
                print("Enemy=", enemies_chosen_name[enemy_id - 1], "Ttest output:", stat_ttest, p_value)
                if p_value > alpha:
                    print("Enemy=", enemies_chosen_name[enemy_id - 1], ", Algorithms have the same distribution")
                else:
                    print("Enemy=", enemies_chosen_name[enemy_id - 1], ", Algorithms have different distributions")
            else:
                # Not normal
                stat_mannwhitneyu, p_value = stats.mannwhitneyu(best_values_1, best_values_2)
                print("Enemy=", enemies_chosen_name[enemy_id - 1], "Mannwhitneyu output:", stat_mannwhitneyu, p_value)
                if p_value > alpha:
                    print("Enemy=", enemies_chosen_name[enemy_id - 1], ", Algorithms have the same distribution")
                else:
                    print("Enemy=", enemies_chosen_name[enemy_id - 1], ", Algorithms have different distributions")
    else:
        print("Cannot perform statistics on number of runs <3 or number of algorithms not egual to 2.")

    # Print elapsed time
    elapsed_time = time.time()-start_time
    hours = int(elapsed_time / 3600)
    mins = int((elapsed_time % 3600) / 60)
    secs = int((elapsed_time % 3600) % 60)
    print('Elapsed time (hh:mm:ss) {:02}:{:02}:{:02}'.format(hours, mins, secs))

def test_weights():
    chromosone = [ 0.42974611470141405 , -0.7539098398239619 , 0.3894357398160905 , -0.5417811288267105 , 0.8840053635422722 , 0.13981559058192938 , -0.30244592803313985 , 0.6979665640225432 , 0.25385724396359627 , 0.29008822291434677 , -0.3425411445848724 , 0.03525296307208378 , 0.6639541696182512 , 0.5190802543277285 , 0.360734327684724 , 0.6736842646929979 , 0.1374189698747943 , -0.6117799325982178 , 0.6803211620313281 , 0.9609280184596374 , -0.18745197382321602 , 0.9673553112669373 , 0.6907817891401041 , 0.30088284115181774 , -0.7803548391931214 , -0.5813373793565586 , 0.014543192485157341 , 0.44694016219678123 , -0.04108865286155572 , -0.7344107286597232 , -0.23170594361289765 , -0.3960065248220176 , -0.9095051623059918 , 0.9215258026334837 , -0.6667145434231638 , -0.5557437522198427 , -1.0 , 0.4710355327019202 , -0.2861027302017096 , -0.32451595933886335 , -0.7623064715857385 , -0.35368645519943237 , 0.9345237919416922 , 0.6522360359022245 , -0.7150711953782244 , -0.6961302120051512 , 0.9966703951359189 , 0.4810370023364121 , -0.30086589051137197 , -0.21521216338449203 , 0.7675812580032917 , 0.20894944972781226 , -0.4594926986528443 , -0.6317310378232637 , -0.7423095009846024 , -0.4987760617367579 , 0.13968989056301528 , -0.10660878461665327 , 0.8825067287660455 , -0.931071438794744 , -0.4860408474525579 , 0.28508149727020077 , 0.5590409297204508 , -0.5152096586877966 , 0.8495299628697575 , 0.25715428203631074 , -0.5282886430054434 , 0.008468460260954325 , -0.23480029965110824 , 0.6691640355806329 , 0.7133584893999662 , -0.4023744807091112 , -0.4938199609284917 , 0.05906103060957186 , -0.04806527176445312 , -0.9080634747633105 , -0.35947122473647664 , 0.20878550730051143 , 0.5561186099048754 , -0.6530294242416901 , 0.5696224802255943 , 0.47647547453205413 , -0.06075366227671607 , 0.7639750869800962 , -0.417211052535301 , -0.14884596640607625 , -0.5647588379454915 , 0.4126354732873191 , 0.7766887939658491 , 0.9185988525833628 , 0.23375028229216613 , -0.2791999837845029 , -0.20487645048690303 , -0.054180842979735744 , -0.7999980996489735 , -0.6172137681883004 , -0.39196053699411015 , -0.4264910274576829 , -0.6918939435071036 , 0.6096615394861338 , -0.5221287869027167 , 0.07924557697876922 , 0.09681669486776733 , -0.8268304810507578 , -0.5136677554149363 , -0.8646292811470945 , -0.2833848909928989 , -0.4589872443491152 , -0.2592761963124992 , 0.5393915692346126 , -0.7042992868191348 , 0.4083204904408184 , -0.7995796138361597 , -0.867396820170665 , -0.7427752910605647 , 0.8867844623052115 , -0.09478458927790923 , -0.265608473248804 , 0.003993797057971718 , -0.6406648037458403 , -0.11434128371737473 , -0.11345672696882309 , -0.10656200552551416 , 0.6638965363183884 , 0.40351527943200716 , -0.35686133352497684 , -0.46534967534487504 , -0.14834998044159087 , 0.014425688730098862 , -0.7317203938804664 , -0.29529171151825967 , -0.6325414325120693 , -0.19937035317028262 , 0.705137246626309 , 0.8103321427144079 , 0.9096696388204764 , -0.14198922540037917 , -0.986471365855057 , -0.4942494363840022 , -0.40272934315577946 , -0.14242992782843011 , -0.28742725782927236 , 0.3014160276910689 , 0.4957002576086518 , -0.7846426948457808 , 0.9294796411305928 , -0.10126726298511894 , -0.5132998603159205 , 1.0 , 0.5570695974982578 , -0.7565935294165356 , -0.8712863702102855 , -0.29777396464638584 , 0.4944184587273359 , -0.6794580614720133 , 0.4416546700631112 , -0.3594626011557744 , -0.5811417545571824 , -0.8353583503520254 , 0.27038035701462776 , -0.5866274947756035 , 0.48932286786300283 , 0.48807913802122377 , 1.0 , 0.28077008524038594 , -0.6565956579451184 , 0.044698766828778255 , 0.021285670222998185 , -0.143212076102804 , 0.4929359563916986 , 0.9144259103221594 , -0.4603349975637192 , -0.23604323691603257 , -0.15540350764214111 , 0.41036503913606215 , -0.06475127381276713 , -0.8997853038472552 , 0.13517944601751597 , -0.43079546372446037 , 0.35633016286563635 , -0.2783013808528314 , -0.034399650812173846 , 0.6488798595612134 , 0.3565182361023422 , 0.019319583634994503 , 0.20550377802475372 , 0.33829116979225116 , -0.5423593244981835 , 0.8511028623543953 , -0.8906943912800078 , 0.7136033039531007 , 0.07196483334111203 , 0.4545381191622702 , -0.462928231773988 , -0.6485931021443954 , 0.00456310948522648 , 0.4502029168319563 , -0.23077302082103818 , 0.02817821588828885 , -0.7807892696181364 , 0.1707551848557896 , 0.07389903347642468 , -0.5035111956604239 , 0.4146574199286435 , 0.34131533771505285 , 0.9476119087501158 , -0.9893148658834962 , -0.529075610753948 , 0.7939680723967499 , -1.0 , -0.8757640829904998 , -0.09505907984498418 , 0.07283476203806252 , 0.6586307353978704 , 0.7271741438177872 , 0.2938475935033853 , 0.017289227223743486 , 1.0 , 0.31234361527746035 , -0.7483681573835528 , 1.0 , 0.9912270091126187 , 0.8399710557236535 , -0.7558459161849004 , 0.05298931793338865 , -0.6012620136347341 , 0.7750750584842008 , -0.17268412012691026 , 0.929985742480536 , -0.9567937542682047 , 0.7541380720739951 , 0.03266464106690764 , -0.37777299638982265 , -0.5331082132963513 , -0.48937183239464765 , 0.5904861380720917 , 1.0 , 0.18802093470272108 , -0.3072012677830338 , 0.6805555221153432 , 0.6393614546243517 , -0.8624317404467121 , -0.8926985001563004 , -0.6687558989591429 , 0.39989580912953815 , 0.2502183468707019 , -0.9706509662712428 , -0.8678691030278618 , -0.42348399783892743 , 0.19685023690131898 , 0.24720346984013994 , -0.27784850477448925 , 0.1496763899896948 , 0.2079970572742461 , -0.09652733410838227 , -0.12740352767826996 , 0.25269294307882834 , 0.46938696945186503 , -0.4856674759103754 , -0.5974610331131234 , -0.5431688797718575 , -0.5706332774153464 , 0.04829434270762116 , 0.8763565254448054 , 0.45171445776798924 ]


    chromosone = np.array(chromosone)
    experiment_name = "best"
    n_hidden_neurons = 10
    enemy = 4

     # Do not use visuals (headless)
    headless = False
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Setup storage folder
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    # headless = False

    env = Environment(experiment_name=experiment_name,
                                  player_controller=player_controller(n_hidden_neurons),
                                  enemies=[enemy])

    # writes all variables related to game state into log
    # env.state_to_log()

    fitness, _, _, _ = env.play(pcont=chromosone)

if __name__ == '__main__':
    start_training()
    # test_weights()
