import numpy as np
import pandas as pd
import random
import scipy
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def selection_roulette_wheel(pop_ranking, parents_count):
    """
    The selection process emulates the survival of the fittest principle of nature.  After calculating the fitness score for each, a
    set of the best individuals are selected from the population. Roulette wheel selection emulates the selection after assigning
    different selection probability to each feature set on the basis of score.
    :param pop_ranking: pop ranking contains the fitness score for each feature set
    :type pop_ranking: 2 Dimensional array
    :param parents_count:number of parents selected for crossover and mutation.
    :type parents_count: Integer
    :return: individuals which are selected by the roulette wheel selection procedure
    :rtype: 2 dimensional array
    """
    results = []
    data_frame = pd.DataFrame(np.array(pop_ranking), columns=["id", "Fitness"])
    data_frame['cumSum'] = data_frame.Fitness.cumsum()
    data_frame['cumPer'] = 100 * data_frame.cumSum / data_frame.Fitness.sum()

    for i in range(0, parents_count):
        roulette_pick = 100 * random.random()
        for j in range(0, len(pop_ranking)):
            if roulette_pick <= data_frame.iat[j, 3]:
                results.append(pop_ranking[j][0])
                break
    return results


def select_features(selected_features, complete_features):
    """
    Method to select specific features from the complete set of features in the dataset
    :param selected_features: features selected for the current individual
    :type selected_features: numpy 1 dimensional array
    :param complete_features: complete set of features for the dataset
    :type complete_features: numpy 2 dimensional array
    :return: reduced features set
    :rtype: numpy 2d array with only selected features
    """
    reduced_features = complete_features[:, np.where(selected_features == 1)[0]]
    return reduced_features


def rank_population(fitness):
    """
    Method to return different features sets sorted by fitness score.
    :param fitness: feature sets along with corresponding fitness scores
    :type fitness: 2 Dimensional array
    :return: returns an array of feature set sorted by fitness score.
    :rtype: 2 Dimensional array
    """
    fitness_results = {}
    fitness_length = fitness.shape[0]
    for idx in range(fitness_length):
        fitness_results[idx] = fitness[idx]
    return sorted(fitness_results.items(), key=np.operator.itemgetter(1), reverse=True)


def evaluate_fitness(population, data, labels, model):
    """
    method to evaluate the fitness of each of the individual in the population for the given models and feature set
    :param population: 2 D array of 0's and 1's. 1 means current feature is selected and 0 mean
    :type population: 2 dimensional array
    :param data: Complete data set used to train and evaluate the models in evolution process
    :type data: 2 Dimensional numpy array
    :param labels: labels corresponding to each data point in the dataset
    :type labels: 1 dimensional array
    :param model: Machine learning model to be used for prediction.
    :type model:
    :return: An array of fitness scores where each ith entry corresponds to score for the ith individual in the population
    :rtype: 1 Dimensional array of Integers
    """
    fitness = np.zeros(population.shape[0])
    index = 0

    internal_train_indexes = np.arange(1, data.shape[0], 2)
    internal_evaluation_indexes = np.arange(0, data.shape[0], 2)

    for curr_population in population:
        selected_features = select_features(curr_population, data)
        train_data = selected_features[internal_train_indexes, :]
        test_data = selected_features[internal_evaluation_indexes, :]

        train_labels = labels[internal_train_indexes]
        test_labels = labels[internal_evaluation_indexes]

        model.fit(X=train_data, y=train_labels)
        predictions = model.predict(test_data)
        fitness[index] = accuracy_score(test_labels, predictions)
        index = index + 1
    return fitness


def mutation_operator(offspring, num_mutations):
    """
    Method to selectively mutate certain genes in the offsprings for diversity. In the context of feature selection, some
    of the selected features are mutated so that offsprings are not predominated by the features from parents with maximum fitness score.
    :param offspring: Offspring to be mutated.
    :type offspring: 2 D array of 0s and 1s
    :param num_mutations: number of mutation operations performed for each of the offspring(feature selection)
    :type num_mutations: An Integer
    :return: offspring with mutated genes as some features chosen at random were mutated
    :rtype: An array of 0s and 1s where 0s corresponds to features not selected and 1s corresponds to features selected
    """
    max_range = offspring.shape[1]
    for index in range(offspring.shape[0]):
        mutation_index = np.random.randint(low=0, high=max_range, size=num_mutations)
        offspring[index, mutation_index] = 1 - offspring[index, mutation_index]
    return offspring


def select_best_population(total_population, fitness_scores, parents_count):
    population_shape = total_population.shape[1]
    parents = np.empty((parents_count, population_shape))

    # uncomment this code for using roulette wheel selection to select the best population
    # rankingPopulation = rankPopulation(fitness_scores);
    # selection = selection_roulette_wheel(rankingPopulation,parents_count)
    # for p_num in range(0, len(selection)):
    # index = selection[p_num]
    # parents[p_num, :] = total_population[index]

    for parentNumber in range(parents_count):
        max_index = np.where(fitness_scores == np.max(fitness_scores))[0][0]
        parents[parentNumber, :] = total_population[max_index, :]
        fitness_scores[max_index] = -99999
    return parents


def crossover_operation(parents, offspring_shape):
    """
    Perform the crossover operation for the selected most fit parents and generate offsprings from them.
    :param parents: The fittest parent or group of different feature selection
    :type parents: numpy 2 dimensional array
    :param offspring_shape: shape of the offspring to be generated from parents
    :type offspring_shape:tuple for length 2
    :return: offspring: offspring generated after crossover
    :rtype: numpy 2 dimensional array
    """
    offspring = np.empty(offspring_shape)
    crossover_index = np.random.randint(low=0, high=offspring_shape[1])

    for idx in range(offspring_shape[0]):
        first_parent = idx % parents.shape[0]
        second_parent = (idx + 1) % parents.shape[0]
        offspring[idx, 0:crossover_index] = parents[first_parent, 0:crossover_index]
        offspring[idx, crossover_index:] = parents[second_parent, crossover_index:]
    return offspring


def run_iterations(num_generations, x_train, y_train, population_size, num_parents, num_mutations, model):

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    total_number_features = x_train.shape[1]

    # randomly generate population
    current_population = np.random.randint(low=0, high=2, size=(population_size, total_number_features))

    for generation in range(num_generations):
        # just to track the progress #todo a progress bar can be used.
        if generation % 10 == 0:
            print("Generation", generation)

        # calculating fitness of current population
        fitness = evaluate_fitness(current_population, x_train, y_train, model)

        # selecting best population by elitism strategy
        parents = select_best_population(current_population, fitness, num_parents)

        off_spring_shape = (population_size - parents.shape[0], total_number_features)
        offspring_crossover = crossover_operation(parents, off_spring_shape)
        offspring_mutation = mutation_operator(offspring_crossover, num_mutations)
        current_population[0:parents.shape[0], :] = parents
        current_population[parents.shape[0]:, :] = offspring_mutation

    return current_population[0]


def mean_confidence_interval(data, confidence=0.95):
    """
    Method to calculate the confidence interval using Scipy library
    :param data: The data used to calculate the mean and confidence interval
    :type data: List of values
    :param confidence: The percentage of experiments will include the true mean within the confidence interval
    :type confidence: A floating point number with default value of .95
    :return: this method return mean and the default 95% confidence interval for the data
    :rtype: floating Numbers
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    standard_error = scipy.stats.sem(data)
    h = standard_error * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, h


def evaluate_performance(model, num_times_average, x_train, y_train, x_test, y_test, current_population):
    accuracy_list = np.empty(num_times_average)
    precision_list = np.empty(num_times_average)
    recall_list = np.empty(num_times_average)
    auc_score_list = np.empty(num_times_average)

    for i in range(num_times_average):
        reduced_features_train = select_features(current_population, np.array(x_train))

        model.fit(X=reduced_features_train, y=y_train)

        reduced_features_test = select_features(current_population, np.array(x_test))
        predictions_test = model.predict(reduced_features_test)

        # evaluation
        accuracy = accuracy_score(y_test, predictions_test)
        accuracy_list[i] = accuracy

        average_precision = precision_score(y_test, predictions_test)
        precision_list[i] = average_precision

        average_recall = recall_score(y_test, predictions_test, average='macro')
        recall_list[i] = average_recall

        lr_prob = model.predict_proba(reduced_features_test)
        lr_prob = lr_prob[:, 1]
        lr_auc = roc_auc_score(y_test, lr_prob)
        auc_score_list[i] = lr_auc

    print("Average accuracy with C.I. is ", mean_confidence_interval(accuracy_list))
    print("Average precision with C.I is", mean_confidence_interval(precision_list))
    print("Average recall with C.I. is", mean_confidence_interval(recall_list))
    print("Average auc score with C.I. is", mean_confidence_interval(auc_score_list))


def evaluate_performance_rfe(selected_features_mapping, X, y, model):
    """
    Method to evaluate the performance of the model under consideration by selecting equal number feature as selected by genetic
    algorithm based feature selection using recursive feature elimination. This performance is used as benchmark for comparison.
    Average of scores over multiple iterations is taken and Confidence interval is calculated for the scores.
    :param selected_features_mapping: Binary array of feature mapping where 1 corresponds to selected features
    :type selected_features_mapping: 1d numpy array
    :param X: Test data set to evaluate the performance
    :type X: Numpy 2 dimensional array where number of rows is equal to test set size and number of columns is equal to feature set
    :param y: Target variable for the prediction task
    :type y: Numpy 1d array of size same as test set size
    :param model: Prediction model (Machine learning model) used for feature selection using genetic algorithm, model training and
    evaluation
    """
    num_times_average = 20
    accuracy_list = np.empty(num_times_average)
    precision_list = np.empty(num_times_average)
    recall_list = np.empty(num_times_average)
    auc_score_list = np.empty(num_times_average)

    number_of_features_selected = np.count_nonzero(selected_features_mapping == 1)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    selector = RFE(model, step=1, n_features_to_select=number_of_features_selected)
    selector = selector.fit(x_train, y_train)

    print("Selected Features by RFE", selector.support_)
    for i in range(num_times_average):
        # split the for each time

        reduced_features_train = select_features(selector.support_, np.array(x_train))

        model.fit(X=reduced_features_train, y=y_train)

        reduced_features_test = select_features(selector.support_, np.array(x_test))
        predictions_test = model.predict(reduced_features_test)

        # evaluation
        accuracy = accuracy_score(y_test, predictions_test)
        accuracy_list[i] = accuracy

        average_precision = precision_score(y_test, predictions_test)
        precision_list[i] = average_precision

        average_recall = recall_score(y_test, predictions_test, average='macro')
        recall_list[i] = average_recall

        lr_prob = model.predict_proba(reduced_features_test)
        lr_prob = lr_prob[:, 1]
        lr_auc = roc_auc_score(y_test, lr_prob)
        auc_score_list[i] = lr_auc

    print("Average accuracy with C.I. is ", mean_confidence_interval(accuracy_list))
    print("Average precision with C.I is", mean_confidence_interval(precision_list))
    print("Average recall with C.I. is", mean_confidence_interval(recall_list))
    print("Average auc score with C.I. is", mean_confidence_interval(auc_score_list))
