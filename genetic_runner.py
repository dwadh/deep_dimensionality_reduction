import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import genetic

models = [LogisticRegression, RandomForestClassifier, sklearn.svm.SVC(probability=True)]


def run_and_evaluate_performance(x_train, y_train, x_test, y_test):
    """
    Method will run the specific number of iterations of Genetic algorithm for feature selection
    along with the hyper parameters which include number of parents, number of mutations and population size
    """
    number_of_generations: int = 50
    population_size: int = 40
    number_of_parents: int = 20
    number_of_mutations: int = 4

    for model in models:
        selected_features_mapping = genetic.run_iterations(number_of_generations, x_train, y_train, population_size,
                                                           number_of_parents,
                                                           number_of_mutations,
                                                           model())
        print("Selected Features ", selected_features_mapping)
        genetic.evaluate_performance(model(), 20, x_train, y_train, x_test, y_test, selected_features_mapping)
        genetic.evaluate_performance_rfe(selected_features_mapping, X, y, model())
    return


if __name__ == '__main__':
    # load polish companies bankruptcy data and evaluate performance after feature selection using genetic algorithm
    taiwan_bankruptcy = "../data/Year5_resampled.csv"
    df = pd.read_csv(taiwan_bankruptcy)
    y = df['Class'].copy()
    X = df.drop(['Class', 'Unnamed: 0'], axis=1)
    X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X, y, test_size=0.20, random_state=42)
    run_and_evaluate_performance(X_train_5, y_train_5, X_test_5, y_test_5)

    # load taiwan credit data and evaluate performance
    # the data is resampled before to loading
    taiwan_credit_data_path = '../data/t_credit_resampled.csv'
    df = pd.read_csv(taiwan_credit_data_path);
    features = ['LIMIT_BAL', 'EDUCATION', 'MARRIAGE', 'PAY_1', 'PAY_2', 'PAY_3',
                'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    y_taiwan = df['def_pay'].copy()
    X_taiwan = df[features].copy()
    X_train_taiwan, X_test_taiwan, y_train_taiwan, y_test_taiwan = train_test_split(X_taiwan, y_taiwan, test_size=0.20,
                                                                                    random_state=42)
    run_and_evaluate_performance(X_train_taiwan, y_train_taiwan, X_test_taiwan, y_test_taiwan)

    # evaluate performance for lending club data
    lending_club_resampled = '../data/lending_resampled.csv';
    df = pd.read_csv(lending_club_resampled);

    y_lending_club = df['defaulted'].copy()  # target
    X_lending_club = df.drop(['defaulted', 'Unnamed: 0'], axis=1)

    X_train_lending, X_test_lending, y_train_lending, y_test_lending = train_test_split(X_lending_club, y_lending_club,
                                                                                        test_size=0.20, random_state=42);

    run_and_evaluate_performance(X_train_lending, y_train_lending, X_test_lending, y_test_lending);









