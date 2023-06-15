"""
This script has a simple setup of the most important steps for machine learning on tabular data. At the moment it is set up for regression, but can be easily adapted for classification.
"""
import random
from pathlib import Path
from typing import List
from datetime import datetime
from copy import deepcopy
import shutil

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from MyFirstML.machine_learning.RunML import RunML
from MyFirstML.utils.input_output import make_new_output_directory, write_to_yaml
from MyFirstML.utils.ml_utils import load_data, get_hparams, get_train_test_splits


def get_all_models(hparams: dict, n_features: int, use_models: List[str]) -> dict:
    """
    Get all models that are to be used in the machine learning run. This function contains a lot of models with brief explanations of how these models work in the comments. The models are divided into two categories: simple models and complex models. Simple models are models that are easy to understand and interpret, and are therefore useful for debugging purposes. Complex models are models that are more powerful, but also more complex and harder to understand and interpret. These models are therefore not useful for debugging purposes, but are useful for getting good predictions.
    @param hparams: Hyperparameters.
    @param n_features: Number of features.
    @param use_models: List of model names to use.
    @return: Dictionary of all models with their names as keys.
    """
    all_models = {}

    ############ 1 NEAREST NEIGHBOR (1NN) ############
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    # 1NN is a useful model particularly for debugging purposes, since it is very simple and easy to understand. It is also a good baseline model to compare other models to. Setting n_neighbors>1 might improve the performance of the model, but it will also make it more complex and harder to understand. Therefore, we only use n_neighbors=1, since it is meant for debugging and baseline purposes.
    if '1NN' in use_models:
        Nearest_Neighbors = KNeighborsRegressor(n_neighbors=1)
        all_models['1NN'] = Nearest_Neighbors

    ############ Linear Regression (LR) ############
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    # Linear regression is a simple model that is easy to understand and interpret. It is also a good baseline model to compare other models to. However, it is not very powerful in non-linear problems, so it is not expected to perform very well for many problems. This model is primarily meant for debugging and baseline purposes.
    if 'LR' in use_models:
        Linear_Regression = LinearRegression()
        all_models['LR'] = Linear_Regression

    ############ Random Forest (RF) ############
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    # Random Forests are quite powerful and extremely useful models in many applications. Often, they will be the best choice. They are relatively robust to hyperparameter settings, so they are easy to use. They will often outperform Neural Networks for smaller data sizes with hundreds of data points. A useful feature of Random Forests and derivative models is that they can be used to estimate the importance of each feature for the prediction. This can be useful for understanding the data and the model.

    if 'RF' in use_models:
        Random_Forest = RandomForestRegressor(
                                                n_estimators=hparams['RF_n_estimators'],
                                                max_depth=hparams['RF_max_depth'],
                                                max_features=hparams['RF_max_features'],
                                                min_samples_leaf=hparams['RF_min_samples_leaf'],
                                                min_impurity_decrease=hparams['RF_min_impurity_decrease']
                                                )
        all_models['RF'] = Random_Forest

    ############ XGBoost (XGB) ############
    # https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
    # Very similar to Random Forests with similar strengths and weaknesses. However, XGB is much faster and often performs a little better. Many people swear on it as the best choice because it is blazing fast, high-performant and quite robust to overfitting. Like Random Forests, it can be used to estimate the importance of each feature for the prediction. It even supports missing values in the features, which is very useful.
    if 'XGB' in use_models:
        XGBoost = XGBRegressor(
                                n_estimators=hparams['XGB_n_estimators'],
                                max_depth=hparams['XGB_max_depth'],
                                learning_rate=hparams['XGB_learning_rate']
                                )
        all_models['XGB'] = XGBoost

    ############ Neural Network (NN) ############
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    # Neural networks are very powerful machine learning models. They are also very complex and hard to understand. For many problems, Neural Networks will most likely be the best performing ML models for data sizes of at least a few thousand data points. However, they are also  computationally a little expensive, so they are not always feasible to use. Tuning the hyperparameters of neural networks can make them perform much better, but is tricky and time-consuming. Neural Networks also easily overfit to data.
    if 'NN' in use_models:
        NN = MLPRegressor(
                            hidden_layer_sizes=hparams['NN_hidden_layer_sizes'],
                            activation=hparams['NN_act'],
                            solver=hparams['NN_solver'],
                            max_iter=hparams['NN_epochs'],
                            early_stopping=hparams['NN_early_stopping'],
                            validation_fraction=hparams['NN_validation_fraction'],
                            alpha=hparams['NN_alpha'],
                            batch_size=hparams['NN_batch_size'],
                            learning_rate_init=hparams['NN_learning_rate'],
                            )
        all_models['NN'] = NN

    ############ Gaussian Process (GP) ############
    # https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
    # Gaussian Processes are quite interesting models. On problems with very little data (tens to hundreds of data points) they often outperform other models, but they scale badly to larger data sizes. One very interesting property is that they have an inbuilt uncertainty estimate, which can be very useful. For larger data sizes, one can find implementations of Sparse Gaussian Processes by gpflow or GPyTorch, which are very fast and can also be used for very large data sizes.
    if 'GP' in use_models:
        lengthscales = np.full(n_features, hparams['GP_length_scale'])
        kernel = ConstantKernel() * RBF(length_scale=lengthscales)      # The kernel is another hyperparameter to tune.
        GP = sklearn.gaussian_process.GaussianProcessRegressor(
                                                                kernel=kernel,
                                                                alpha=hparams['GP_alpha'],
                                                                normalize_y=hparams['GP_normalize_y']
                                                                )
        all_models['GP'] = GP

    # Sort all_models in order of use_models
    all_models = {k: deepcopy(all_models[k]) for k in use_models}

    return all_models


def main(experiment, dataset, reference_run, features, target, CV, n_reps, trainfrac, group, scores, outdir, hparams_file, use_data_frac, xscaler, yscaler, random_seed, use_models):

    ##############################################
    # Starting the experiment
    ##############################################
    print(f'Starting experiment "{experiment}".')

    # Record start time for printing the duration of the run.
    starttime = datetime.now()

    # Set random seeds for deterministic results. If you ever use tensorflow or pytorch, you will need to set their own random seeds as well.
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Load hyperparameters from file
    hparams = get_hparams(hparams_file=hparams_file)

    # Define models
    models = get_all_models(hparams=hparams, n_features=len(features), use_models=use_models)

    # Load data from .csv into a pd.DataFrame
    df = load_data(
                    dataset=dataset,
                    features=features,
                    targets=target,
                    use_data_frac=use_data_frac,
                    shuffle=True,
                    reset_index=True,
                    header=1,
                    )

    # Split data into test and train set
    df, CV_cols = get_train_test_splits(df, CV, n_reps, trainfrac=trainfrac, group=group)

    # Make new output directory in /rootdir
    run_outdir = make_new_output_directory(rootdir=outdir, label=experiment)

    # Save hyperparameters to file in output directory
    shutil.copy(str(hparams_file), str(run_outdir))

    # Save experiment settings to file in output directory
    write_to_yaml(
        {
            'experiment': experiment,
            'dataset': dataset,
            'features': features,
            'target': target,
            'CV': CV,
            'n_reps': n_reps,
            'trainfrac': trainfrac,
            'group': group,
            'scores': list(scores.keys()),
            'outdir': outdir,
            'hparams_file': hparams_file,
            'use_data_frac': use_data_frac,
            'xscaler': str(xscaler),
            'yscaler': str(yscaler),
            'random_seed': random_seed,
            'use_models': use_models,
        },
        output_path=Path(run_outdir, 'settings.yml'),
        comment=None,
    )

    # Run experiment
    ml = RunML(
        df=df,
        models=models,
        features=features,
        target=target,
        CV_cols=CV_cols,
        scores=scores,
        outdir=run_outdir,
        xscaler=xscaler,
        yscaler=yscaler,
    )
    ml.run()

    # For refactoring: Check if the output is still the same. This is very useful when refactoring and writing code, to know exactly which files have changed.
    if reference_run is not None:
        ml.check_if_output_same_as_reference(reference_run=reference_run, detailed=False)

    # Print duration of the run
    duration = datetime.now() - starttime
    print(f'\nScript duration:  {duration}')

    print('Done.')

    return ml


if __name__ == '__main__':

    ###### START OF OPTIONS ######
    # Please set these options to control the ML script.

    # General
    experiment = 'test'     # str: name of experiment for labeling the output directory and printing.
    random_seed = 43        # [int,None]: random seed for reproducibility. Set to None for non-deterministic results.
    use_models = ['1NN', 'LR', 'RF', 'XGB', 'NN', 'GP']                     # list: models to use out of ['1NN', 'LR', 'RF', 'XGB', 'NN', 'GP']
    reference_run = Path('..', '..', 'results', 'runs', 'results_0_test')   # Provide a path of a previous run directory to check for changes in output files. Incredibly useful for refactoring and writing code.
    # Cross validation
    CV = 'KFold'            # str: cross-validation method: 'KFold' or 'Random'
    n_reps = 5              # int: number of repetitions for cross-validation (the K in Kfold or the number of repetitions for Random)
    trainfrac = 0.8         # float: fraction of data to use for training. Only used if CV == 'Random'
    group = None            # [str,None]: grouping variable for cross-validation. None for no grouping.
    # Data
    dataset = Path('..', '..', 'data', 'toy_datasets', 'regression', 'diabetes_toy_dataset.csv')        # str: path to dataset
    use_data_frac = None            # [float,None]: desired fraction of data points in range (0,1) or None for using all data.
    features = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']         # list: features to use
    target = 'target'               # str: target to predict
    xscaler = StandardScaler()       # scaler for scaling the input features before feeding into the model, None for no scaling
    yscaler = StandardScaler()       # scaler for scaling the input targets before feeding into the model, None for no scaling
    scores = {                      # dict: scores to use for evaluation of models
                'r2': sklearn.metrics.r2_score,
                'MAE': sklearn.metrics.mean_absolute_error,
                # 'MSE': sklearn.metrics.mean_squared_error,
                }

    # Secondary options you will usually not need to change (but you can if you want to)
    hparams_file = 'hparams.yml'  # str: path to hyperparameters for models
    outdir = Path('', '../..', 'results', 'runs')        # directory for saving results, in which a new directory will be created for each run

    ###### END OF OPTIONS ######


    # Run ML pipeline
    ml = main(
                experiment=experiment,
                use_models=use_models,
                reference_run=reference_run,
                features=features,
                target=target,
                CV=CV,
                n_reps=n_reps,
                trainfrac=trainfrac,
                group=group,
                dataset=dataset,
                hparams_file=hparams_file,
                use_data_frac=use_data_frac,
                random_seed=random_seed,
                scores=scores,
                outdir=outdir,
                xscaler=xscaler,
                yscaler=yscaler,
                )













        
        