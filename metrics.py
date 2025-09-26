import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
from tabulate import tabulate
import time

from kan_model import kan_net
from utils import load_data, get_width


def process_train_instance_kan(params):  # This is the function that will be parallelized

    dataset = params['dataset']
    task = params['task']
    model_name = params['model_name']

    if not os.path.exists(os.path.join(params['checkpoint_folder'], f'params_{task}.pkl')): # To resume training in case of failure

        np.random.seed(0)  # Seed for reproducibility
        data_train, data_test = load_data(dataset, task)

        x_train = data_train[[col for col in data_train.columns if 'x' in col]].values
        y_train, t_train = data_train['y_factual'].values[:, None], data_train['treatment'].values[:, None]
        x_test = data_test[[col for col in data_test.columns if 'x' in col]].values
        y_test, t_test = data_test['y_factual'].values[:, None], data_test['treatment'].values[:, None]
        real_ite_train = data_train['mu1'].values - data_train['mu0'].values
        real_ite_test = data_test['mu1'].values - data_test['mu0'].values

        params['real_ate'] = np.mean(real_ite_test)
        params['real_ite'] = real_ite_test

        width = get_width(model_name, x_train.shape[1], params['hidden_dims'], params['mult_kan'])
        model = kan_net(model_name, width, grid=params['grid'], k=params['k'], seed=task, sparse_init=params['sparse_init'],
                        try_gpu=False, real_ite_train=real_ite_train, real_ite_test=real_ite_test,
                        model_id=f'{model_name}_{task}_{dataset}', save_folder=params['checkpoint_folder'])
        print(f'training {model_name}...')
        tic = time.time()
        results = model.fit(x_train, y_train, t_train, x_test, y_test, t_test, early_stop=True, patience=30,
                            batch=1000, steps=10000, lamb=params['lamb'], lamb_entropy=0.1, lr=params['lr'], verbose=VERBOSE)
        training_time = time.time() - tic
        params['training_time'] = training_time
        # Plot results
        os.makedirs(model.plot_folder, exist_ok=True)
        plt.figure()
        plt.plot(results['train_loss'], label='Train loss')
        plt.plot(results['test_loss'], label='Test loss')
        plt.legend()
        plt.savefig(model.plot_folder + '_loss.png', bbox_inches='tight')
        plt.close()

        for metric in results['train_metrics'][0].keys():
            plt.plot([r[metric] for r in results['train_metrics']], label=f'Train {metric}')
            plt.plot([r[metric] for r in results['test_metrics']], label=f'Test {metric}')
            plt.legend()
            plt.savefig(model.plot_folder + f'_{metric}.png', bbox_inches='tight')
            plt.close()

        if model_name != 'tlearner':
            model.plot(True)

        tic = time.time()
        res = model.predict(x_test, t_test)
        inference_time = time.time() - tic

        params['ite'] = res['y_pred_1'] - res['y_pred_0']
        params[f'ate'] = np.mean(params['ite'])
        params[f'pehe'] = np.sqrt(np.mean((params['ite'] - real_ite_test) ** 2))
        params['inference_time'] = inference_time
        # Update params with results dict
        params.update(results)
        # Save the params to a file
        with open(os.path.join(params['checkpoint_folder'], f'params_{task}.pkl'), 'wb') as f:
            pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"Skipping {params['checkpoint_folder']}_{task} as it already exists")


def get_dataset_params_kan(datasets, n_tasks, base_folder):  # Note that the hyperparameters are set based on the search experiment results
    out_params = []
    for dataset in datasets:
        if dataset == 'IHDP_A':
            base_params_slearner = {'dataset': 'IHDP_A', 'hidden_dims': 5, 'lamb': 0.01, 'lr': 0.001, 'grid': 1, 'k': 1,
                                    'mult_kan': True, 'sparse_init': False, 'model_name': 'slearner'}
            base_params_tlearner = {'dataset': 'IHDP_A', 'hidden_dims': 5, 'lamb': 0.01, 'lr': 0.001, 'grid': 1, 'k': 5,
                                    'mult_kan': False, 'sparse_init': False, 'model_name': 'tlearner'}
            base_params_tarnet = {'dataset': 'IHDP_A', 'hidden_dims': 5, 'lamb': 0.01, 'lr': 0.001, 'grid': 1, 'k': 5,
                                  'mult_kan': False, 'sparse_init': False, 'model_name': 'tarnet'}
            base_params_dragonnet = {'dataset': 'IHDP_A', 'hidden_dims': 5, 'lamb': 0.01, 'lr': 0.001, 'grid': 1, 'k': 3,
                                     'mult_kan': False, 'sparse_init': False, 'model_name': 'dragonnet'}
        elif dataset == 'IHDP_B':
            base_params_slearner = {'dataset': 'IHDP_B', 'hidden_dims': 5, 'lamb': 0.01, 'lr': 0.001, 'grid': 1, 'k': 3,
                                    'mult_kan': False, 'sparse_init': False, 'model_name': 'slearner'}
            base_params_tlearner = {'dataset': 'IHDP_B', 'hidden_dims': 5, 'lamb': 0.01, 'lr': 0.001, 'grid': 3, 'k': 3,
                                    'mult_kan': False, 'sparse_init': False, 'model_name': 'tlearner'}
            base_params_tarnet = {'dataset': 'IHDP_B', 'hidden_dims': 5, 'lamb': 0.01, 'lr': 0.001, 'grid': 3, 'k': 3,
                                  'mult_kan': True, 'sparse_init': False, 'model_name': 'tarnet'}
            base_params_dragonnet = {'dataset': 'IHDP_B', 'hidden_dims': 0, 'lamb': 0.01, 'lr': 0.001, 'grid': 1, 'k': 3,
                                     'mult_kan': True, 'sparse_init': False, 'model_name': 'dragonnet'}
        elif dataset == 'ACIC_2':
            base_params_slearner = {'dataset': 'ACIC_2', 'hidden_dims': 0, 'lamb': 0.001, 'lr': 0.001, 'grid': 3, 'k': 1,
                                    'mult_kan': False, 'sparse_init': False, 'model_name': 'slearner'}
            base_params_tlearner = {'dataset': 'ACIC_2', 'hidden_dims': [5, 5], 'lamb': 0.01, 'lr': 0.001, 'grid': 1, 'k': 1,
                                    'mult_kan': True, 'sparse_init': True, 'model_name': 'tlearner'}
            base_params_tarnet = {'dataset': 'ACIC_2', 'hidden_dims': [5, 5], 'lamb': 0.01, 'lr': 0.001, 'grid': 1, 'k': 3,
                                  'mult_kan': False, 'sparse_init': False, 'model_name': 'tarnet'}
            base_params_dragonnet = {'dataset': 'ACIC_2', 'hidden_dims': [5, 5], 'lamb': 0.001, 'lr': 0.001, 'grid': 1, 'k': 1,
                                     'mult_kan': False, 'sparse_init': False, 'model_name': 'dragonnet'}
        elif dataset == 'ACIC_7':
            base_params_slearner = {'dataset': 'ACIC_7', 'hidden_dims': [5, 5], 'lamb': 0.01, 'lr': 0.001, 'grid': 1, 'k': 5,
                                    'mult_kan': False, 'sparse_init': False, 'model_name': 'slearner'}
            base_params_tlearner = {'dataset': 'ACIC_7', 'hidden_dims': 0, 'lamb': 0.01, 'lr': 0.001, 'grid': 3, 'k': 3,
                                    'mult_kan': False, 'sparse_init': False, 'model_name': 'tlearner'}
            base_params_tarnet = {'dataset': 'ACIC_7', 'hidden_dims': 0, 'lamb': 0.01, 'lr': 0.001, 'grid': 3, 'k': 3,
                                  'mult_kan': False, 'sparse_init': False, 'model_name': 'tarnet'}
            base_params_dragonnet = {'dataset': 'ACIC_7', 'hidden_dims': 0, 'lamb': 0.01, 'lr': 0.001, 'grid': 3, 'k': 3,
                                     'mult_kan': False, 'sparse_init': False, 'model_name': 'dragonnet'}
        elif dataset == 'ACIC_26':
            base_params_slearner = {'dataset': 'ACIC_26', 'hidden_dims': 5, 'lamb': 0.01, 'lr': 0.001, 'grid': 1, 'k': 3,
                                    'mult_kan': False, 'sparse_init': False, 'model_name': 'slearner'}
            base_params_tlearner = {'dataset': 'ACIC_26', 'hidden_dims': 0, 'lamb': 0.01, 'lr': 0.001, 'grid': 3, 'k': 1,
                                    'mult_kan': False, 'sparse_init': True, 'model_name': 'tlearner'}
            base_params_tarnet = {'dataset': 'ACIC_26', 'hidden_dims': 0, 'lamb': 0.01, 'lr': 0.001, 'grid': 3, 'k': 1,
                                  'mult_kan': False, 'sparse_init': False, 'model_name': 'tarnet'}
            base_params_dragonnet = {'dataset': 'ACIC_26', 'hidden_dims': 0, 'lamb': 0.01, 'lr': 0.001, 'grid': 3, 'k': 1,
                                     'mult_kan': False, 'sparse_init': False, 'model_name': 'dragonnet'}
        else:
            raise ValueError(f'Dataset {dataset} not found')

        for p in [base_params_slearner, base_params_tlearner, base_params_tarnet, base_params_dragonnet]:
            p['checkpoint_folder'] = os.path.join(base_folder, f"{p['model_name']}_{dataset}")
            os.makedirs(p['checkpoint_folder'], exist_ok=True)
            for task in range(n_tasks):
                out_params.append({**p, 'task': task + 1})
    return out_params


import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
from tabulate import tabulate
import time

from mlp_model import mlp_net
from utils import load_data, get_dims_mlp


def process_train_instance_mlp(params):  # This is the function that will be parallelized

    dataset = params['dataset']
    task = params['task']
    model_name = params['model_name']

    if not os.path.exists(os.path.join(params['checkpoint_folder'], f'params_{task}.pkl')): # To resume training in case of failure

        np.random.seed(0)  # Seed for reproducibility
        data_train, data_test = load_data(dataset, task)

        x_train = data_train[[col for col in data_train.columns if 'x' in col]].values
        y_train, t_train = data_train['y_factual'].values[:, None], data_train['treatment'].values[:, None]
        x_test = data_test[[col for col in data_test.columns if 'x' in col]].values
        y_test, t_test = data_test['y_factual'].values[:, None], data_test['treatment'].values[:, None]
        real_ite_train = data_train['mu1'].values - data_train['mu0'].values
        real_ite_test = data_test['mu1'].values - data_test['mu0'].values

        params['real_ate'] = np.mean(real_ite_test)
        params['real_ite'] = real_ite_test
        input_dim = x_train.shape[1]
        dims = get_dims_mlp(model_name, input_dim, params['hidden_dims'])
        model = mlp_net(model_name, dims, seed=task,
                        try_gpu=False, real_ite_train=real_ite_train, real_ite_test=real_ite_test,
                        model_id=f'{model_name}_{task}_{dataset}', save_folder=params['checkpoint_folder'])
        print(f'\n training {model_name} on {dataset} task {task} \n')
        tic = time.time()
        results = model.fit(x_train, y_train, t_train, x_test, y_test, t_test, early_stop=True, patience=100,
                            batch=1000, steps=10000, lr=params['lr'], verbose=VERBOSE,
                            lr_scheduler='plateau')
        training_time = time.time() - tic
        params['training_time'] = training_time
        # Plot results
        os.makedirs(model.plot_folder, exist_ok=True)
        plt.plot(results['train_loss'], label='Train loss')
        plt.plot(results['test_loss'], label='Test loss')
        plt.legend()
        plt.savefig(model.plot_folder + '_loss.png', bbox_inches='tight')
        plt.close()

        for metric in results['train_metrics'][0].keys():
            plt.plot([r[metric] for r in results['train_metrics']], label=f'Train {metric}')
            plt.plot([r[metric] for r in results['test_metrics']], label=f'Test {metric}')
            plt.legend()
            plt.savefig(model.plot_folder + f'_{metric}.png', bbox_inches='tight')
            plt.close()

        tic = time.time()
        res = model.predict(x_test, t_test)
        inference_time = time.time() - tic
        params['inference_time'] = inference_time

        params['ite'] = res['y_pred_1'] - res['y_pred_0']
        params[f'ate'] = np.mean(params['ite'])
        params[f'pehe'] = np.sqrt(np.mean((params['ite'] - real_ite_test) ** 2))
        # Update params with results dict
        params.update(results)
        # Save the params to a file
        with open(os.path.join(params['checkpoint_folder'], f'params_{task}.pkl'), 'wb') as f:
            pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"Skipping {params['checkpoint_folder']}_{task} as it already exists")


def get_dataset_params_mlp(datasets, n_tasks, base_folder):  # Note that the hyperparameters are set based on the search experiment results
    out_params = []
    for dataset in datasets:
        if dataset == 'IHDP_A':
            base_params_slearner = {'dataset': 'IHDP_A', 'hidden_dims': [100, 100, 100], 'lr': 0.001, 'activation': 'elu', 'model_name': 'slearner'}
            base_params_tlearner = {'dataset': 'IHDP_A', 'hidden_dims': [100, 100, 100], 'lr': 0.0001, 'activation':'elu', 'model_name': 'tlearner'}
            base_params_tarnet = {'dataset': 'IHDP_A', 'hidden_dims': [[200, 200, 200, 200], [100, 100, 100]], 'lr': 0.001, 'activation':'elu', 'model_name': 'tarnet'}
            base_params_dragonnet = {'dataset': 'IHDP_A', 'hidden_dims': [[200, 200, 200, 200], [100, 100, 100], []], 'lr': 0.001, 'activation':'elu', 'model_name': 'dragonnet'}
        elif dataset == 'IHDP_B':
            base_params_slearner = {'dataset': 'IHDP_B', 'hidden_dims': [100, 100, 100], 'lr': 0.001, 'activation': 'elu', 'model_name': 'slearner'}
            base_params_tlearner = {'dataset': 'IHDP_B', 'hidden_dims': [100, 100, 100], 'lr': 0.0001, 'activation': 'elu', 'model_name': 'tlearner'}
            base_params_tarnet = {'dataset': 'IHDP_B', 'hidden_dims': [[100,100], [50, 50]], 'lr': 0.0001, 'activation': 'elu', 'model_name': 'tarnet'}
            base_params_dragonnet = {'dataset': 'IHDP_B', 'hidden_dims': [[200, 200, 200], [100, 100], []], 'lr': 0.00001, 'activation': 'elu', 'model_name': 'dragonnet'}
        elif dataset == 'ACIC_2':
            base_params_slearner = {'dataset': 'ACIC_2', 'hidden_dims': [100, 100, 100], 'lr': 0.0001, 'activation': 'elu',
                                    'model_name': 'slearner'}
            base_params_tlearner = {'dataset': 'ACIC_2', 'hidden_dims': [100, 100, 100], 'lr': 0.00001, 'activation': 'relu',
                                    'model_name': 'tlearner'}
            base_params_tarnet = {'dataset': 'ACIC_2', 'hidden_dims': [[200, 200, 200, 200], [100, 100, 100]], 'lr': 0.0001, 'activation': 'elu',
                                  'model_name': 'tarnet'}
            base_params_dragonnet = {'dataset': 'ACIC_2', 'hidden_dims': [[200, 200, 200, 200], [100, 100, 100], []], 'lr': 0.0001, 'activation': 'elu',
                                     'model_name': 'dragonnet'}
        elif dataset == 'ACIC_7':
            base_params_slearner = {'dataset': 'ACIC_7', 'hidden_dims': [100, 100, 100], 'lr': 0.001, 'activation': 'elu',
                                    'model_name': 'slearner'}
            base_params_tlearner = {'dataset': 'ACIC_7', 'hidden_dims': [100, 100, 100], 'lr': 0.0001, 'activation': 'leaky_relu',
                                    'model_name': 'tlearner'}
            base_params_tarnet = {'dataset': 'ACIC_7', 'hidden_dims': [[200, 200, 200], [100, 100]], 'lr': 0.0001, 'activation': 'elu',
                                  'model_name': 'tarnet'}
            base_params_dragonnet = {'dataset': 'ACIC_7', 'hidden_dims': [[200, 200, 200], [100, 100], []], 'lr': 0.0001, 'activation': 'elu',
                                     'model_name': 'dragonnet'}
        elif dataset == 'ACIC_26':
            base_params_slearner = {'dataset': 'ACIC_7', 'hidden_dims': [100, 100, 100], 'lr': 0.001, 'activation': 'elu',
                                    'model_name': 'slearner'}
            base_params_tlearner = {'dataset': 'ACIC_7', 'hidden_dims': [100, 100, 100], 'lr': 0.0001, 'activation': 'leaky_relu',
                                    'model_name': 'tlearner'}
            base_params_tarnet = {'dataset': 'ACIC_7', 'hidden_dims': [[200, 200, 200], [100, 100]], 'lr': 0.0001, 'activation': 'elu',
                                  'model_name': 'tarnet'}
            base_params_dragonnet = {'dataset': 'ACIC_7', 'hidden_dims': [[200, 200, 200], [100, 100], []], 'lr': 0.0001, 'activation': 'elu',
                                     'model_name': 'dragonnet'}
        else:
            raise ValueError(f'Dataset {dataset} not found')

        for p in [base_params_slearner, base_params_tlearner, base_params_tarnet, base_params_dragonnet]:
            p['checkpoint_folder'] = os.path.join(base_folder, f"{p['model_name']}_{dataset}")
            os.makedirs(p['checkpoint_folder'], exist_ok=True)
            for task in range(n_tasks):
                out_params.append({**p, 'task': task + 1})
    return out_params

def map_names_kan(name):
    if name == 'slearner':
        return 'S-KAN'
    elif name == 'tlearner':
        return 'T-KAN'
    elif name == 'tarnet':
        return 'TARKAN'
    elif name == 'dragonnet':
        return 'DragonKAN'
    else:
        return name

if __name__ == '__main__':
    # First, define all the needed parameters
    # First, define all the needed parameters
    train_flag = True  # If false, only the results stored are plot
    datasets = ['IHDP_A', 'IHDP_B', 'ACIC_2', 'ACIC_7', 'ACIC_26']
    n_tasks = 100  # Number of datasets to use
    n_jobs = 25  # Number of parallel jobs
    models = ['slearner', 'tlearner', 'tarnet', 'dragonnet']

    DEBUG = False
    VERBOSE = 0

    load_results = True
    results_path = 'results_dict.pkl'

    if load_results and os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            results_dict = pickle.load(f)
        # If results are loaded, no need to train
        train_flag = False

        with open('results_tab.txt', 'r') as f:
            results_tab = f.read()
    else:

        if DEBUG:
            n_jobs = 1
            n_tasks = 1
            VERBOSE = 1


        base_folder_kan = os.path.join(os.getcwd(), 'results_kan_metrics')
        os.makedirs(base_folder_kan, exist_ok=True)

        base_folder_mlp = os.path.join(os.getcwd(), 'results_mlp_metrics')
        os.makedirs(base_folder_mlp, exist_ok=True)

        params_kan = get_dataset_params_kan(datasets, n_tasks, base_folder_kan)
        params_mlp = get_dataset_params_mlp(datasets, n_tasks, base_folder_mlp)



        if train_flag:
            results_kan = Parallel(n_jobs=n_jobs)(delayed(process_train_instance_kan)(p) for p in params_mlp)
            results_mlp = Parallel(n_jobs=n_jobs)(delayed(process_train_instance_mlp)(p) for p in params_kan)

        # Load and plot the results
        results_tab = []
        results_tab_names = ['Dataset', 'Model name', 'Real ATE', 'ATE', 'ATE MAE', 'PEHE']
        results_dict = {}
        for dataset in datasets:
            results_dataset = {}
            for model in models:
                results_folder_kan = os.path.join(base_folder_kan, f'{model}_{dataset}')
                results_folder_mlp = os.path.join(base_folder_mlp, f'{model}_{dataset}')
                results_kan = []
                results_mlp = []
                for task in range(n_tasks):
                    with open(os.path.join(results_folder_kan, f'params_{task + 1}.pkl'), 'rb') as f:
                        results_kan.append(pickle.load(f))
                    with open(os.path.join(results_folder_mlp, f'params_{task + 1}.pkl'), 'rb') as f:
                        results_mlp.append(pickle.load(f))

                # Now, compute the average results, print them, and plot them
                real_ate_kan = [r['real_ate'] for r in results_kan]
                ate_kan = [r['ate'] for r in results_kan]
                ate_error_kan = [np.abs(r['ate'] - r['real_ate']) for r in results_kan]
                pehe_kan = [r['pehe'] for r in results_kan]

                real_ate_mlp = [r['real_ate'] for r in results_mlp]
                ate_mlp = [r['ate'] for r in results_mlp]
                ate_error_mlp = [np.abs(r['ate'] - r['real_ate']) for r in results_mlp]
                pehe_mlp = [r['pehe'] for r in results_mlp]

                if 'training_time' in results_kan[0]:
                    training_time_kan = [r['training_time'] for r in results_kan]
                    inference_time_kan = [r['inference_time'] for r in results_kan]
                    training_time_mlp = [r['training_time'] for r in results_mlp]
                    inference_time_mlp = [r['inference_time'] for r in results_mlp]

                    results_tab.append([dataset,
                                        map_names_kan(model),
                                        f'{np.mean(real_ate_kan):.3f} ({np.std(real_ate_kan):.3f})',
                                        f'{np.mean(ate_kan):.3f} ({np.std(ate_kan):.3f})',
                                        f'{np.mean(ate_error_kan):.3f} ({np.std(ate_error_kan):.3f})',
                                        f'{np.mean(pehe_kan):.3f} ({np.std(pehe_kan):.3f})',
                                        f'{np.mean(training_time_kan):.3f} ({np.std(training_time_kan):.3f})',
                                        f'{np.mean(inference_time_kan):.3f} ({np.std(inference_time_kan):.3f})'])

                    results_tab.append([dataset,
                                        model,
                                        f'{np.mean(real_ate_mlp):.3f} ({np.std(real_ate_mlp):.3f})',
                                        f'{np.mean(ate_mlp):.3f} ({np.std(ate_mlp):.3f})',
                                        f'{np.mean(ate_error_mlp):.3f} ({np.std(ate_error_mlp):.3f})',
                                        f'{np.mean(pehe_mlp):.3f} ({np.std(pehe_mlp):.3f})',
                                        f'{np.mean(training_time_mlp):.3f} ({np.std(training_time_mlp):.3f})',
                                        f'{np.mean(inference_time_mlp):.3f} ({np.std(inference_time_mlp):.3f})'])




                else:
                    results_tab.append([dataset,
                                        map_names_kan(model),
                                        f'{np.mean(real_ate_kan):.3f} ({np.std(real_ate_kan):.3f})',
                                        f'{np.mean(ate_kan):.3f} ({np.std(ate_kan):.3f})',
                                        f'{np.mean(ate_error_kan):.3f} ({np.std(ate_error_kan):.3f})',
                                        f'{np.mean(pehe_kan):.3f} ({np.std(pehe_kan):.3f})'])

                    results_tab.append([dataset,
                                        model,
                                        f'{np.mean(real_ate_mlp):.3f} ({np.std(real_ate_mlp):.3f})',
                                        f'{np.mean(ate_mlp):.3f} ({np.std(ate_mlp):.3f})',
                                        f'{np.mean(ate_error_mlp):.3f} ({np.std(ate_error_mlp):.3f})',
                                        f'{np.mean(pehe_mlp):.3f} ({np.std(pehe_mlp):.3f})'])

                results_dataset[map_names_kan(model)] = results_kan
                results_dataset[model] = results_mlp

                results_dict[f'{map_names_kan(model)}_{dataset}'] = {'PEHE': pehe_kan, 'ATE MAE': ate_error_kan}
                results_dict[f'{model}_{dataset}'] = {'PEHE': pehe_mlp, 'ATE MAE': ate_error_mlp}

            # NOTE: other plots can be obtained here (histograms, etc...)
        # Print the results, limit to 3 decimal places
        if len(results_tab[0]) == 8:
            results_tab_names.append('Training time (s)')
            results_tab_names.append('Inference time (s)')


    # save results_dict
        with open('results_dict.pkl', 'wb') as f:
            pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    # save results tab
        with open('results_tab.txt', 'w') as f:
            f.write(tabulate(results_tab, headers=results_tab_names, tablefmt='grid'))

    print(results_tab)

    # Now, with the table, perform statistical tests on the means of those quantities
    from utils import get_p_values_from_table_data, plot_kan_mlp_boxplots, make_grid_2x5
    # pehe

    # construct a table methods_to_compare x metrics x datasets/folds and store in a Numpy array
    methods_to_compare = ['S-KAN', 'T-KAN', 'TARKAN', 'DragonKAN', 'slearner', 'tlearner', 'tarnet', 'dragonnet']
    metrics = ['PEHE', 'ATE MAE']
    datasets = ['IHDP_A', 'IHDP_B', 'ACIC_2', 'ACIC_7', 'ACIC_26']
    data_map = {}
    # Make 1 test per dataset (in each dataset there are 100 folds)
    for dataset in datasets:
        # determine the number of datasets
        dataset_length = len(results_dict[f'{methods_to_compare[0]}_{datasets[0]}'][metrics[0]])
        np_array = np.zeros((len(methods_to_compare), len(metrics), dataset_length))
        for i in range(len(methods_to_compare)):
            for j in range(len(metrics)):
                np_array[i, j, :] = results_dict[f'{methods_to_compare[i]}_{dataset}'][metrics[j]]

        print('DATASET', dataset)
        get_p_values_from_table_data(np_array, alpha=0.05, higher_is_better=False, output_latex=False,
                                    list_of_methods=methods_to_compare, list_of_metrics=metrics)


        transparency = 0.9
        from tueplots import axes, bundles, figsizes, fonts

        with plt.rc_context({**bundles.iclr2024(ncols=2),
                             # **fonts.iclr2024(),
                             **axes.lines()}):
            # Plots and wilcoxon tests
            fig_handles = plot_kan_mlp_boxplots(np_array, metrics=['PEHE', 'ATE MAE'], methods_to_compare=methods_to_compare,
                                  dataset_name=dataset, alpha=0.1, plot_showfliers=False, transparency=transparency)
            for i, fig in enumerate(fig_handles):
                fig.savefig(f"figures/boxplot_{metrics[i].replace(' ', '_')}_{dataset}.pdf", bbox_inches='tight')
                plt.close(fig)
        data_map[dataset] = np_array

            # Build your 5-dataset map (each value: (8,2,100) array)
    with plt.rc_context({**bundles.iclr2024(ncols=5, nrows=2),
                         # **fonts.iclr2024(),
                         **axes.lines()}):

        fig, axs = make_grid_2x5(
            data_map,
            kan_color='tab:blue',
            mlp_color='tab:orange',
            alpha=0.1,
            plot_whis=1.5,
            plot_showfliers=False,
            transparency=0.9
        )
        fig.savefig('figures/boxplots_2x5_shared_legend.pdf', bbox_inches='tight')
        plt.close(fig)



















