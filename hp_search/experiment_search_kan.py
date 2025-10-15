import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
import shutil

from models.kan_model import kan_net
from utils.utils_results import load_data, get_width
import time



def model_pipeline(model, x_train, y_train, t_train, x_test, y_test, t_test, plot_flag=False, lamb=0.01, lr=0.001, **kwargs):
    tic = time.time()
    results = model.fit(x_train, y_train, t_train, x_test, y_test, t_test, early_stop=True, patience=30, batch=1000,
                           steps=10000, lamb=lamb, lamb_entropy=0.1, lr=lr, verbose=0, **kwargs)
    training_time = time.time() - tic
    # Plot results
    os.makedirs(model.plot_folder, exist_ok=True)
    plt.plot(results['train_loss'], label='Train loss')
    plt.plot(results['test_loss'], label='Test loss')
    plt.legend()
    plt.savefig(model.plot_folder + '_loss.png', bbox_inches='tight')
    if plot_flag:
        plt.show()
    else:
        plt.close()

    for metric in results['train_metrics'][0].keys():
        plt.plot([r[metric] for r in results['train_metrics']], label=f'Train {metric}')
        plt.plot([r[metric] for r in results['test_metrics']], label=f'Test {metric}')
        plt.legend()
        plt.savefig(model.plot_folder + f'_{metric}.png', bbox_inches='tight')
        if plot_flag:
            plt.show()
        else:
            plt.close()
    if plot_flag:
        model.plot(plot_flag)

    tic = time.time()
    res = model.predict(x_test, t_test)
    inference_time = time.time() - tic
    res['test_loss'] = results['test_loss']
    res['train_loss'] = results['train_loss']
    res['training_time'] = training_time
    res['inference_time'] = inference_time

    return res


def process_job(i, params, dataset):

    data_train, data_test = load_data(dataset, i)

    x_train, y_train, t_train = data_train[[col for col in data_train.columns if 'x' in col]].values, data_train['y_factual'].values[:, None], data_train['treatment'].values[:, None]
    x_test, y_test, t_test = data_test[[col for col in data_test.columns if 'x' in col]].values, data_test['y_factual'].values[:, None], data_test['treatment'].values[:, None]

    real_ite_train = data_train['mu1'].values - data_train['mu0'].values
    real_ite_test = data_test['mu1'].values - data_test['mu0'].values

    r = {'real_ate': np.mean(real_ite_test)}

    for model_name in ['slearner', 'tlearner', 'tarnet', 'dragonnet']:
        width = get_width(model_name, x_train.shape[1], params['hidden_dims'], params['mult_kan'])
        model = kan_net(model_name, width, grid=params['grid'], k=params['k'], seed=i, sparse_init=params['sparse_init'], try_gpu=False, real_ite_train=real_ite_train, real_ite_test=real_ite_test, model_id=f'{model_name}_{i}_{dataset}', save_folder='../checkpoints_search')
        res = model_pipeline(model, x_train, y_train, t_train, x_test, y_test, t_test, lamb=params['lamb'], lr=params['lr'])
        ite = res['y_pred_1'] - res['y_pred_0']
        r[f'ate_{model_name}'] = np.mean(ite)
        r[f'pehe_{model_name}'] = np.sqrt(np.mean((ite - real_ite_test)**2))
        r[f'train_loss_{model_name}'] = res['train_loss']
        r[f'test_loss_{model_name}'] = res['test_loss']
        r[f'training_time_{model_name}'] = res['training_time']

    return r


def run_experiment(params, plot_flag=False, delete_flag=False, n_tasks=25, n_jobs=25):

    if delete_flag:
        # Remove all content in './checkpoints directory'
        if os.path.exists('./checkpoints'):
            shutil.rmtree('./checkpoints')

    dataset = params['dataset']
    out = Parallel(n_jobs=n_jobs, verbose=10)(delayed(process_job)(i + 1, dataset=dataset, params=params) for i in range(n_tasks))

    ate_slearner = [o['ate_slearner'] for o in out]
    pehe_slearner = [o['pehe_slearner'] for o in out]
    ate_tlearner = [o['ate_tlearner'] for o in out]
    pehe_tlearner = [o['pehe_tlearner'] for o in out]
    ate_tarnet = [o['ate_tarnet'] for o in out]
    pehe_tarnet = [o['pehe_tarnet'] for o in out]
    ate_dragonnet = [o['ate_dragonnet'] for o in out]
    pehe_dragonnet = [o['pehe_dragonnet'] for o in out]
    ate_real = [o['real_ate'] for o in out]
    train_loss_slearner = [o['train_loss_slearner'][-1] for o in out]  # Keep only the last loss value
    test_loss_slearner = [o['test_loss_slearner'][-1] for o in out]  # Keep only the last loss value
    train_loss_tlearner = [o['train_loss_tlearner'][-1] for o in out]  # Keep only the last loss value
    test_loss_tlearner = [o['test_loss_tlearner'][-1] for o in out]  # Keep only the last loss value
    train_loss_tarnet = [o['train_loss_tarnet'][-1] for o in out]  # Keep only the last loss value
    test_loss_tarnet = [o['test_loss_tarnet'][-1] for o in out]  # Keep only the last loss value
    train_loss_dragonnet = [o['train_loss_dragonnet'][-1] for o in out]  # Keep only the last loss value
    test_loss_dragonnet = [o['test_loss_dragonnet'][-1] for o in out]  # Keep only the last loss value

    # Prepare a dataframe with the results: one dataframe where colums are the different methods and rows are the different results (mean and std)
    res_dict = {'ate_real': [np.mean(ate_real), np.std(ate_real)],
                'ate_slearner': [np.mean(ate_slearner), np.std(ate_slearner)],
                'pehe_slearner': [np.mean(pehe_slearner), np.std(pehe_slearner)],
                'ate_tlearner': [np.mean(ate_tlearner), np.std(ate_tlearner)],
                'pehe_tlearner': [np.mean(pehe_tlearner), np.std(pehe_tlearner)],
                'ate_tarnet': [np.mean(ate_tarnet), np.std(ate_tarnet)],
                'pehe_tarnet': [np.mean(pehe_tarnet), np.std(pehe_tarnet)],
                'ate_dragonnet': [np.mean(ate_dragonnet), np.std(ate_dragonnet)],
                'pehe_dragonnet': [np.mean(pehe_dragonnet), np.std(pehe_dragonnet)],
                'test_loss_slearner': [np.mean(test_loss_slearner), np.std(test_loss_slearner)],
                'test_loss_tlearner': [np.mean(test_loss_tlearner), np.std(test_loss_tlearner)],
                'test_loss_tarnet': [np.mean(test_loss_tarnet), np.std(test_loss_tarnet)],
                'test_loss_dragonnet': [np.mean(test_loss_dragonnet), np.std(test_loss_dragonnet)],
                'train_loss_slearner': [np.mean(train_loss_slearner), np.std(train_loss_slearner)],
                'train_loss_tlearner': [np.mean(train_loss_tlearner), np.std(train_loss_tlearner)],
                'train_loss_tarnet': [np.mean(train_loss_tarnet), np.std(train_loss_tarnet)],
                'train_loss_dragonnet': [np.mean(train_loss_dragonnet), np.std(train_loss_dragonnet)]}

    if plot_flag:
        df = pd.DataFrame(res_dict, index=['mean', 'std'])
        df.to_csv(f'results_{dataset}.csv')

        print(f"Final results for dataset {dataset}")
        print(df)

        # Plot histograms
        plt.hist([ate_real, ate_slearner, ate_tlearner, ate_tarnet, ate_dragonnet], bins=30, label=['Real', 'S-learner', 'T-learner', 'TARNET', 'DRAGONNET'])
        plt.legend()
        plt.title('ATE')
        plt.savefig(f'ate_hist_{dataset}.png', bbox_inches='tight', dpi=200)
        plt.show()

        plt.hist([pehe_slearner, pehe_tlearner, pehe_tarnet, pehe_dragonnet], bins=30, label=['S-learner', 'T-learner', 'TARNET', 'DRAGONNET'])
        plt.legend()
        plt.title('PEHE')
        plt.savefig(f'pehe_hist_{dataset}.png', bbox_inches='tight', dpi=200)
        plt.show()
    # Add all the keys in the params dictionary to the res_dict
    res_dict.update(params)
    return res_dict



if __name__ == '__main__':

    # Define the loop
    datasets = ['IHDP_A', 'IHDP_B', 'ACIC_2', 'ACIC_7', 'ACIC_26']
    hidden_dims = [0, 5, [5, 5]]
    lambdas = [0.001, 0.01]
    lrs = [0.001]
    grids = [1, 3, 5]
    ks = [1, 3, 5]
    mult_kans = [True, False]  # Only makes sense where there are hidden layers present!
    sparse_inits = [True, False]
    n_tasks = 10  # Number of IHDP datasets to test (max is 100, we use a reduced set to do hyperparamenter tuning for computational reasons). Also, note that we only use 1000 training patients for the ACIC dataset, to speed up the hyperparameter search

    n_jobs = 10  # Number of parallel jobs to run during training
    train_flag = not True  # Whether to train or not
    resume_training = True  # If True, it will resume training from the last saved file, in case there was an error (i.e., if a results file already exists, it skips the training)

    if train_flag:
        os.makedirs('../search_experiment', exist_ok=True)
        for dataset in datasets:
            for hidden_dim in hidden_dims:
                for lamb in lambdas:
                    for lr in lrs:
                        for grid in grids:
                            for sparse_init in sparse_inits:
                                for k in ks:
                                    if hidden_dim == 0:  # Only use mult_kan for hidden layers
                                        mkans = [False]
                                    else:
                                        mkans = mult_kans
                                    for mult_kan in mkans:
                                        file_name = os.path.join('../search_experiment', f"results_{dataset}_{hidden_dim}_{lamb}_{lr}_{grid}_{k}_{sparse_init}_{mult_kan}.pkl")
                                        if os.path.exists(file_name) and resume_training:
                                            print(f"File {file_name} already exists. Skipping...")
                                        else:
                                            params = {'dataset': dataset, 'hidden_dims': hidden_dim, 'lamb': lamb, 'lr': lr, 'grid': grid, 'k': k, 'sparse_init': sparse_init, 'mult_kan': mult_kan}
                                            res = run_experiment(params, plot_flag=False, delete_flag=True, n_tasks=n_tasks, n_jobs=n_jobs)
                                            with open(file_name, 'wb') as f:
                                                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)





