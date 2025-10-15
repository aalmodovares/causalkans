import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
import shutil
from scipy.stats import ttest_ind_from_stats
import time

# from kan_model import kan_net
from models.mlp_model import mlp_net
from utils.utils_results import load_data, get_dims_mlp


def model_pipeline(model, x_train, y_train, t_train, x_test, y_test, t_test, plot_flag=False, lr=0.001, **kwargs):
    tic = time.time()
    results = model.fit(x_train, y_train, t_train, x_test, y_test, t_test, early_stop=True, patience=30, batch=1000,
                           steps=EPOCHS, lr=lr, verbose=VERBOSE, **kwargs)
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
        dims = get_dims_mlp(model_name, x_train.shape[1], params['hidden_dims'][model_name])
        model = mlp_net(model_name, dims, seed=i, try_gpu=False, real_ite_train=real_ite_train, real_ite_test=real_ite_test, model_id=f'{model_name}_{i}_{dataset}',
                        save_folder='../checkpoints_search_mlp', activation=params['activation'], dropout=params['dropout'])
        res = model_pipeline(model, x_train, y_train, t_train, x_test, y_test, t_test, lr=params['lr'])
        ite = res['y_pred_1'] - res['y_pred_0']
        r[f'ate_{model_name}'] = np.mean(ite)
        r[f'pehe_{model_name}'] = np.sqrt(np.mean((ite - real_ite_test)**2))
        r[f'train_loss_{model_name}'] = res['train_loss']
        r[f'test_loss_{model_name}'] = res['test_loss']
        r[f'training_time_{model_name}'] = res['training_time']
        r[f'inference_time_{model_name}'] = res['inference_time']

    return r


def run_experiment(params, plot_flag=False, delete_flag=False, n_tasks=25, n_jobs=25):

    if delete_flag:
        # Remove all content in './checkpoints directory'
        if os.path.exists('./checkpoints_mlp'):
            shutil.rmtree('./checkpoints_mlp')

    dataset = params['dataset']
    out = Parallel(n_jobs=n_jobs, verbose=10)(delayed(process_job)(i + 1, dataset=dataset, params=params) for i in range(n_tasks))

    ate_slearner = [o['ate_slearner'] for o in out]
    pehe_slearner = [o['pehe_slearner'] for o in out]
    training_time_slearner = [o['training_time_slearner'] for o in out]
    inference_time_slearner = [o['inference_time_slearner'] for o in out]
    train_loss_slearner = [o['train_loss_slearner'][-1] for o in out]  # Keep only the last loss value
    test_loss_slearner = [o['test_loss_slearner'][-1] for o in out]  # Keep only the last loss value

    ate_tlearner = [o['ate_tlearner'] for o in out]
    pehe_tlearner = [o['pehe_tlearner'] for o in out]
    training_time_tlearner = [o['training_time_tlearner'] for o in out]
    inference_time_tlearner = [o['inference_time_tlearner'] for o in out]
    train_loss_tlearner = [o['train_loss_tlearner'][-1] for o in out]  # Keep only the last loss value
    test_loss_tlearner = [o['test_loss_tlearner'][-1] for o in out]  # Keep only the last loss value

    ate_tarnet = [o['ate_tarnet'] for o in out]
    pehe_tarnet = [o['pehe_tarnet'] for o in out]
    training_time_tarnet = [o['training_time_tarnet'] for o in out]
    inference_time_tarnet = [o['inference_time_tarnet'] for o in out]
    train_loss_tarnet = [o['train_loss_tarnet'][-1] for o in out]  # Keep only the last loss value
    test_loss_tarnet = [o['test_loss_tarnet'][-1] for o in out]  # Keep only the last loss value

    ate_dragonnet = [o['ate_dragonnet'] for o in out]
    pehe_dragonnet = [o['pehe_dragonnet'] for o in out]
    training_time_dragonnet = [o['training_time_dragonnet'] for o in out]
    inference_time_dragonnet = [o['inference_time_dragonnet'] for o in out]
    train_loss_dragonnet = [o['train_loss_dragonnet'][-1] for o in out]  # Keep only the last loss value
    test_loss_dragonnet = [o['test_loss_dragonnet'][-1] for o in out]  # Keep only the last loss value


    ate_real = [o['real_ate'] for o in out]





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
                'train_loss_dragonnet': [np.mean(train_loss_dragonnet), np.std(train_loss_dragonnet)],
                'training_time_slearner': [np.mean(training_time_slearner), np.std(training_time_slearner)],
                'training_time_tlearner': [np.mean(training_time_tlearner), np.std(training_time_tlearner)],
                'training_time_tarnet': [np.mean(training_time_tarnet), np.std(training_time_tarnet)],
                'training_time_dragonnet': [np.mean(training_time_dragonnet), np.std(training_time_dragonnet)],
                'inference_time_slearner': [np.mean(inference_time_slearner), np.std(inference_time_slearner)],
                'inference_time_tlearner': [np.mean(inference_time_tlearner), np.std(inference_time_tlearner)],
                'inference_time_tarnet': [np.mean(inference_time_tarnet), np.std(inference_time_tarnet)],
                'inference_time_dragonnet': [np.mean(inference_time_dragonnet), np.std(inference_time_dragonnet)],
                }

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

    VERBOSE = 0
    EPOCHS=10000
    DEBUG = False

    # Define the loop
    datasets = ['IHDP_A', 'IHDP_B', 'ACIC_2', 'ACIC_7', 'ACIC_26']
    hidden_dims = [{'slearner': [50,50],
                    'tlearner':[50,50],
                    'tarnet':[[100,100], [50,50]],
                    'dragonnet':[[100,100], [50,50], []],},
                   # --------1---------------
                   {'slearner': [100, 100],
                    'tlearner': [100, 100],
                    'tarnet': [[200,200,200],[100,100]],
                    'dragonnet': [[200, 200, 200], [100, 100], []],},
                   # --------2---------------
                   {'slearner': [100, 100, 100],
                    'tlearner': [100, 100, 100],
                    'tarnet': [[200, 200, 200, 200], [100, 100, 100]],
                    'dragonnet': [[200, 200, 200, 200], [100, 100, 100], []]}
                   ]
    lrs = [1e-3, 1e-4, 1e-5]
    dropouts = [0.0, 0.1]
    activations = ['relu', 'elu', 'leaky_relu']



    n_tasks = 10  # Number of IHDP datasets to test (max is 100, we use a reduced set to do hyperparamenter tuning for computational reasons). Also, note that we only use 1000 training patients for the ACIC dataset, to speed up the hyperparameter search

    n_jobs = 25  # Number of parallel jobs to run during training
    train_flag = True  # Whether to train or not
    resume_training = True  # If True, it will resume training from the last saved file, in case there was an error (i.e., if a results file already exists, it skips the training)

    if DEBUG:
        n_jobs=1
        n_tasks=1
        VERBOSE=1
        EPOCHS=10

    if train_flag:
        os.makedirs('../search_experiment_mlp', exist_ok=True)
        for dataset in datasets:
            for h_index, hidden_dim in enumerate(hidden_dims):
                for dropout in dropouts:
                    for lr in lrs:
                        for activation in activations:

                            file_name = os.path.join('../search_experiment_mlp', f"results_{dataset}_{h_index}_{dropout}_{lr}_{activation}.pkl")
                            if os.path.exists(file_name) and resume_training:
                                print(f"File {file_name} already exists. Skipping...")
                            else:
                                params = {'dataset': dataset, 'hidden_dims': hidden_dim, 'dropout': dropout, 'lr': lr, 'activation': activation,}
                                res = run_experiment(params, plot_flag=False, delete_flag=True, n_tasks=n_tasks, n_jobs=n_jobs)
                                with open(file_name, 'wb') as f:
                                    pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Show the results
    for dataset in datasets:
        results = []
        for h_index, hidden_dim in enumerate(hidden_dims):
            for dropout in dropouts:
                for lr in lrs:
                    for activation in activations:
                        file_name = os.path.join('../search_experiment_mlp', f"results_{dataset}_{h_index}_{dropout}_{lr}_{activation}.pkl")
                        if os.path.exists(file_name):
                            with open(file_name, 'rb') as f:
                                res = pickle.load(f)
                            results.append(res)

        print(f"Results for dataset {dataset}")

        def find_best_model(results, model_name):
            br = np.inf  # Best result
            bm = None  # Best model
            for r in results:
                if r[f'pehe_{model_name}'][0] < br:
                    br = r[f'pehe_{model_name}'][0]
                    bm = r
            print(f"Best {model_name} result: {br} with ATE {bm[f'ate_{model_name}']} and PEHE {bm[f'pehe_{model_name}']} with model: hidden_dims={bm['hidden_dims']}, dropout={bm['dropout']}, lr={bm['lr']}, activation={bm['activation']}")

            # Now, proceed to find all the models that are not significantly different from the best model
            _, data_test = load_data(dataset, 1)
            n = data_test.shape[0]  # Number of patients in the test split (needed for the t-test)
            threshold = 0.1  # Threshold for the t-test
            similar_models = []
            for r in results:
                _, pvalue = ttest_ind_from_stats(mean1=br, std1=bm[f'pehe_{model_name}'][1], nobs1=n, mean2=r[f'pehe_{model_name}'][0], std2=r[f'pehe_{model_name}'][1], nobs2=n, equal_var=False, alternative='less')
                if pvalue > threshold:
                    r['pvalue'] = pvalue
                    similar_models.append(r)
            # Sort sm by ascending complexity for printing in a more readable way
            # similar_models = sorted(similar_models, key=lambda x: x['training_time'])
            print(f"There are {len(similar_models)} similar models to the best {model_name} model:")
            for sm in similar_models:
                print(f"PEHE: {sm[f'pehe_{model_name}']} with pvalue {sm['pvalue']}"
                      # f" and training_time {sm['training_time']}"
                      f", ATE {sm[f'ate_{model_name}']} and PEHE {sm[f'pehe_{model_name}']}, hidden_dims={sm['hidden_dims']}, dropout={sm['dropout']}, lr={sm['lr']}, activation={sm['activation']}")
            print('\n')

        find_best_model(results, 'slearner')
        find_best_model(results, 'tlearner')
        find_best_model(results, 'tarnet')
        find_best_model(results, 'dragonnet')
        

        # Plot PEHE vs test_loss
        plt.scatter([r['test_loss_slearner'][0] for r in results], [r['pehe_slearner'][0] for r in results], label='S-learner')
        plt.scatter([r['test_loss_tlearner'][0] for r in results], [r['pehe_tlearner'][0] for r in results], label='T-learner')
        plt.scatter([r['test_loss_tarnet'][0] for r in results], [r['pehe_tarnet'][0] for r in results], label='TARNET')
        plt.scatter([r['test_loss_dragonnet'][0] for r in results], [r['pehe_dragonnet'][0] for r in results], label='DRAGONNET')
        plt.legend(loc='best')
        plt.xlabel('Test loss')
        plt.ylabel('PEHE')
        plt.title(f'PEHE vs Test loss for dataset {dataset}')
        # Make the x-axis logarithmic
        plt.xscale('log')
        plt.yscale('log')
        # Limit both axis to a reasonable upper range: 10 for y, 1000 for x
        plt.xlim(right=1000)
        plt.ylim(top=10)
        plt.savefig(f'pehe_vs_test_loss_mlp_{dataset}.png', bbox_inches='tight', dpi=200)
        plt.show()