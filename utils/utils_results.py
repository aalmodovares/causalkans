import os
import pandas as pd
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b

import scipy.stats as stats
import statsmodels.stats.multitest as multitest
from tabulate import tabulate




#### DATA utils ####
def load_ihdp(i, mode='A'):  # Based on Dragonnet's code
    if mode == 'A':
        data_train = pd.read_csv(os.path.join(os.getcwd(), 'datasets', 'IHDP_a', 'ihdp_npci_train_' + str(i) + '.csv'))
        data_test = pd.read_csv(os.path.join(os.getcwd(), 'datasets', 'IHDP_a', 'ihdp_npci_test_' + str(i) + '.csv'))
    else:
        data_train = pd.read_csv(os.path.join(os.getcwd(), 'datasets', 'IHDP_b', 'ihdp_npci_train_' + str(i) + '.csv'))
        data_test = pd.read_csv(os.path.join(os.getcwd(), 'datasets', 'IHDP_b', 'ihdp_npci_test_' + str(i) + '.csv'))

    return data_train, data_test


def load_acic_data(i, setting, test_split=0.2, n_samples=1000):
    data_path = os.path.join(os.getcwd(), 'datasets', 'ACIC')
    data_x = pd.read_csv(os.path.join(data_path, 'x.csv'))  # Covariates
    categorical_positions = [1, 20, 23]
    # Categorical positions are capital letters: replace them by their order number (i.e., A=0, B=1, etc.)
    for pos in categorical_positions:
        data_x.iloc[:, pos] = pd.Categorical(data_x.iloc[:, pos]).codes
    data_x = data_x.astype(np.float32)
    data_x = (data_x - data_x.mean()) / data_x.std()    # Standardize all the data

    data_path = os.path.join(data_path, setting)  # The rest of information is in the setting folder, covariates are common among setings
    list_of_files = os.listdir(data_path)
    #Load the data
    # columns "z","y0","y1","mu0","mu1". z is the treatment, y0 and y1 are the potential outcomes with noise, mu0 and mu1 are the potential outcomes without noise
    data_tymu = pd.read_csv(os.path.join(data_path, list_of_files[i - 1]))
    t = data_tymu['z']
    # define the outcome: y = y1 if t = 1, y = y0 if t = 0
    y_factual = data_tymu['y1'] * t + data_tymu['y0'] * (1 - t)
    y_cfactual = data_tymu['y1'] * (1 - t) + data_tymu['y0'] * t
    mu_0, mu_1 = data_tymu['mu0'], data_tymu['mu1']
    mus = np.asarray([mu_0, mu_1]).squeeze().T.astype(np.float32)

    dataset = np.hstack((np.array(t).reshape(-1, 1), np.array(y_factual).reshape(-1, 1), np.array(y_cfactual).reshape(-1, 1), mus, data_x.values))
    columns = ['treatment', 'y_factual', 'y_cfactual', 'mu0', 'mu1'] + ['x' + str(i+1) for i in range(data_x.shape[1])]
    data = pd.DataFrame(dataset, columns=columns)

    data_test = data.sample(frac=test_split, random_state=42)
    data_train = data.drop(data_test.index).reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    # Use only a subset of the data for training (to speed up simulations). Set to -1 to use all data
    if n_samples > 0:
        data_train = data_train.iloc[:n_samples]

    return data_train, data_test


def load_data(dataset, i, n_samples=1000):
    if dataset == 'IHDP_A':
        data_train, data_test = load_ihdp(i, mode='A')
    elif dataset == 'IHDP_B':
        data_train, data_test = load_ihdp(i, mode='B')
    elif dataset == 'ACIC_2':
        data_train, data_test = load_acic_data(i, setting='2', n_samples=n_samples)
    elif dataset == 'ACIC_7':
        data_train, data_test = load_acic_data(i, setting='7', n_samples=n_samples)
    elif dataset == 'ACIC_26':
        data_train, data_test = load_acic_data(i, setting='26', n_samples=n_samples)
    else:
        raise ValueError(f"Dataset {dataset} not recognized")
    return data_train, data_test

#### TRAINING utils ####

def get_width(model_name, input_size, hidden_dims, mult_kan=False):

    if model_name == 'slearner':
        width = [input_size + 1]
    else:
        width = [input_size]
    # Check the hidden dims
    if isinstance(hidden_dims, int) and hidden_dims > 0:
        width.append(hidden_dims)
    elif isinstance(hidden_dims, list):
        width.extend(hidden_dims)  # In case it is a list
    if model_name == 'slearner':
        width.append(1)  # For the output
    elif model_name == 'tarnet':
        width.append(2)  # For the output
    elif model_name == 'tlearner':
        width.append(2)  # For the output
    else:
        width.append(3)  # For the output
    if mult_kan:
        new_width = []
        for i in range(len(width)):
            if i == 0 or i == len(width) - 1:
                new_width.append(width[i])
            else:
                new_width.append([width[i], width[i]])
        return new_width
    else:
        return width

def get_dims_mlp(model_name, input_size, hidden_dims):

    dims = hidden_dims.copy()
    if model_name == 'slearner':
        dims = [input_size + 1] + dims + [1]
    elif model_name=='tlearner':
        if isinstance(dims[0], int):
            dims = [input_size] + dims + [1]

        else:
            dims[0] = [input_size] + dims[0] + [1]
            dims[1] = [input_size] + dims[1] + [1]
    else:
        dims[0] = [input_size] + dims[0]
        for i in range(1, len(dims)):
            dims[i] = [dims[0][-1]] + dims[i]
            dims[i] = dims[i] + [1]

    return dims


#### EVALUATION utils ####

def get_ihdp_baseline(i, mode='A'):
    data_train, data_test = load_ihdp(i, mode=mode)
    real_ite_test = np.squeeze(data_test['mu1'].values - data_test['mu0'].values)
    real_ate_test = np.mean(real_ite_test)
    x_train, y_train, t_train = data_train[[col for col in data_train.columns if 'x' in col]].values, data_train['y_factual'].values[:, None], data_train['treatment'].values[:, None]
    x_test, y_test, t_test = data_test[[col for col in data_test.columns if 'x' in col]].values, data_test['y_factual'].values[:, None], data_test['treatment'].values[:, None]

    if mode == 'A':
        beta_sol, k_sol = solve_ihdp_A(x_train, t_train, y_train)
        y_1_best = x_test @ beta_sol + k_sol
        y_0_best = x_test @ beta_sol

    else:
        beta_sol, ws_sol = solve_ihdp_B(x_train, t_train, y_train)
        y_1_best = x_test @ beta_sol - ws_sol
        y_0_best = np.exp((x_test + 0.5) @ beta_sol)

    ite_best = np.squeeze(y_1_best) - np.squeeze(y_0_best)
    ate_best = np.mean(ite_best)
    pehe_best = np.sqrt(np.mean((ite_best - real_ite_test)**2))

    return {'ate_real': real_ate_test, 'ite_real': real_ite_test, 'ate_best': ate_best, 'ite_best': ite_best, 'pehe_best': pehe_best}


def solve_ihdp_A(x, t, y):
    beta_sol = np.linalg.inv(x.T @ x - x.T @ t @ t.T @ x / (t.T @ t)) @ (x.T @ y - x.T @ t @ y.T @ t / (t.T @ t))
    k_sol = t.T @ (y - x @ beta_sol) / (t.T @ t)
    return beta_sol, k_sol


def objective_function_beta_ihdp_B(vector, x, t, y):
    beta = vector[:-1].reshape((x.shape[1], 1))
    ws = vector[-1] * np.ones((x.shape[0], 1))
    D_1 = np.diag(np.squeeze(t))
    D_0 = np.diag(np.squeeze(1 - t))
    model = D_0 @ (np.exp((x + 0.5) @ beta)) + D_1 @ (x @ beta - ws)
    obj_v = y - model
    fval = np.squeeze(obj_v.T @ obj_v)
    return fval


def solve_ihdp_B(x, t, y):
    sol = fmin_l_bfgs_b(func=objective_function_beta_ihdp_B, x0=np.zeros(x.shape[1] + 1), args=(x, t, y), approx_grad=True)  # Use a quasi-Newton method for optimization
    beta_sol = sol[0][:-1].reshape((x.shape[1], 1))
    ws_sol = sol[0][-1]
    return beta_sol, ws_sol


def generate_ihdp(i, mode='A'):  # Based on IHDP code, to have ground truth

    # Load i's model covariates
    data_train = pd.read_csv(os.path.join(os.getcwd(), 'datasets', 'IHDP_a', 'ihdp_npci_train_' + str(i) + '.csv'))
    data_test = pd.read_csv(os.path.join(os.getcwd(), 'datasets', 'IHDP_a', 'ihdp_npci_test_' + str(i) + '.csv'))

    x_train = data_train[[col for col in data_train.columns if 'x' in col]].values
    t_train = data_train['treatment'].values[:, None]
    x_test = data_test[[col for col in data_test.columns if 'x' in col]].values
    t_test = data_test['treatment'].values[:, None]

    # Now, generate the ground truth
    if mode == 'A':  # Linear model
        beta_A_probs = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
        beta_A_ground_truth = np.random.choice(np.arange(5), p=beta_A_probs, size=(x_train.shape[1], 1))
        mu_0_train = x_train @ beta_A_ground_truth
        mu_1_train = mu_0_train + 4.0
        y_0_train = mu_0_train + np.random.normal(0, 1, size=mu_0_train.shape)
        y_1_train = mu_1_train + np.random.normal(0, 1, size=mu_1_train.shape)
        y_train = t_train * y_1_train + (1 - t_train) * y_0_train
        mu_0_test = x_test @ beta_A_ground_truth
        mu_1_test = mu_0_test + 4.0
        y_0_test = mu_0_test + np.random.normal(0, 1, size=mu_0_test.shape)
        y_1_test = mu_1_test + np.random.normal(0, 1, size=mu_1_test.shape)
        y_test = t_test * y_1_test + (1 - t_test) * y_0_test
        beta_sol, k_sol = solve_ihdp_A(x_train, t_train, y_train)  # Solve the linear model
        ground_truth = {'beta_A': beta_A_ground_truth}
        best_estimator = {'beta': beta_sol, 'k': k_sol}
        y_1_best = x_test @ beta_sol + k_sol
        y_0_best = x_test @ beta_sol
    else:  # Non-linear model
        beta_B_probs = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
        beta_B_ground_truth = np.random.choice(np.arange(5) / 10, p=beta_B_probs, size=(x_train.shape[1], 1))  # Note that beta coefficients are smaller!
        mu_0_train = np.exp((x_train + 0.5) @ beta_B_ground_truth)
        mu_1_train = x_train @ beta_B_ground_truth
        ws = np.mean(mu_1_train) - np.mean(mu_0_train) - 4.0
        mu_1_train = mu_1_train - ws
        y_0_train = mu_0_train + np.random.normal(0, 1, size=mu_0_train.shape)
        y_1_train = mu_1_train + np.random.normal(0, 1, size=mu_1_train.shape)
        y_train = t_train * y_1_train + (1 - t_train) * y_0_train
        mu_0_test = np.exp((x_test + 0.5) @ beta_B_ground_truth)
        mu_1_test = x_test @ beta_B_ground_truth - ws
        y_0_test = mu_0_test + np.random.normal(0, 1, size=mu_0_test.shape)
        y_1_test = mu_1_test + np.random.normal(0, 1, size=mu_1_test.shape)
        y_test = t_test * y_1_test + (1 - t_test) * y_0_test

        beta_sol, ws_sol = solve_ihdp_B(x_train, t_train, y_train)

        ground_truth = {'beta_B': beta_B_ground_truth, 'ws': ws}
        best_estimator = {'beta': beta_sol, 'ws': ws_sol}
        y_1_best = x_test @ beta_sol - ws_sol
        y_0_best = np.exp((x_test + 0.5) @ beta_sol)

    y_f_train = t_train * y_1_train + (1 - t_train) * y_0_train
    y_f_test = t_test * y_1_test + (1 - t_test) * y_0_test

    return {'x_train': x_train, 'y_f_train': y_f_train, 't_train': t_train,
            'x_test': x_test, 'y_f_test': y_f_test, 't_test': t_test,
            'mu_0_train': mu_0_train, 'mu_1_train': mu_1_train, 'y_0_train': y_0_train, 'y_1_train': y_1_train,
            'mu_0_test': mu_0_test, 'mu_1_test': mu_1_test, 'y_0_test': y_0_test, 'y_1_test': y_1_test,
            'ground_truth': ground_truth, 'best_estimator': best_estimator,
            'y_0_best': y_0_best, 'y_1_best': y_1_best}




############## statistics utils ##############
def friedman_test(all_data, comp_index, alpha, higher_is_better):
    """
    Perform the Friedman test on the provided data. Based on Demsar06.
    :param all_data: 2D numpy array of shape (n_methods, n_datasets) where each row is a method and each column is a dataset.
    :param comp_index: Method to set as baseline for post-hoc tests, as in Demsar06. Should be the best performing metric...
    :param alpha: significance level for the test.
    :return: Friedman test p_value, davenport p-value and pairwise to the best baseline post-hoc p-values.
    """
    # Check that comp_index gives the best performing method (double check just in case...)
    avg_performance = np.mean(all_data, axis=1)  # Average performance across datasets for each method
    if higher_is_better:
        assert comp_index == np.argmax(avg_performance), "comp_index must be the index of the best performing method."
    else:
        assert comp_index == np.argmin(avg_performance), "comp_index must be the index of the best performing method."
    # Manual implementation of the Friedman test--to compute post-hoc metrics later on
    n_methods, n_reps = all_data.shape
    ranking_matrix = np.zeros_like(all_data)

    for k in range(n_reps):
        # Rank the methods for each dataset/fold
        if higher_is_better:
            ranking_matrix[:, k] = stats.rankdata(-all_data[:, k], method='average')  # Average ranks for ties
        else:
            ranking_matrix[:, k] = stats.rankdata(all_data[:, k], method='average')  # Average ranks for ties

    # Calculate the Friedman test statistic
    average_rank = np.mean(ranking_matrix, axis=1)
    friedman_stat = (12 * n_reps/ (n_methods * (n_methods + 1))) * (np.sum(np.square(average_rank)) - (n_methods * (n_methods + 1) ** 2 / 4))  # Friedman test statistic
    friedman_p_value = stats.chi2.sf(friedman_stat, df=n_methods - 1)  # p-value for the Friedman test
    davenport_stat = friedman_stat * (n_reps - 1) / (n_reps * (n_methods - 1))  # Davenport's statistic
    davenport_p_value = stats.f.sf(davenport_stat, dfn=n_methods - 1, dfd=(n_methods - 1) * (n_methods) * (n_reps - 1))

    # If we reject, we can perform post-hoc tests here. # TODO: Unsure if this is OK, need to account for higher is better in the p-values!!
    z_stat = np.zeros(n_methods)
    for j in range(n_methods):
        z_stat[j] = (average_rank[comp_index] - average_rank[j]) / np.sqrt((n_methods * (n_methods + 1)) / (6 * n_reps))  # Z-statistic for post-hoc tests
    p_values_post_hoc = stats.norm.cdf(z_stat)
    _, p_values_adjusted_post_hoc, _, _ = multitest.multipletests(p_values_post_hoc, alpha=alpha, method='holm')  # Holm-Bonferroni correction
    return friedman_p_value, davenport_p_value, p_values_post_hoc, p_values_adjusted_post_hoc

def get_p_values_from_table_data(data, alpha=0.05, higher_is_better=True, output_latex=True, list_of_methods=None, list_of_metrics=None):
    """
    Function to get p-values from a table of data in a structured way, automatically comparing with the best method for each metric.
    :param data: Organized as a numpy array: methods_to_compare x metrics x datasets/folds. Note that all datasets/folds need to have the same ordering: we use paired tests!!
    :param alpha: float, significance level for the hypothesis test.
    :param higher_is_better: bool or list of bool, if True, higher values are better, otherwise lower values are better.
    :param output_latex: bool, if True, outputs the table in LaTeX format, to copy and paste into a LaTeX document.
    :param list_of_methods: List of method names, if None, uses the default names.
    :param list_of_metrics: List of metric names, if None, uses the default names.
    :return: Outputs a p-value table comparing each method to the specified comparison method.
    """

    assert isinstance(data, np.ndarray), "Data must be a numpy array."
    assert data.ndim == 3, "Data must be a 3D numpy array with shape (n_methods, n_metrics, n_reps)."

    n_methods, n_metrics, n_reps = data.shape
    average_results = np.mean(data, axis=2)  # Average over repetitions, we have an array of shape (n_methods, n_metrics)

    if list_of_methods is None:
        list_of_methods = [f'Method {i+1}' for i in range(data.shape[0])]
    if list_of_metrics is None:
        list_of_metrics = [f'Metric {i+1}' for i in range(data.shape[1])]

    if not isinstance(higher_is_better, bool):
        assert len(higher_is_better) == n_metrics, "If higher_is_better is a list, it must have the same length as the number of metrics."
    else:
        higher_is_better = [higher_is_better] * data.shape[1]  # If it's a single bool, replicate it for all metrics

    max_idxs = np.argmax(average_results, axis=0)
    min_idxs = np.argmin(average_results, axis=0)
    comp_index = [max_idxs[i] if higher_is_better[i] else min_idxs[i] for i in range(n_metrics)]

    for i in range(n_metrics):

        # Print the data for complete reference
        print(f'\nData for metric {list_of_metrics[i]}, where higher_is_better is {higher_is_better[i]}:')
        # for j in range(n_methods):
            # print(f'{list_of_methods[j]}: {data[j, i, :]}:.3f / avg: {np.mean(data[j, i, :]):.3f}')
        table_metrics = ['Average metric'] + [f"{np.mean(data[j, i, :]):.3f}" for j in range(n_methods)]
        # First method: use paired Wilcoxon signed-rank test to obtain p-values, and correct them using Holm-Bonferroni method. This is done per-metric, so if we have many metrics, we will have many p-values.

        baseline_values = data[comp_index[i], i, :]  # Baseline values for the metric
        p_values = []
        for j in range(n_methods):
            test_values = data[j, i, :]  # Test values for the metric
            if comp_index[i] == j: # If we are comparing the baseline method with itself, we skip this comparison, as the Wilcoxon test will throw an error
                p_values.append(1.0)  # No difference, p-value is 1
                continue
            if higher_is_better[i]:
                # If higher is better, we want to test if the test values are significantly lower than the baseline values (i.e., significantly worse)
                _, p_value = stats.wilcoxon(test_values, baseline_values, alternative='less')
            else:
                # If lower is better, we want to test if the test values are significantly higher than the baseline values (i.e., significantly worse)
                _, p_value = stats.wilcoxon(test_values, baseline_values, alternative='greater')
            p_values.append(p_value)
        # Apply Holm-Bonferroni correction
        p_values = np.array(p_values)
        _, corrected_p_vals, _, _ = multitest.multipletests(np.array(p_values), alpha=alpha, method='holm')
        # Prepare a table to store all data for this metric
        table_wilcoxon_corr = ['Paired Wilcoxon tests (corrected)']
        table_wilcoxon_unc = ['Paired Wilcoxon tests (uncorrected)']
        for j in range(n_methods):
            p_val_str = f"{corrected_p_vals[j]:.3f}" if corrected_p_vals[j] >= 1e-3 else "<1e-3"  # Format p-values
            if corrected_p_vals[j] >= alpha:
                p_val_str += '*'  # Mark best values
            if j == comp_index[i]:
                p_val_str += ' (baseline)'  # Mark the baseline method
            table_wilcoxon_corr.append(p_val_str)

            p_val_str = f"{p_values[j]:.3f}" if p_values[j] >= 1e-3 else "<1e-3"  # Format small p-values
            if p_values[j] >= alpha:
                p_val_str += '*'  # Mark best values
            if j == comp_index[i]:
                p_val_str += ' (baseline)'  # Mark the baseline method
            table_wilcoxon_unc.append(p_val_str)

        # Second method: the Friedman test, which is a non-parametric test for repeated measures done on all metrics at once. Blocks = methods, treatments = datasets / folds (we could also implement one on datasets * metrics, a general one, later on). We rely on Demsar06 for this implementation.
        friedman_p_value, davenport_p_value, p_values_post_hoc_unc, p_values_post_hoc_corr = friedman_test(data[:, i, :], comp_index[i], alpha, higher_is_better[i])

        # Prepare this for the table
        friedman_post_hoc_table_corr = ['Friedman post-hoc tests (Corrected)']
        friedman_post_hoc_table_unc = ['Friedman post-hoc tests (Uncorrected)']
        for j in range(n_methods):
            p_val_str = f"{p_values_post_hoc_corr[j]:.3f}" if p_values_post_hoc_corr[j] >= 1e-3 else "<1e-3"  # Format p-values
            if p_values_post_hoc_corr[j] >= alpha:
                p_val_str += '*'  # Mark best values
            if j == comp_index[i]:
                p_val_str += ' (baseline)'  # Mark the baseline method
            friedman_post_hoc_table_corr.append(p_val_str)

            p_val_str = f"{p_values_post_hoc_unc[j]:.3f}" if p_values_post_hoc_unc[j] >= 1e-3 else "<1e-3"  # Format small p-values
            if p_values_post_hoc_unc[j] >= alpha:
                p_val_str += '*'  # Mark best values
            if j == comp_index[i]:
                p_val_str += ' (baseline)'  # Mark the baseline method
            friedman_post_hoc_table_unc.append(p_val_str)
        if friedman_p_value < 1e-3:
            friedman_p_value_str = "<1e-3"  # Format small p-values
        else:
            friedman_p_value_str = f"{friedman_p_value:.3f}"
        if davenport_p_value < 1e-3:
            davenport_p_value_str = "<1e-3"  # Format small p-values
        else:
            davenport_p_value_str = f"{davenport_p_value:.3f}"
        print(f'Friedman p-value: {friedman_p_value_str}, Davenport p-value: {davenport_p_value_str} for metric {list_of_metrics[i]}')
        if n_reps <= 10 or n_methods <=5:
            print('Since the number of data points is small, the Friedman test may not be reliable. Consider using a larger dataset or a different test.')

        table_data = [table_metrics, table_wilcoxon_unc, table_wilcoxon_corr, friedman_post_hoc_table_unc, friedman_post_hoc_table_corr]

        if output_latex:
            print(tabulate(table_data, headers=[f'Metric {list_of_metrics[i]}'] + list_of_methods, tablefmt='latex'))
        else:
            print(tabulate(table_data, headers=[f'Metric {list_of_metrics[i]}'] + list_of_methods, tablefmt='grid'))

    # Finally, run a Friedman test on all metrics at once
    all_data = data.copy()
    # For all metrics where lower is better, we need to invert the data so that higher is better always
    for j in range(n_metrics):
        if not higher_is_better[j]:
            all_data[:, j, :] = -all_data[:, j, :]  # Invert the data for lower is better metrics
    # Now, reshape the data to have shape (n_methods, n_metrics * n_reps)
    all_data = all_data.reshape(n_methods, n_metrics * n_reps)
    avg_metrics = np.mean(all_data, axis=1)  # Average over repetitions and metrics
    best_method = np.argmax(avg_metrics)  # Best method across all metrics (remember, higher is better now!)
    friedman_p_value, davenport_p_value, p_values_post_hoc_unc, p_values_post_hoc_corr = friedman_test(all_data, best_method, alpha, higher_is_better=True)
    print(f'Friedman test on all metrics: p-value: {friedman_p_value:.4f}, Davenport p-value: {davenport_p_value:.4f}')
    # Prepare this for the table
    friedman_post_hoc_table_unc = ['Friedman post-hoc tests (all metrics, uncorrected)']
    friedman_post_hoc_table_corr = ['Friedman post-hoc tests (all metrics, corrected)']
    for j in range(n_methods):
        p_val_str = f"{p_values_post_hoc_corr[j]:.3f}" if p_values_post_hoc_corr[j] >= 1e-3 else "<1e-3"  # Format p-values
        if p_values_post_hoc_corr[j] >= alpha:
            p_val_str += '*'  # Mark best values
        if j == best_method:
            p_val_str += ' (baseline)'  # Mark the baseline method
        friedman_post_hoc_table_corr.append(p_val_str)

        p_val_str = f"{p_values_post_hoc_unc[j]:.3f}" if p_values_post_hoc_unc[j] >= 1e-3 else "<1e-3"  # Format small p-values
        if p_values_post_hoc_unc[j] >= alpha:
            p_val_str += '*'  # Mark best values
        if j == best_method:
            p_val_str += ' (baseline)'  # Mark the baseline method
        friedman_post_hoc_table_unc.append(p_val_str)
    table_data = [friedman_post_hoc_table_unc, friedman_post_hoc_table_corr]
    if output_latex:
        print(tabulate(table_data, headers=['All metrics'] + list_of_methods, tablefmt='latex'))
    else:
        print(tabulate(table_data, headers=['All metrics'] + list_of_methods, tablefmt='grid'))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ---------------------------
# Plotting utility
# ---------------------------
def _add_sig_bracket(ax, x1, x2, y, h, p, alpha, text_y_offset=0.01):
    """
    Draw a bracket between x1 and x2 at height y with height h.
    Annotate with '*' if p < alpha.
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c='k')
    if p < alpha:
        ax.text((x1 + x2) / 2.0, y + h + text_y_offset*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                "*", ha='center', va='bottom', fontsize=14)

# ---------------------------
# Main function
# ---------------------------
def plot_kan_mlp_boxplots(np_array,
                          methods_to_compare=None,
                          metrics=('PEHE', 'ATE MAE'),
                          dataset_name="",
                          alpha=0.05,
                          kan_color='tab:blue',
                          mlp_color='tab:orange',
                          plot_whis=1.5,
                          plot_showfliers=True,
                          transparency=0.9):
    """
    Create two boxplots (one per metric). X-axis: [S-Learner, T-Learner, TARNet, DragonNet].
    Hue: KAN (kan_color) vs MLP/NN (mlp_color). Add Holm–Bonferroni-corrected Wilcoxon brackets.

    Parameters
    ----------
    np_array : np.ndarray
        Shape (8, 2, N): methods x metrics x realizations. Methods in this order:
        ['S-KAN','T-KAN','TARKAN','DragonKAN','slearner','tlearner','tarnet','dragonnet'].
    methods_to_compare : list[str] or None
        If None, uses the default order above.
    metrics : tuple[str, str]
        Names for the two metrics (used as y-labels).
    dataset_name : str
        Figure title will be 'dataset=<dataset_name>'.
    alpha : float
        Significance level for Holm–Bonferroni-corrected Wilcoxon tests.
    kan_color : str
        Matplotlib color for KAN boxes.
    mlp_color : str
        Matplotlib color for MLP/NN boxes.
    plot_whis : float or tuple
        Whisker definition for Matplotlib boxplot. Example: 1.5 for Tukey; or (5, 95) for percentiles.
    plot_showfliers : bool
        Whether to render individual outlier points.

    Returns
    -------
    figs : list[matplotlib.figure.Figure]
        Two figures (one per metric).
    """
    if methods_to_compare is None:
        methods_to_compare = ['S-KAN','T-KAN','TARKAN','DragonKAN',
                              'slearner','tlearner','tarnet','dragonnet']
    assert isinstance(np_array, np.ndarray) and np_array.shape[0] == 8 and np_array.shape[1] == 2, \
        "np_array must have shape (8, 2, N)."

    base_learners = ['S-Learner', 'T-Learner', 'TARNet', 'DragonNet']
    pair_map = {
        'S-Learner':  ('S-KAN',    'slearner'),
        'T-Learner':  ('T-KAN',    'tlearner'),
        'TARNet':     ('TARKAN',   'tarnet'),
        'DragonNet':  ('DragonKAN','dragonnet'),
    }
    name_to_idx = {name: i for i, name in enumerate(methods_to_compare)}

    x_positions = np.arange(len(base_learners))
    offset = 0.18
    width = 0.30



    # Outlier (flier) appearance matching the group color
    kan_flierprops = dict(marker='o', markersize=3, markerfacecolor=kan_color,
                          markeredgecolor='none', alpha=transparency)
    mlp_flierprops = dict(marker='o', markersize=3, markerfacecolor=mlp_color,
                          markeredgecolor='none', alpha=transparency)

    figs = []
    for m_idx, metric_name in enumerate(metrics):
        kan_data, mlp_data = [], []
        for bl in base_learners:
            kan_name, mlp_name = pair_map[bl]
            kan_idx = name_to_idx[kan_name]
            mlp_idx = name_to_idx[mlp_name]
            kan_data.append(np_array[kan_idx, m_idx, :])
            mlp_data.append(np_array[mlp_idx, m_idx, :])
        #
        fig, ax = plt.subplots()

        # KAN boxes
        kan_boxes = ax.boxplot(
            kan_data,
            positions=x_positions + offset,
            widths=width,
            patch_artist=True,
            manage_ticks=False,
            labels=['']*len(base_learners),
            whis=plot_whis,
            showfliers=plot_showfliers,
            flierprops=kan_flierprops
        )
        for patch in kan_boxes['boxes']:
            patch.set_facecolor(kan_color)
            patch.set_alpha(transparency)

        # MLP/NN boxes
        mlp_boxes = ax.boxplot(
            mlp_data,
            positions=x_positions - offset,
            widths=width,
            patch_artist=True,
            manage_ticks=False,
            labels=base_learners,
            whis=plot_whis,
            showfliers=plot_showfliers,
            flierprops=mlp_flierprops
        )
        for patch in mlp_boxes['boxes']:
            patch.set_facecolor(mlp_color)
            patch.set_alpha(transparency)

        # Axes and labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(base_learners, rotation=0)
        ax.set_ylabel(metric_name.replace('ATE MAE', 'ATE err'))
        # fig.suptitle(f"dataset={dataset_name}")

        # Legend
        legend_handles = [Patch(facecolor=mlp_color, edgecolor='k', alpha=transparency, label='MLP'),
                          Patch(facecolor=kan_color, edgecolor='k', alpha=transparency, label='KAN'),
                          ]
        ax.legend(handles=legend_handles, frameon=False, loc='best')

        # Paired Wilcoxon tests (two-sided) and Holm–Bonferroni correction
        raw_p = []
        for i in range(len(base_learners)):
            try:
                _, p = stats.wilcoxon(kan_data[i], mlp_data[i], alternative='two-sided', zero_method='wilcox')
            except ValueError:
                p = 1.0  # all differences zero
            raw_p.append(p)
        _, p_corr, _, _ = multitest.multipletests(raw_p, alpha=alpha, method='holm')

        # Significance brackets
        y_max = np.max([np.max(d) for d in kan_data + mlp_data])
        y_min = np.min([np.min(d) for d in kan_data + mlp_data])
        y_range = (y_max - y_min) if y_max > y_min else 1.0

        # upper_whiskers = [whisker.get_ydata()[1] for whisker in mlp_boxes['whiskers'][1::2]]
        # # take the whisker endpoints instead of raw data maxima
        # upper_whiskers_kan = [w.get_ydata()[1] for w in kan_boxes['whiskers'][1::2]]
        # upper_whiskers_mlp = [w.get_ydata()[1] for w in mlp_boxes['whiskers'][1::2]]
        # upper_whiskers = np.maximum(upper_whiskers_kan, upper_whiskers_mlp)
        # base_height = upper_whiskers[i]  # per pair inside the loop
        # step = 0.02 * (np.max(upper_whiskers) - np.min(upper_whiskers) + 1e-9)  # smaller spacing

        # Fix the y-axis height reference before adding brackets
        ymin, ymax = ax.get_ylim()
        yspan = ymax - ymin

        y_brace = ymin + 0.90 * yspan  # 90% of original y-axis
        h_brace = 0.015 * yspan  # bracket height
        text_off = 0.005 * yspan


        # base_height = y_max + 0.05 * y_range
        # step = 0.06 * y_range

        for i in range(len(base_learners)):
            x1 = x_positions[i] - offset
            x2 = x_positions[i] +offset
            _add_sig_bracket(ax, x1, x2, y_brace, h=h_brace, p=p_corr[i], alpha=alpha, text_y_offset=text_off)

        ax.margins(x=0.05)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # ax.legend(handles=legend_handles, frameon=False, loc='best')

        ax.legend(handles=legend_handles,
                  frameon=False,
                  loc='center left',
                  bbox_to_anchor=(1.02, 0.5),
                  handlelength=1, handleheight=1,  # make it look square
                  handletextpad=0.5)

        fig.tight_layout(rect=[0, 0, 0.85, 0.95])
        figs.append(fig)

    return figs


def _draw_one_panel(ax, np_array, metric_name,
                    *, kan_color='tab:blue', mlp_color='tab:orange',
                    plot_whis=1.5, plot_showfliers=False, transparency=0.9,
                    alpha=0.05):
    """Draw a single metric panel on a provided Axes (no legend, no titles)."""
    base_learners = ['S-Learner', 'T-Learner', 'TARNet', 'DragonNet']
    methods_to_compare = ['S-KAN','T-KAN','TARKAN','DragonKAN',
                          'slearner','tlearner','tarnet','dragonnet']
    pair_map = {
        'S-Learner':  ('S-KAN',    'slearner'),
        'T-Learner':  ('T-KAN',    'tlearner'),
        'TARNet':     ('TARKAN',   'tarnet'),
        'DragonNet':  ('DragonKAN','dragonnet'),
    }
    name_to_idx = {name: i for i, name in enumerate(methods_to_compare)}

    # Determine metric index from the provided name
    metric_name_norm = metric_name.replace('ATE MAE', 'ATE err')
    metric_idx = 0 if metric_name_norm.upper().startswith('PEHE') else 1

    x_positions = np.arange(len(base_learners))
    offset, width = 0.18, 0.30

    # Data split per group
    kan_data, mlp_data = [], []
    for bl in base_learners:
        kan_name, mlp_name = pair_map[bl]
        kan_idx = name_to_idx[kan_name]
        mlp_idx = name_to_idx[mlp_name]
        kan_data.append(np_array[kan_idx, metric_idx, :])
        mlp_data.append(np_array[mlp_idx, metric_idx, :])

    # Flier styling
    kan_flierprops = dict(marker='o', markersize=3, markerfacecolor=kan_color,
                          markeredgecolor='none', alpha=transparency)
    mlp_flierprops = dict(marker='o', markersize=3, markerfacecolor=mlp_color,
                          markeredgecolor='none', alpha=transparency)

    # MLP (left)
    mlp_boxes = ax.boxplot(
        mlp_data,
        positions=x_positions - offset,
        widths=width,
        patch_artist=True,
        manage_ticks=False,
        labels=['']*len(base_learners),
        whis=plot_whis,
        showfliers=plot_showfliers,
        flierprops=mlp_flierprops
    )
    for patch in mlp_boxes['boxes']:
        patch.set_facecolor(mlp_color)
        patch.set_alpha(transparency)

    # KAN (right)
    kan_boxes = ax.boxplot(
        kan_data,
        positions=x_positions + offset,
        widths=width,
        patch_artist=True,
        manage_ticks=False,
        labels=base_learners,
        whis=plot_whis,
        showfliers=plot_showfliers,
        flierprops=kan_flierprops
    )
    for patch in kan_boxes['boxes']:
        patch.set_facecolor(kan_color)
        patch.set_alpha(transparency)

    # Common axes formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(base_learners, rotation=0)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Wilcoxon + Holm
    raw_p = []
    for i in range(len(base_learners)):
        try:
            _, p = stats.wilcoxon(kan_data[i], mlp_data[i], alternative='two-sided', zero_method='wilcox')
        except ValueError:
            p = 1.0
        raw_p.append(p)
    _, p_corr, _, _ = multitest.multipletests(raw_p, alpha=alpha, method='holm')

    # Brackets at 90% of axis
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin if ymax > ymin else 1.0
    y_brace = ymin + 0.90 * yspan
    h_brace  = 0.015 * yspan
    text_off = 0.005 * yspan
    ax.set_ylim(ymin, ymax)  # lock

    def _add_brace(x1, x2, p):
        ax.plot([x1, x1, x2, x2], [y_brace, y_brace + h_brace, y_brace + h_brace, y_brace], lw=1.2, c='k')
        if p < alpha:
            ax.text((x1 + x2)/2.0, y_brace + h_brace + text_off, '*', ha='center', va='bottom')

    for i in range(len(base_learners)):
        _add_brace(x_positions[i] - offset, x_positions[i] + offset, p_corr[i])

    ax.margins(x=0.05)

def make_grid_2x5(data_map,
                  *, kan_color='tab:blue', mlp_color='tab:orange',
                  alpha=0.1, plot_whis=1.5, plot_showfliers=False,
                  transparency=0.9):
    """
    Build one figure with 2x5 panels:
      row 0: PEHE
      row 1: ATE err
      columns: datasets in the order of data_map keys.
    Shows y-labels only on the left column, hides top-row x-ticks,
    puts dataset names below bottom row, and adds one shared legend above.
    """
    datasets = list(data_map.keys())
    assert len(datasets) == 5, "Expect data_map with exactly 5 datasets."

    fig, axes = plt.subplots(2, 5, figsize=(11,3.5), sharex=False, sharey=False)

    for col, ds in enumerate(datasets):
        arr = data_map[ds]

        # Top row: PEHE
        ax0 = axes[0, col]
        _draw_one_panel(ax0, arr, 'PEHE',
                        kan_color=kan_color, mlp_color=mlp_color,
                        plot_whis=plot_whis, plot_showfliers=plot_showfliers,
                        transparency=transparency, alpha=alpha)
        # Hide top row x tick labels
        ax0.set_xticklabels([])

        # Bottom row: ATE err
        ax1 = axes[1, col]
        _draw_one_panel(ax1, arr, 'ATE err',
                        kan_color=kan_color, mlp_color=mlp_color,
                        plot_whis=plot_whis, plot_showfliers=plot_showfliers,
                        transparency=transparency, alpha=alpha)

        # Dataset name below bottom row
        ax1.set_xlabel(ds.replace('_', ' '), labelpad=6)

        # Y-axis labels only on left column
        if col == 0:
            ax0.set_ylabel('PEHE')
            ax1.set_ylabel('ATE err')
        else:
            ax0.set_ylabel('')
            ax1.set_ylabel('')

    # Shared legend above
    legend_handles = [
        Patch(facecolor=mlp_color, edgecolor='k', alpha=transparency, label='MLP'),
        Patch(facecolor=kan_color, edgecolor='k', alpha=transparency, label='KAN'),
    ]
    fig.legend(legend_handles, ['MLP', 'KAN'],
               loc='upper center', ncol=2, frameon=False,
               handlelength=1, handleheight=1, handletextpad=0.5, columnspacing=1.0)

    # Tight layout with room for the legend
    fig.tight_layout(rect=[0, 0, 1, 0.92], w_pad=1.0, h_pad=0.8)
    return fig, axes

