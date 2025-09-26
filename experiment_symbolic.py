import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import sympy
import os

from scipy.optimize import minimize as minimize
# from stochopy.optimize import minimize as sminimize  # Note that this is only needed if using stochastic optimization, e.g., PSO. If not, can be commented out
from sklearn.metrics import mean_absolute_error, r2_score
#
# from models import get_model, get_metrics
# from data import load_data


class symbolic_kan_regressor(object):  # Class implemented for symbolic regression with single-layer KAN objects
    def __init__(self, x_names=None, y_names=None):
        self.x_names = x_names
        self.y_names = y_names
        self.symbolic_functions = None
        self.epsilon = 1e-10 # Small value to prevent division by zero
        self.complex_functions = ['triangle', 'sqrt', 'inv_sqrt', 'exp', 'log', 'abs', 'sin', 'cos', 'tan', 'tanh', 'sgn', 'arccos', 'arctan', 'arctanh']  # Functions that require 4 parameters: a, b, c, d

    def get_formula(self, ex_round=4):
        assert self.symbolic_functions is not None, "You must fit the model before getting the formula"
        # Create a sympy expression for each output

        n_outputs = max([j for _, _, _, j in self.symbolic_functions]) + 1
        n_inputs = max([i for _, _, i, _ in self.symbolic_functions]) + 1

        assert n_outputs * n_inputs == len(self.symbolic_functions), "The number of symbolic functions does not match the number of inputs and outputs"

        formulas = []
        for j in range(n_outputs):
            expr = 0
            for fun_name, params, i, j2 in self.symbolic_functions:
                if j2 == j:
                    x = sympy.symbols(self.x_names[i] if self.x_names is not None else f'x{i+1}')
                    if fun_name == 'polynomial':
                        expr += sum([p * x**k for k, p in enumerate(params)])
                    elif fun_name == 'inv_polynomial':
                        expr += sum([p * (1 / (x + self.epsilon))**k for k, p in enumerate(params)])
                    elif fun_name in self.complex_functions:
                        a, b, c, d = params[0], params[1], params[2], params[3]
                        if fun_name == 'triangle':
                            # Parameters are as follows: a (first slope), b (y offset of first slope), c (second slope), d (x offset of second slope)
                            # Compute the x value where the triangle changes slope
                            x_th = (d - b) / (a - c + self.epsilon)  # Avoid division by zero
                            expr += sympy.Piecewise((a * x + b, x < x_th), (c * x + d, True))
                        elif fun_name == 'exp':
                            expr += c * sympy.exp(a * x + b) + d
                        elif fun_name == 'sqrt':
                            expr += c * sympy.sqrt(sympy.Abs(a * x + b) + self.epsilon) + d
                        elif fun_name == 'inv_sqrt':
                            expr += c / (sympy.sqrt(sympy.Abs(a * x + b) + self.epsilon)) + d
                        elif fun_name == 'log':
                            expr += c * sympy.log(sympy.Abs(a * x + b) + self.epsilon) + d
                        elif fun_name == 'abs':
                            expr += c * sympy.Abs(a * x + b) + d
                        elif fun_name == 'sin':
                            expr += c * sympy.sin(a * x + b) + d
                        elif fun_name == 'cos':
                            expr += c * sympy.cos(a * x + b) + d
                        elif fun_name == 'tan':
                            expr += c * sympy.tan(a * x + b) + d
                        elif fun_name == 'tanh':
                            expr += c * sympy.tanh(a * x + b) + d
                        elif fun_name == 'sgn':
                            expr += c * sympy.sign(a * x + b) + d
                        elif fun_name == 'arccos':
                            expr += c * sympy.acos(sympy.Max(sympy.Min(a * x + b, 1 - self.epsilon), -1 + self.epsilon)) + d
                        elif fun_name == 'arctan':
                            expr += c * sympy.atan(a * x + b) + d
                        elif fun_name == 'arctanh':
                            expr += c * sympy.atanh(sympy.Max(sympy.Min(a * x + b, 1 - self.epsilon), -1 + self.epsilon)) + d
                    else:
                        raise ValueError(f"Function {fun_name} not implemented")

            expr_round = expr
            for a in sympy.preorder_traversal(expr):
                if isinstance(a, sympy.Float):
                    expr_round = expr_round.subs(a, round(a, ex_round))
            formulas.append(expr_round)
        return formulas

    def predict_individual(self, x, fun_name, params):

        if fun_name == 'polynomial':
            p = np.polynomial.Polynomial(params)
            return p(x)
        elif fun_name == 'inv_polynomial':
            p = np.polynomial.Polynomial(params)
            return p(1 / (x + self.epsilon))  # Avoid division by zero
        elif fun_name in self.complex_functions:
            a, b, c, d = params[0], params[1], params[2], params[3]
            if fun_name == 'triangle':
                # Parameters are as follows: a (first slope), b (y offset of first slope), c (second slope), d (x offset of second slope)
                # Compute the x value where the triangle changes slope
                x_th = (d - b) / (a - c + self.epsilon)  # Avoid division by zero
                return np.where(x < x_th, a * x + b, c * x + d)
            elif fun_name == 'exp':
                return c * np.exp(a * x + b) + d
            elif fun_name == 'sqrt':
                return c * np.sqrt(np.abs(a * x + b) + self.epsilon) + d
            elif fun_name == 'inv_sqrt':
                return c / (np.sqrt(np.abs(a * x + b) + self.epsilon)) + d
            elif fun_name == 'log':
                return c * np.log(np.abs(a * x + b) + self.epsilon) + d
            elif fun_name == 'abs':
                return c * np.abs(a * x + b) + d
            elif fun_name == 'sin':
                return c * np.sin(a * x + b) + d
            elif fun_name == 'cos':
                return c * np.cos(a * x + b) + d
            elif fun_name == 'tan':
                return c * np.tan(a * x + b) + d
            elif fun_name == 'tanh':
                return c * np.tanh(a * x + b) + d
            elif fun_name == 'sgn':
                return c * np.sign(a * x + b) + d
            elif fun_name == 'arccos':
                return c * np.arccos(np.clip(a * x + b, -1 + self.epsilon, 1 - self.epsilon)) + d
            elif fun_name == 'arctan':
                return c * np.arctan(a * x + b) + d
            elif fun_name == 'arctanh':
                return c * np.arctanh(np.clip(a * x + b, -1 + self.epsilon, 1 - self.epsilon)) + d
        else:
            raise ValueError(f"Function {fun_name} not implemented")

    def predict(self, x):

        assert self.symbolic_functions is not None, "You must fit the model before predicting"
        y_pred = np.zeros((x.shape[0], len(self.y_names)))
        for fun_name, params, i, j in self.symbolic_functions:
            x_i = x[:, i]
            y_pred[:, j] += self.predict_individual(x_i, fun_name, params)
        return y_pred

    def get_metrics(self, y_true, y_pred):

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mae, r2

    def fit(self, kan_object, x_train, y_train, x_test, y_test, denorm_function=None, stochastic=False, r2_threshold=0.95, show_results=False, min_param_val=-100, max_param_val=100, save_dir=None):

        if stochastic:
            print("Warning: Stochastic optimization has NOT been tested extensively. Use at your own risk.")

        assert len(kan_object.width_in) == 2, "This class only supports single-layer KAN objects"
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)

        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)

        if denorm_function is None:
            denorm_function = lambda x, y: x # Identity function if no denorm function is provided

        # We must ensure that acts and spline_postacts are computed
        y_pred_test_original = kan_object.forward(torch.from_numpy(x_test).to(kan_object.device).float()).detach().cpu().numpy()
        y_pred_train_original = kan_object.forward(torch.from_numpy(x_train).to(kan_object.device).float()).detach().cpu().numpy()

        self.symbolic_functions = []

        if self.x_names is None:
            self.x_names = [f'x{i+1}' for i in range(kan_object.width_in[0])]
        if self.y_names is None:
            self.y_names = [f'y{j+1}' for j in range(kan_object.width_out[1])]

        l = 0  # Only one layer
        t_init = time.time()
        for i in range(kan_object.width_in[l]):  # i indexes the input features
            for j in range(kan_object.width_out[l + 1]): # j indexes the output features

                best_r = -np.inf
                best_fun = -np.inf

                t_iter = time.time()

                x = kan_object.acts[l][:, i].detach().cpu().numpy()
                y = kan_object.spline_postacts[l][:, j, i].detach().cpu().numpy()
                success = False

                def evaluate_params(params, fname):
                    y_pred = self.predict_individual(x, fname, params)
                    mae, r2 = self.get_metrics(denorm_function(y, j), denorm_function(y_pred, j))
                    if r2 >= r2_threshold:
                        print(f"Found {fname} of degree {degree} for input {i + 1}/{kan_object.width_in[l]} and output {j + 1}/{kan_object.width_out[l + 1]} with r2 {r2:.4f}. Total time: {time.time() - t_init:.2f}s, Iteration time: {time.time() - t_iter:.2f}s, Average time per output: {(time.time() - t_init) / ((i * kan_object.width_out[l + 1]) + j + 1):.2f}s")
                        self.symbolic_functions.append((fname, params, i, j))
                        return True, mae, r2
                    else:
                        return False, mae, r2

                if len(np.unique(x)) ==1:
                    params = np.polynomial.Polynomial.fit(x, y, 0).convert().coef
                    success, mae, r2 = evaluate_params(params, 'polynomial')
                    best_r = r2
                    best_fun = ('polynomial', params, i, j)
                else:
                    # Start from simple to more complex functions
                    for optimization_strategy in ['polynomial', 'inv_polynomial', 'complex']:
                        if success:
                            break

                        if optimization_strategy == 'polynomial':
                            opt_iter = range(5)
                        elif optimization_strategy == 'inv_polynomial':
                            opt_iter = range(1, 5)
                        else:
                            opt_iter = self.complex_functions

                        if optimization_strategy == 'polynomial':
                            fname = 'polynomial'
                            for degree in opt_iter:
                                params = np.polynomial.Polynomial.fit(x, y, degree).convert().coef
                                success, mae, r2 = evaluate_params(params, fname)
                                if r2 > best_r:
                                    best_r = r2
                                    best_fun = (fname, params, i, j)
                                if success:
                                    break
                        elif optimization_strategy == 'inv_polynomial':
                            fname = 'inv_polynomial'
                            for degree in opt_iter:
                                x_inv = 1 / (x + self.epsilon)  # Avoid division by zero
                                params = np.polynomial.Polynomial.fit(x_inv, y, degree).convert().coef
                                success, mae, r2 = evaluate_params(params, fname)
                                if r2 > best_r:
                                    best_r = r2
                                    best_fun = (fname, params, i, j)
                                if success:
                                    break
                        else:
                            for fname in opt_iter:
                                bounds_nm = [(None, None)] * 4
                                bounds_pso = [(min_param_val, max_param_val)] * 4

                                def error_function(params):
                                    y_pred = self.predict_individual(x, fname, params)

                                    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)): # Penalize invalid predictions
                                        return np.inf

                                    mae, r2 = self.get_metrics(denorm_function(y, j), denorm_function(y_pred, j))
                                    return mae

                                if stochastic:
                                    pop_size = 10
                                    # Seed the numpy random generator to have reproducible results
                                    np.random.seed(0)
                                    initial_guess = np.random.uniform(low=min_param_val, high=max_param_val, size=(pop_size, 4))  # PSO needs a population of initial guesses
                                    result = sminimize(error_function, x0=initial_guess, method='pso', bounds=bounds_pso, options={"maxiter": 10000, "popsize": pop_size, "seed": 0})
                                else:
                                    if fname == 'triangle': # Triangle is very sensitive to initial guess, so we first do a coarse grid search to find a good initial guess. If it does not work visually, try changing the grid_size and min/max_grid_val
                                        grid_size = 5
                                        min_grid_val = -1
                                        max_grid_val = 1
                                        a_vals = np.linspace(min_grid_val, max_grid_val, grid_size)
                                        b_vals = np.linspace(min_grid_val, max_grid_val, grid_size)
                                        c_vals = np.linspace(min_grid_val, max_grid_val, grid_size)
                                        d_vals = np.linspace(min_grid_val, max_grid_val, grid_size)
                                        best_mae = np.inf
                                        best_params_grid = None
                                        for a in a_vals:
                                            for b in b_vals:
                                                for c in c_vals:
                                                    for d in d_vals:
                                                        params_grid = np.array([a, b, c, d])
                                                        mae_grid = error_function(params_grid)
                                                        if mae_grid < best_mae:
                                                            best_mae = mae_grid
                                                            best_params_grid = params_grid
                                        initial_guess = best_params_grid
                                    else:
                                        initial_guess = np.array([1.0, 0.0, 1.0, 0.0])
                                    result = minimize(error_function, initial_guess, method='Nelder-Mead', bounds=bounds_nm, options={"maxiter": 100000, "maxfev": 100000})

                                params_opt = result.x

                                success, mae, r2 = evaluate_params(params_opt, fname)
                                if r2 > best_r:
                                    best_r = r2
                                    best_fun = (fname, params_opt, i, j)
                                if success:
                                    break
                if not success:
                    print(f"Could not find a good function for input {i + 1}/{kan_object.width_in[l]} and output {j + 1}/{kan_object.width_out[l + 1]}. Best r2 was {best_r:.4f} with function {best_fun[0]}. Total time: {time.time() - t_init:.2f}s, Iteration time: {time.time() - t_iter:.2f}s, Average time per output: {(time.time() - t_init) / ((i * kan_object.width_out[l + 1]) + j + 1):.2f}s")
                    self.symbolic_functions.append(best_fun)

                if show_results or save_dir is not None:
                    plt.scatter(x, denorm_function(y, j), label='KAN Spline', alpha=0.5)
                    x_lin = np.linspace(x.min(), x.max(), 100)
                    y_lin = self.predict_individual(x_lin, self.symbolic_functions[-1][0], self.symbolic_functions[-1][1])
                    plt.plot(x_lin, denorm_function(y_lin, j), 'r-', label='Symbolic Fit')
                    plt.xlabel(f"Activation of input {i + 1} (Feature: {self.x_names[i] if self.x_names is not None else i + 1})")
                    plt.ylabel(f"Spline post-activation of output {j + 1} (Target: {self.y_names[j] if self.y_names is not None else j + 1})")
                    plt.title(f"Symbolic Fit for input {self.x_names[i] if self.x_names is not None else i + 1} and output {self.y_names[j] if self.y_names is not None else j + 1}")
                    plt.legend()
                    if save_dir is not None:
                        plt.savefig(os.path.join(save_dir, f'symbolic_fit_input_{i+1}_output_{j+1}.png'), bbox_inches='tight', dpi=300)
                    if show_results:
                        plt.show()
                    plt.close()

        y_pred_symbolic_train = self.predict(x_train)
        y_pred_symbolic_test = self.predict(x_test)

        print("Symbolic regression results:")
        for i, target in enumerate(self.y_names):
            mae_train_original, r2_train_original = self.get_metrics(denorm_function(y_train[:, i], i), denorm_function(y_pred_train_original[:, i], i))
            mae_test_original, r2_test_original = self.get_metrics(denorm_function(y_test[:, i], i), denorm_function(y_pred_test_original[:, i], i))
            train_mae, train_r2 = self.get_metrics(denorm_function(y_train[:, i], i), denorm_function(y_pred_symbolic_train[:, i], i))
            test_mae, test_r2 = self.get_metrics(denorm_function(y_test[:, i], i), denorm_function(y_pred_symbolic_test[:, i], i))

            print(f"Target {target}:")
            print(f"  Train MAE: {train_mae:.4f} (Original: {mae_train_original:.4f}), Train R2: {train_r2:.4f} (Original: {r2_train_original:.4f})")
            print(f"  Test MAE: {test_mae:.4f} (Original: {mae_test_original:.4f}), Test R2: {test_r2:.4f} (Original: {r2_test_original:.4f})")

            if save_dir is not None:
                with open(os.path.join(save_dir, 'metrics.txt'), 'a') as f:
                    f.write(f"Target {target}:\n")
                    f.write(f"  Train MAE: {train_mae:.4f} (Original: {mae_train_original:.4f}), Train R2: {train_r2:.4f} (Original: {r2_train_original:.4f})\n")
                    f.write(f"  Test MAE: {test_mae:.4f} (Original: {mae_test_original:.4f}), Test R2: {test_r2:.4f} (Original: {r2_test_original:.4f})\n\n")
                    f.write("----------------------------------------\n")

            if show_results or save_dir is not None:
                plt.figure()
                plt.scatter(denorm_function(y_train[:, i], i), denorm_function(y_pred_symbolic_train[:, i], i), label='Train SYM', alpha=0.5)
                plt.scatter(denorm_function(y_test[:, i], i), denorm_function(y_pred_symbolic_test[:, i], i), label='Test SYM', alpha=0.5)
                plt.scatter(denorm_function(y_train[:, i], i), denorm_function(y_pred_train_original[:, i], i), label='Train Spline', alpha=0.5)
                plt.scatter(denorm_function(y_test[:, i], i), denorm_function(y_pred_test_original[:, i], i), label='Test Spline', alpha=0.5)
                plt.plot([denorm_function(y_test[:, i], i).min(), denorm_function(y_test[:, i], i).max()], [denorm_function(y_test[:, i], i).min(), denorm_function(y_test[:, i], i).max()], 'k--', label='Ideal')
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.title(f'Symbolic Regression Predictions for {target}')
                plt.legend()
                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, f'symbolic_regression_predictions_{target}.png'), bbox_inches='tight', dpi=300)
                if show_results:
                    plt.show()
                plt.close()

# if __name__ == '__main__':
#
#     x_train, x_test, y_train, y_test, denorm_values, target_names, metric_names = load_data('rbattery', max_values=None)
#     #cols_to_keep = ['Potential', 'Phase_1', 'Zmag_1', 'Zre_1', 'Zim_1']
#     #x_train = x_train[cols_to_keep]
#     #x_test = x_test[cols_to_keep]
#     feat_names = x_train.columns.tolist()
#     x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
#
#     model = get_model('kan_gam')
#     model.set_params(hidden_dim=0, k=5, grid=10, batch_size=-1, lamb=0.01, lamb_entropy=0.1, lr=0.001, mult_kan=False, seed=0, sparse_init=False, steps=10000, try_gpu=True)
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     metrics = get_metrics(y_test, y_pred, denorm_values=denorm_values, target_names=target_names)
#     print(metrics)
#
#     # Since we want the metrics and representation in the original scale, we create a denorm target function fot our symbolic regressor and pass it to have metrics and representations in the original scale
#     def denorm_target(y, target_index):
#         return y * denorm_values['y_std'].iloc[target_index] + denorm_values['y_mean'].iloc[target_index]
#
#     # Symbolic regression
#     save_dir = './results_symbolic'
#     os.makedirs(save_dir, exist_ok=True)
#     symb_obj = symbolic_kan_regressor(x_names=feat_names, y_names=target_names)
#     out = symb_obj.fit(model.model, x_train, y_train, x_test, y_test, denorm_function=denorm_target, stochastic=False, r2_threshold=0.95, show_results=True, save_dir=save_dir) # Important: stochastic has NOT been tested!!
#     formulas = symb_obj.get_formula()
#     for i, formula in enumerate(formulas):
#         print(f"Formula for {target_names[i]}: {formula}")
#     # Save the formulas to a text file
#     with open(os.path.join(save_dir, 'formulas.txt'), 'w') as f:
#         for i, formula in enumerate(formulas):
#             f.write(f"Formula for {target_names[i]}: {formula}\n")
#             f.write(f"Latex for {target_names[i]}: {sympy.latex(formula)}\n\n")
#


