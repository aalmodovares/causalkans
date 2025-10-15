import os
import numpy as np
import pandas as pd
import sympy  # Not used yet, see https://github.com/knottwill/CoxKAN/blob/main/reprod/results.ipynb
from sympy.printing.latex import latex
import matplotlib.patches as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from kan import KAN, ex_round
from tqdm import tqdm
from joblib import Parallel, delayed


class kan_net(object):
    def __init__(self, model_name, dims, grid=10, k=3, seed=10, sparse_init=True, try_gpu=True, real_ite_train=None, real_ite_test=None, model_id=None, save_folder='checkpoints', **kwargs):

        self.device = torch.device('cuda' if torch.cuda.is_available() and try_gpu else 'cpu')
        self.model_name = model_name

        self.dims = dims  # List of input + hidden(s) + output dimensions
        self.grid = grid
        self.k = k
        self.seed = seed
        self.sparse_init = sparse_init

        self.real_ite_train = real_ite_train
        self.real_ite_test = real_ite_test

        # Assumption: the outputs are of dimension 1 (scalar y, treatment as input, this is an S-learner),
        # 2 (scalar y and y_cf for a single treatment) or of dimension 3 if we add the propensity score estimation, P(t=1|x). The inputs are always the covariates...
        if self.dims[-1] == 1:
            assert self.model_name == 'slearner'
        elif self.dims[-1] == 2:
            assert self.model_name == 'tarnet' or self.model_name == 'tlearner'
        elif self.dims[-1] == 3:
            assert self.model_name == 'dragonnet'
        else:
            raise ValueError('The output dimension must be 1, 2 or 3')

        self.seed_all(self.seed)

        self.loss_fn_y = nn.MSELoss()
        self.loss_fn_ps = nn.BCEWithLogitsLoss()

        # Create a random ID for the model to avoid overwriting
        self.model_id = model_id if model_id is not None else np.random.randint(0, 1000000)
        self.ckpt_path = os.path.join(os.getcwd(), save_folder, f'kan_{str(self.model_id)}') + os.sep
        self.img_folder = os.path.join(self.ckpt_path, 'video') + os.sep
        self.plot_folder = os.path.join(self.ckpt_path, 'plots') + os.sep
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.img_folder, exist_ok=True)
        os.makedirs(self.plot_folder, exist_ok=True)
        if self.model_name == 'tlearner':
            dims_tlearner = self.dims
            dims_tlearner[-1] = 1  # T-learner outputs are of dimension 1 (scalar y)
            self.model = t_learner([KAN(width=dims_tlearner, grid=self.grid, k=self.k,
                                        device=self.device, sparse_init=self.sparse_init,
                                        ckpt_path=self.ckpt_path, seed=seed+i, **kwargs) for i in range(2)])
        else:
            self.model = KAN(width=self.dims, grid=self.grid, k=self.k, device=self.device, sparse_init=self.sparse_init, ckpt_path=self.ckpt_path, seed=seed, **kwargs)

        self.dataset = None  # Will be set in the fit method

    def get_loss(self, y_pred, y_true, t_true=None):
        if self.model_name == 'slearner':
            loss = self.loss_fn_y(y_pred, y_true)  # Simple MSE loss
        else:
            idx_t_0 = torch.squeeze(t_true == 0)
            idx_t_1 = torch.squeeze(t_true == 1)
            y0_pred = y_pred[idx_t_0, 0]
            y1_pred = y_pred[idx_t_1, 1]
            y0_true = torch.squeeze(y_true[idx_t_0])
            y1_true = torch.squeeze(y_true[idx_t_1])
            loss = self.loss_fn_y(y0_pred, y0_true) + self.loss_fn_y(y1_pred, y1_true)  # TARNET loss
            if self.model_name == 'dragonnet':
                ps_pred = torch.squeeze(y_pred[:, 2])
                ps_true = torch.squeeze(t_true)
                loss += self.loss_fn_ps(ps_pred, ps_true.float())
        return loss

    def seed_all(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_metrics(self, y_pred, y_true, t_true, real_ite=None):
        metrics = {}
        y_pred = y_pred.detach().cpu().numpy()
        y_true = np.squeeze(y_true.detach().cpu().numpy())
        t_true = np.squeeze(t_true.detach().cpu().numpy())
        if self.model_name == 'slearner':
            metrics['mae'] = np.mean(np.abs(y_pred - y_true))
            metrics['mse'] = np.mean((y_pred - y_true)**2)
        else:
            idx_t_0 = np.squeeze(t_true == 0)
            idx_t_1 = np.squeeze(t_true == 1)
            y0_pred = y_pred[idx_t_0, 0]
            y1_pred = y_pred[idx_t_1, 1]
            y0_true = np.squeeze(y_true[idx_t_0])
            y1_true = np.squeeze(y_true[idx_t_1])
            y_f = y_pred[:, 0] * (1 - t_true) + y_pred[:, 1] * t_true
            metrics['mae_t0'] = np.mean(np.abs(y0_pred - y0_true))
            metrics['mae_t1'] = np.mean(np.abs(y1_pred - y1_true))
            metrics['mse_t0'] = np.mean((y0_pred - y0_true)**2)
            metrics['mse_t1'] = np.mean((y1_pred - y1_true)**2)
            metrics['mae'] = np.mean(np.abs(y_f - y_true))
            metrics['mse'] = np.mean((y_f - y_true)**2)
            t_pred = np.argmax(y_pred[:, 0:2], axis=1)
            metrics['t_acc'] = accuracy_score(t_true, t_pred)
            metrics['t_roc_auc'] = roc_auc_score(t_true, y_pred[:, 1])
            metrics['t_f1'] = f1_score(t_true, t_pred)
            if self.model_name == 'dragonnet':
                ps_pred = 1. / (1. + np.exp(-y_pred[:, 2]))
                ps_true = t_true
                metrics['ps_roc_auc'] = roc_auc_score(ps_true, ps_pred)
                metrics['ps_f1'] = f1_score(ps_true, ps_pred > 0.5)
                metrics['ps_acc'] = accuracy_score(ps_true, ps_pred > 0.5)
            if real_ite is not None:
                ite_pred = y_pred[:, 1] - y_pred[:, 0]
                metrics['ate'] = np.mean(ite_pred)
                metrics['pehe'] = np.sqrt(np.mean((ite_pred - np.squeeze(real_ite))**2))
        return metrics


    def custom_fit(self, dataset, steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0.,
            lamb_coefdiff=0., update_grid=True, grid_update_num=10, lr=1., start_grid_update_step=-1,
            stop_grid_update_step=50, batch=-1,
            save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1,
            img_folder='./video', singularity_avoiding=False, y_th=1000., reg_metric='edge_forward_spline_n',
            display_metrics=None, early_stop=False, patience=20, verbose=1):

        if self.model_name == 'tlearner':
            if lamb > 0. and not self.model.model[0].save_act:
                print('setting lamb=0. If you want to set lamb > 0, set self.save_act=True')

            old_save_act, old_symbolic_enabled = self.model.model[0].disable_symbolic_in_fit(lamb)
            if lamb > 0. and not self.model.model[1].save_act:
                print('setting lamb=0. If you want to set lamb > 0, set self.save_act=True')
            old_save_act, old_symbolic_enabled = self.model.model[1].disable_symbolic_in_fit(lamb)
        else:
            if lamb > 0. and not self.model.save_act:
                print('setting lamb=0. If you want to set lamb > 0, set self.save_act=True')

            old_save_act, old_symbolic_enabled = self.model.disable_symbolic_in_fit(lamb)

        if verbose > 0:
            pbar = tqdm(range(steps), desc='description', ncols=100)
        else:
            pbar = range(steps)

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        optimizer = torch.optim.Adam(self.model.get_params(), lr=lr)

        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
        results['train_metrics'] = []
        results['test_metrics'] = []

        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
        else:
            batch_size = batch
        batch_size_test = dataset['test_input'].shape[0]

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        best_loss = np.inf
        patience_counter = 0

        for _ in pbar:

            if _ == steps - 1 and old_save_act:
                self.model.save_act = True

            if save_fig and _ % save_fig_freq == 0:
                save_act = self.model.save_act
                self.model.save_act = True

            n_batches_train = len(dataset['train_input']) // batch_size

            for ibt in range(n_batches_train):

                batch_start = ibt * batch_size
                batch_end = min((ibt + 1) * batch_size, len(dataset['train_input']))
                train_id = np.arange(batch_start, batch_end)

                if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid and _ >= start_grid_update_step:
                    if self.model_name == 'tlearner':
                        self.model.model[0].update_grid(dataset['train_input'][train_id])
                        self.model.model[1].update_grid(dataset['train_input'][train_id])
                    else:
                        self.model.update_grid(dataset['train_input'][train_id])

                pred_train = self.model.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
                train_loss = self.get_loss(pred_train, dataset['train_label'][train_id], dataset['train_treatment'][train_id])

                if self.model_name=='tlearner':
                    reg_ = torch.tensor(0.)
                    for i in range(len(self.model.model)):
                        if self.model.model[i].save_act:
                            if reg_metric == 'edge_backward':
                                self.model.model[i].attribute()
                            if reg_metric == 'node_backward':
                                self.model.model[i].node_attribute()
                            reg_ += self.model.model[i].get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)

                else:
                    if self.model.save_act:
                        if reg_metric == 'edge_backward':
                            self.model.attribute()
                        if reg_metric == 'node_backward':
                            self.model.node_attribute()
                        reg_ = self.model.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
                    else:
                        reg_ = torch.tensor(0.)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pred_test = self.model.forward(dataset['test_input'])
            test_loss = self.get_loss(pred_test, dataset['test_label'], dataset['test_treatment'])

            # For conveniency, we get train loss and reg on the last batch only

            results['train_loss'].append(train_loss.cpu().detach().numpy())
            results['test_loss'].append(test_loss.cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

            train_metrics = self.get_metrics(self.model.forward(dataset['train_input']), dataset['train_label'], dataset['train_treatment'], self.real_ite_train)
            test_metrics = self.get_metrics(pred_test, dataset['test_label'], dataset['test_treatment'], self.real_ite_test)
            results['train_metrics'].append(train_metrics)
            results['test_metrics'].append(test_metrics)

            if _ % log == 0 and verbose > 0:
                if display_metrics == None:
                    pbar.set_description("| train_loss: %.2e | test_loss: %.2e | reg: %.2e | " % (
                    torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(),
                    reg_.cpu().detach().numpy()))
                else:
                    string = ''
                    data = ()
                    for metric in display_metrics:
                        string += f' {metric}: %.2e |'
                        try:
                            results[metric]
                        except:
                            raise Exception(f'{metric} not recognized')
                        data += (results[metric][-1],)
                    pbar.set_description(string % data)

            if save_fig and _ % save_fig_freq == 0:
                if self.model_name=='tlearner':
                    for i in range(len(self.model.model)):
                        save_act = self.model.model[i].save_act
                        self.model.model[i].save_act = True
                        self.model.model[i].plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars,
                                                 title="Step {} - Model {}".format(_, i), beta=beta)
                        plt.savefig(img_folder + '/' + str(_) + '_model_' + str(i) + '.jpg', bbox_inches='tight', dpi=200)
                        plt.close()
                        self.model.model[i].save_act = save_act
                else:
                    self.model.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_),
                                    beta=beta)
                    plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                    plt.close()
                    self.model.save_act = save_act

            if early_stop:
                if results['test_loss'][-1] < best_loss:
                    best_loss = results['test_loss'][-1]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > patience:
                        print(f'Early stopping at step {_}')
                        break

        if self.model_name=='tlearner':
            for i in range(len(self.model.model)):
                self.model.model[i].log_history('fit')
                # revert back to original state
                self.model.model[i].symbolic_enabled = old_symbolic_enabled
        else:
            self.model.log_history('fit')
            # revert back to original state
            self.model.symbolic_enabled = old_symbolic_enabled
        return results

    def fit(self, x_train, y_train, t_train, x_test, y_test, t_test, early_stop=True, patience=30, prune=False, **kwargs):

        self.seed_all(self.seed)

        if isinstance(x_train, np.ndarray):
            self.x_train = torch.from_numpy(x_train).to(self.device).float()
            self.y_train = torch.from_numpy(y_train).to(self.device).float()
            self.t_train = torch.from_numpy(t_train).to(self.device).long()
            self.x_test = torch.from_numpy(x_test).to(self.device).float()
            self.y_test = torch.from_numpy(y_test).to(self.device).float()
            self.t_test = torch.from_numpy(t_test).to(self.device).long()
        else:
            self.x_train = torch.from_numpy(x_train.values).to(self.device).float()
            self.y_train = torch.from_numpy(y_train.values).to(self.device).float()
            self.t_train = torch.from_numpy(t_train.values).to(self.device).long()
            self.x_test = torch.from_numpy(x_test.values).to(self.device).float()
            self.y_test = torch.from_numpy(y_test.values).to(self.device).float()
            self.t_test = torch.from_numpy(t_test.values).to(self.device).long()

        if self.model_name == 'slearner':
            # The train is composed by covariates AND treatment!
            self.x_train = torch.cat([self.x_train, self.t_train.float()], dim=1)
            self.x_test = torch.cat([self.x_test, self.t_test.float()], dim=1)

        self.dataset = {'train_input': self.x_train, 'train_label': self.y_train, 'test_input': self.x_test, 'test_label': self.y_test, 'train_treatment': self.t_train, 'test_treatment': self.t_test}

        # Note that it is important to update the grid in the fit, as it allows adapting to the input range automatically (see the update_grid method in the KAN class). By default, set to True: do not change that!
        results = self.custom_fit(self.dataset, early_stop=early_stop, patience=patience, img_folder=self.img_folder, **kwargs)

        if prune:
            self.prune()

        return results

    def prune(self, **kwargs):
        try:
            if self.model_name == 'tlearner':
                for i in range(len(self.model.model)):
                    self.model.model[i] = self.model.model[i].prune(**kwargs)
            else:

                self.model = self.model.prune(**kwargs)
        except:
            print('Pruning failed')

    def predict(self, x, t):
        x = torch.from_numpy(x).to(self.device).float()
        t = torch.from_numpy(t).to(self.device).long()
        t_cf = 1 - t  # Counterfactual treatment

        if self.model_name == 'slearner':
            xf = torch.cat([x, t.float()], dim=1)  # The input is composed by covariates AND treatment (factual in this case)
            xcf = torch.cat([x, t_cf.float()], dim=1)  # The input is composed by covariates AND treatment (counterfactual in this case)
            x0 = torch.cat([x, torch.zeros_like(t).float()], dim=1)  # The input is composed by covariates AND treatment (t=0)
            x1 = torch.cat([x, torch.ones_like(t).float()], dim=1)  # The input is composed by covariates AND treatment (t=1)
            y_f = np.squeeze(self.model.forward(xf).detach().cpu().numpy())
            y_cf = np.squeeze(self.model.forward(xcf).detach().cpu().numpy())
            y_0 = np.squeeze(self.model.forward(x0).detach().cpu().numpy())
            y_1 = np.squeeze(self.model.forward(x1).detach().cpu().numpy())
            results = {'y_pred_f': y_f, 'y_pred_cf': y_cf, 'y_pred_0': y_0, 'y_pred_1': y_1}
        else:
            y_pred = np.squeeze(self.model.forward(x).detach().cpu().numpy())
            t = np.squeeze(t.detach().cpu().numpy())
            results = {'y_pred_0': np.squeeze(y_pred[:, 0]), 'y_pred_1': np.squeeze(y_pred[:, 1]),
                       'y_pred_f': np.squeeze(y_pred[:, 0] * (1 - t) + y_pred[:, 1] * t),
                       'y_pred_cf': np.squeeze(y_pred[:, 0] * t + y_pred[:, 1] * (1 - t))}

            if self.model_name == 'dragonnet':
                       results['ps_pred'] = np.squeeze(1. / (1. + np.exp(-y_pred[:, 2])))

        results['pred_best_treatment'] = (results['y_pred_1'] > results['y_pred_0']).astype(int)
        return results

    def interprete(self, lib='all', n_digit=3, show_res=False):
        # lib can be a string to use a predefined library or a list of strings to use a custom library
        if isinstance(lib, str):
            print(f'Using library {lib}')
            if lib == 'linear':
                lib = ['x']
            elif lib == 'polynomial':
                lib = ['x', 'x^2', 'x^3', 'x^4', 'x^5']
            elif lib == 'all':
                lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'tan', 'abs']  # Model with many functions
            else:
                raise ValueError(f'Library {lib} not recognized')
        # Do a forward pass using the training data before interpreting, otherwise, the activations may not be properly computed

        if self.model_name == 'tlearner':
            formula = []
            for i in range(len(self.model.model)):
                self.model.model[i].forward(self.dataset['train_input'])
                self.model.model[i].auto_symbolic(lib=lib)
                formula_i = self.model.model[i].symbolic_formula()[0]
                formula_i = [ex_round(f, n_digit) for f in formula_i]
                formula.append(formula_i[0])
        else:
            self.model(self.dataset['train_input'])
            self.model.auto_symbolic(lib=lib)
            formula = self.model.symbolic_formula()[0]
            formula = [ex_round(f, n_digit) for f in formula]

        if show_res:
            print(f"Formulas obtained: ")
            for f in formula:
                print(latex(f))
            if self.model_name == 'tarnet' or self.model_name == 'dragonnet' or self.model_name == 'tlearner':
                print(f"ITE: {latex(formula[1] - formula[0])}")

        return formula

    def plot(self, plot_flag, plot_folder=None, **kwargs):
        if plot_folder is None:
            self.model.plot(folder=self.plot_folder, **kwargs)
            plt.savefig(os.path.join(self.plot_folder, 'kan.png'), bbox_inches='tight', dpi=200)

        else:
            self.model.plot(folder=plot_folder, **kwargs)
            plt.savefig(os.path.join(plot_folder, 'kan.png'), bbox_inches='tight', dpi=200)
        if plot_flag:
            plt.show()

        plt.close()

class t_learner(torch.nn.Module):
    def __init__(self, model_list):
        super(t_learner, self).__init__()
        self.model = torch.nn.ModuleList(model_list)
        self.device = model_list[0].device

    def forward(self, x, **kwargs):
        # Forward pass through both networks
        y_pred_0 = self.model[0](x, **kwargs)
        y_pred_1 = self.model[1](x, **kwargs)
        return torch.cat([y_pred_0.view(-1, 1), y_pred_1.view(-1, 1)], dim=1)

    def prune(self, **kwargs):
        # Prune both models
        for i in range(len(self.model)):
            self.model[i] = self.model[i].prune(**kwargs)
        return self

    def get_params(self):
        params = []
        for i in range(len(self.model)):
            params += list(self.model[i].parameters())
        return params


