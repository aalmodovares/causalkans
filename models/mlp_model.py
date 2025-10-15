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


class mlp_net(object):
    def __init__(self, model_name, dims, seed=10, try_gpu=True, real_ite_train=None, real_ite_test=None, model_id=None, save_folder='checkpoints', **kwargs):

        self.device = torch.device('cuda' if torch.cuda.is_available() and try_gpu else 'cpu')
        self.model_name = model_name

        self.dims = dims  # List of input + hidden(s) + output dimensions
        self.seed = seed

        self.real_ite_train = real_ite_train
        self.real_ite_test = real_ite_test

        # Assumption: the outputs are of dimension 1 (scalar y, treatment as input, this is an S-learner),
        # 2 (scalar y and y_cf for a single treatment) or of dimension 3 if we add the propensity score estimation, P(t=1|x). The inputs are always the covariates...
        # if self.dims[-1] == 1:
        #     assert self.model_name == 'slearner'
        # elif self.dims[-1] == 2:
        #     assert self.model_name == 'tarnet' or self.model_name == 'tlearner'
        # elif self.dims[-1] == 3:
        #     assert self.model_name == 'dragonnet'
        # else:
        #     raise ValueError('The output dimension must be 1, 2 or 3')

        self.seed_all(self.seed)

        self.loss_fn_y = nn.MSELoss()
        self.loss_fn_ps = nn.BCEWithLogitsLoss()

        # Create a random ID for the model to avoid overwriting
        self.model_id = model_id if model_id is not None else np.random.randint(0, 1000000)
        self.ckpt_path = os.path.join(os.getcwd(), save_folder, f'mlp_{str(self.model_id)}') + os.sep
        self.plot_folder = os.path.join(self.ckpt_path, 'plots') + os.sep
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.plot_folder, exist_ok=True)

        if self.model_name == 'slearner':
            assert isinstance(self.dims[0], int), 'the slearner is a single network'
            self.model = slearner(mlp(width=self.dims, device=self.device, ckpt_path=self.ckpt_path, seed=seed, **kwargs))
        elif self.model_name == 'tlearner':
            assert len(self.dims) == 2 or isinstance(self.dims[0], int), 'the tlearner is a list of two networks'
            if isinstance(self.dims[0], int):
                self.model = tlearner([mlp(width=self.dims, device=self.device, ckpt_path=self.ckpt_path, seed=seed+i, **kwargs) for i in range(2)])
            else:
                self.model = tlearner([mlp(width=self.dims[i], device=self.device, ckpt_path=self.ckpt_path, seed=seed+i, **kwargs) for i in range(2)])
        elif self.model_name == 'tarnet':
            assert len(self.dims)==2 or len(self.dims)==3, 'tarnet has 3 networks, insert the dim of each network'
            if len(self.dims)==3:
                self.model = tarnet([mlp(width=self.dims[0], device=self.device, ckpt_path=self.ckpt_path, seed=seed, **kwargs),
                                    mlp(width=self.dims[1], device=self.device, ckpt_path=self.ckpt_path, seed=seed+1, **kwargs),
                                    mlp(width=self.dims[2], device=self.device, ckpt_path=self.ckpt_path, seed=seed+2, **kwargs)])
            else:
                self.model = tarnet([mlp(width=self.dims[0], device=self.device, ckpt_path=self.ckpt_path, seed=seed, **kwargs),
                                    mlp(width=self.dims[1], device=self.device, ckpt_path=self.ckpt_path, seed=seed+1, **kwargs),
                                    mlp(width=self.dims[1], device=self.device, ckpt_path=self.ckpt_path, seed=seed+2, **kwargs)])
        elif self.model_name == 'dragonnet':
            assert len(self.dims)==4 or len(self.dims)==3, 'dragonnet has 4 networks, insert the dim of each network'
            if len(self.dims)==3:
                self.model = dragonnet(
                    [mlp(width=self.dims[0], device=self.device, ckpt_path=self.ckpt_path, seed=seed, **kwargs),
                     mlp(width=self.dims[1], device=self.device, ckpt_path=self.ckpt_path, seed=seed+1,
                         **kwargs),
                     mlp(width=self.dims[1], device=self.device, ckpt_path=self.ckpt_path, seed=seed+2,
                         **kwargs),
                     mlp(width=self.dims[2], device=self.device, ckpt_path=self.ckpt_path, seed=seed+3,
                         **kwargs)])
            else:
                self.model = dragonnet([mlp(width=self.dims[0], device=self.device, ckpt_path=self.ckpt_path, seed=seed, **kwargs),
                                        mlp(width=self.dims[1], device=self.device, ckpt_path=self.ckpt_path, seed=seed+1,
                                            **kwargs),
                                        mlp(width=self.dims[2], device=self.device, ckpt_path=self.ckpt_path, seed=seed+2,
                                            **kwargs),
                                        mlp(width=self.dims[3], device=self.device, ckpt_path=self.ckpt_path, seed=seed+3,
                                            **kwargs)])



        # if self.model_name == 'tlearner':
        #     self.model = tlearner([mlp(width=self.dims[:-1],
        #                                 device=self.device,
        #                                 ckpt_path=self.ckpt_path, seed=seed+i, **kwargs) for i in range(2)])
        # else:
        #     self.model = mlp(width=self.dims, device=self.device, ckpt_path=self.ckpt_path, seed=seed, **kwargs)

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
        y_pred = np.squeeze(y_pred.detach().cpu().numpy())
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
                ps_pred = 1. / (1. + np.exp(-y_pred[:, 2])) #sigmoid
                ps_true = t_true
                metrics['ps_roc_auc'] = roc_auc_score(ps_true, ps_pred)
                metrics['ps_f1'] = f1_score(ps_true, ps_pred > 0.5)
                metrics['ps_acc'] = accuracy_score(ps_true, ps_pred > 0.5)
            if real_ite is not None:
                ite_pred = y_pred[:, 1] - y_pred[:, 0]
                metrics['ate'] = np.mean(ite_pred)
                metrics['pehe'] = np.sqrt(np.mean((ite_pred - np.squeeze(real_ite))**2))
        return metrics


    def custom_fit(self, dataset, steps=100, log=1,  lr=1., batch=-1,
            display_metrics=None, early_stop=False, patience=20, verbose=1,
            lr_scheduler=None, lr_patience=10, lr_factor=0.5, min_lr=1e-8):
        
        if verbose > 0:
            pbar = tqdm(range(steps), desc='description', ncols=100)
        else:
            pbar = range(steps)

        optimizer = torch.optim.Adam(self.model.get_params(), lr=lr)

        if lr_scheduler is not None:
            if lr_scheduler == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience, min_lr=min_lr)
        else:
            scheduler = None

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

        best_loss = np.inf
        patience_counter = 0

        for _ in pbar:

            n_batches_train = len(dataset['train_input']) // batch_size

            for ibt in range(n_batches_train):

                batch_start = ibt * batch_size
                batch_end = min((ibt + 1) * batch_size, len(dataset['train_input']))
                train_id = np.arange(batch_start, batch_end)

                pred_train = self.model.forward(dataset['train_input'][train_id])
                train_loss = self.get_loss(pred_train, dataset['train_label'][train_id], dataset['train_treatment'][train_id])
                loss = train_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pred_test = self.model.forward(dataset['test_input'])
            test_loss = self.get_loss(pred_test, dataset['test_label'], dataset['test_treatment'])

            # For conveniency, we get train loss and reg on the last batch only

            results['train_loss'].append(train_loss.cpu().detach().numpy())
            results['test_loss'].append(test_loss.cpu().detach().numpy())

            train_metrics = self.get_metrics(self.model.forward(dataset['train_input']), dataset['train_label'], dataset['train_treatment'], self.real_ite_train)
            test_metrics = self.get_metrics(pred_test, dataset['test_label'], dataset['test_treatment'], self.real_ite_test)
            results['train_metrics'].append(train_metrics)
            results['test_metrics'].append(test_metrics)

            if _ % log == 0 and verbose > 0:
                if display_metrics == None:
                    pbar.set_description("| train_loss: %.2e | test_loss: %.2e" % (
                    torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy()))
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

            # Step the LR scheduler using test_loss if enabled
            if scheduler is not None:
                scheduler.step(test_loss)

            if early_stop:
                if results['test_loss'][-1] < best_loss:
                    best_model = self.model.state_dict()
                    best_loss = results['test_loss'][-1]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > patience:
                        print(f'Early stopping at step {_}')
                        break
            else:
                best_model = self.model.state_dict()

        # set the model to the best model
        self.model.load_state_dict(best_model)
        return results

    def fit(self, x_train, y_train, t_train, x_test, y_test, t_test, early_stop=True, patience=30, **kwargs):

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
        results = self.custom_fit(self.dataset, early_stop=early_stop, patience=patience, **kwargs)
        # save the model
        torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, f'model.pt'))

        return results

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


class tlearner(torch.nn.Module):
    def __init__(self, model_list):
        super(tlearner, self).__init__()
        assert len(model_list) == 2, ('The T-learner model must have 2 models:'
                                      ' one for the treatment 0 and one for the treatment 1.')
        self.model = torch.nn.ModuleList(model_list)
        self.device = model_list[0].device

    def forward(self, x, **kwargs):
        # Forward pass through both networks
        y_pred_0 = self.model[0](x, **kwargs)
        y_pred_1 = self.model[1](x, **kwargs)
        return torch.cat([y_pred_0.view(-1, 1), y_pred_1.view(-1, 1)], dim=1)


    def get_params(self):
        params = []
        for i in range(len(self.model)):
            params += list(self.model[i].parameters())
        return params

class tarnet(torch.nn.Module):
    def __init__(self, model_list):
        super(tarnet, self).__init__()
        assert len(model_list) ==3, ('The TARNET model must have 3 models: one for the representation layer (phi),'
                                     ' one for the treatment 0 and one for the treatment 1.')
        self.model = torch.nn.ModuleList(model_list)
        self.device = model_list[0].device

    def forward(self, x, **kwargs):
        # Forward pass through both networks
        phi = self.model[0](x, **kwargs)  # Representation layer
        y_pred_0 = self.model[1](phi, **kwargs)
        y_pred_1 = self.model[2](phi, **kwargs)
        return torch.cat([y_pred_0.view(-1, 1), y_pred_1.view(-1, 1)], dim=1)

    def get_params(self):
        params = []
        for i in range(len(self.model)):
            params += list(self.model[i].parameters())
        return params

class dragonnet(torch.nn.Module):
    def __init__(self, model_list):
        super(dragonnet, self).__init__()
        assert len(model_list) == 4, ('The Dragonnet model must have 4 models: one for the representation layer (phi),'
                                      ' one for the treatment 0, one for the treatment 1 and one for the propensity score.')
        self.model = torch.nn.ModuleList(model_list)
        self.device = model_list[0].device

    def forward(self, x, **kwargs):
        # Forward pass through both networks
        phi = self.model[0](x, **kwargs)  # Representation layer
        y_pred_0 = self.model[1](phi, **kwargs)
        y_pred_1 = self.model[2](phi, **kwargs)
        ps_pred = self.model[3](phi, **kwargs)
        return torch.cat([y_pred_0.view(-1, 1), y_pred_1.view(-1, 1), ps_pred.view(-1, 1)], dim=1)

    def get_params(self):
        params = []
        for i in range(len(self.model)):
            params += list(self.model[i].parameters())
        return params

class slearner(torch.nn.Module):
    def __init__(self, model):
        super(slearner, self).__init__()
        self.model = model
        self.device = model.device

    def forward(self, x, **kwargs):
        # Forward pass through the model
        y_pred = self.model(x, **kwargs)
        return y_pred

    def get_params(self):
        return list(self.model.parameters())

        
class mlp(torch.nn.Module):
    '''wrapper for MLP with ckpt saving and loading'''
    def __init__(self, width, device=torch.device('cpu'), ckpt_path=None, seed=10, dropout=0.0,
                 activation='elu',
                 kernel_init=None):
        super(mlp, self).__init__()
        self.device = device
        self.width = width
        self.ckpt_path = ckpt_path
        self.seed = seed

        self.seed_all(self.seed)

        if kernel_init is not None:
            torch.manual_seed(self.seed)
            torch.nn.init.xavier_uniform_(self.weight, gain=kernel_init)


        layers = []
        for i in range(len(self.width) - 1):
            layers.append(nn.Linear(self.width[i], self.width[i + 1]))
            if i < len(self.width) - 2:  # No activation on the last layer
                if activation=='relu':
                    layers.append(nn.ReLU())
                elif activation=='elu':
                    layers.append(nn.ELU())
                elif activation=='leaky_relu':
                    layers.append(nn.LeakyReLU())
                else:
                    raise ValueError(f'Activation {activation} not recognized. Use "relu" or "elu".')
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers).to(self.device)

    def forward(self, x):
        return self.network(x)


    def seed_all(self, seed: int):
        """
        Seed CPU and GPU RNGs for reproducibility.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)





