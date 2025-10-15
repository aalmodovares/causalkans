
# Important: to run this code, please first run the hyperparameter search script hp_search/experiment_search_kan.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
import shutil
from scipy.stats import ttest_ind_from_stats

import seaborn as sns
from scipy.stats import linregress
from matplotlib.lines import Line2D

from models.kan_model import kan_net
from utils.utils_results import load_data, get_width, get_ihdp_baseline
import time
from scipy.stats import pearsonr

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

# Show the results
plt.rcParams["font.family"] = "serif"
fig_all_datasets, axs = plt.subplots(nrows=1, ncols=len(datasets), figsize=(15, 4))
fig_all_complexity, axs_c = plt.subplots(nrows=1, ncols=len(datasets), figsize=(15, 4))
fig_all_layers, axs_l = plt.subplots(nrows=1, ncols=len(datasets), figsize=(15, 4))

for i_dataset, dataset in enumerate(datasets):
    results = []
    for hidden_dim in hidden_dims:
        for lamb in lambdas:
            for lr in lrs:
                for grid in grids:
                    for sparse_init in sparse_inits:
                        for k in ks:
                            if hidden_dim == 0:
                                mkans = [False]
                            else:
                                mkans = mult_kans
                            for mult_kan in mkans:
                                file_name = os.path.join('search_experiment',
                                                         f"results_{dataset}_{hidden_dim}_{lamb}_{lr}_{grid}_{k}_{sparse_init}_{mult_kan}.pkl")
                                if os.path.exists(file_name):
                                    with open(file_name, 'rb') as f:
                                        res = pickle.load(f)
                                    results.append(res)

    print(f"Results for dataset {dataset}")

def compute_complexity(r):
    """Match the complexity scoring you used inside find_best_model()."""
    c = 0
    c += 1 if r['hidden_dims'] == 0 else (2 if r['hidden_dims'] == 5 else 3)
    c += 1 if r['lamb'] == 0.01 else 2
    c += 1 if r['grid'] == 1 else (2 if r['grid'] == 3 else 3)
    c += 1 if r['k'] == 1 else (2 if r['k'] == 3 else 3)
    c += 1 if not r['sparse_init'] else 0
    # Note: mult_kan intentionally excluded to mirror your find_best_model()
    return c


def num_layers_from_hidden_dims(hd):
    """Map hidden_dims -> #layers: 0 if 0, 1 if 5, else 2."""
    return 0 if hd == 0 else (1 if hd == 5 else 2)


def find_best_model(results, model_name):
    br = np.inf  # Best result
    bm = None  # Best model
    for r in results:
        if r[f'pehe_{model_name}'][0] < br:
            br = r[f'pehe_{model_name}'][0]
            bm = r
    print(
        f"Best {model_name} result: {br} with ATE {bm[f'ate_{model_name}']} and PEHE {bm[f'pehe_{model_name}']} with model: hidden_dims={bm['hidden_dims']}, lamb={bm['lamb']}, lr={bm['lr']}, grid={bm['grid']}, k={bm['k']}, 'mult_kan': {bm['mult_kan']}, 'sparse init': {bm['sparse_init']}")

    # Now, proceed to find all the models that are not significantly different from the best model
    _, data_test = load_data(dataset, 1)
    n = data_test.shape[0]  # Number of patients in the test split (needed for the t-test)
    threshold = 0.1  # Threshold for the t-test
    similar_models = []
    for r in results:
        _, pvalue = ttest_ind_from_stats(mean1=br, std1=bm[f'pehe_{model_name}'][1], nobs1=n,
                                         mean2=r[f'pehe_{model_name}'][0], std2=r[f'pehe_{model_name}'][1], nobs2=n,
                                         equal_var=False, alternative='less')
        if pvalue > threshold:
            r['pvalue'] = pvalue
            complexity = 0
            complexity += 1 if r['hidden_dims'] == 0 else (2 if r['hidden_dims'] == 5 else 3)
            complexity += 1 if r['lamb'] == 0.01 else 2
            complexity += 1 if r['grid'] == 1 else (2 if r['grid'] == 3 else 3)
            complexity += 1 if r['k'] == 1 else (2 if r['k'] == 3 else 3)
            complexity += 1 if not r['sparse_init'] else 0
            r['complexity'] = complexity
            similar_models.append(r)
    # Sort sm by ascending complexity for printing in a more readable way
    similar_models = sorted(similar_models, key=lambda x: x['complexity'])
    print(f"There are {len(similar_models)} similar models to the best {model_name} model:")
    for sm in similar_models:
        print(
            f"PEHE: {sm[f'pehe_{model_name}']} with pvalue {sm['pvalue']} and complexity {sm['complexity']}, ATE {sm[f'ate_{model_name}']} and PEHE {sm[f'pehe_{model_name}']}, hidden_dims={sm['hidden_dims']}, lamb={sm['lamb']}, lr={sm['lr']}, grid={sm['grid']}, k={sm['k']}, mult_kan={bm['mult_kan']}, sparse init={bm['sparse_init']}")
    print('\n')


find_best_model(results, 's_learner')
find_best_model(results, 'tlearner')
find_best_model(results, 'tarnet')
find_best_model(results, 'dragonnet')


# # Plot PEHE vs test_loss
# 1) define your models
models = {
    'S-KAN': 's_learner',
    'T-KAN': 'tlearner',
    'TARKAN': 'tarnet',
    'DragonKAN': 'dragonnet'
}

# 2) gather all log-space values to compute robust bounds
logx_all, logy_all = [], []
for key in models.values():
    src = results
    xs = np.array([r[f'test_loss_{key}'][0] for r in src])
    ys = np.array([r[f'pehe_{key}'][0] for r in src])
    logx_all.append(np.log10(xs))
    logy_all.append(np.log10(ys))
logx_all = np.concatenate(logx_all)
logy_all = np.concatenate(logy_all)

# 3) compute 1st & 99th percentiles, then floor/ceil to integer exponents
ci = 95
p1x, p99x = np.percentile(logx_all, [100 - ci, ci])
p1y, p99y = np.percentile(logy_all, [100 - ci, ci])
lx_min, lx_max = np.floor(p1x), np.ceil(p99x)
ly_min, ly_max = np.floor(p1y), np.ceil(p99y)

xtick_exps = np.arange(int(lx_min), int(lx_max) + 1)
ytick_exps = np.arange(int(ly_min), int(ly_max) + 1)
xtick_labels = [str(int(10. ** e)) for e in xtick_exps]
ytick_labels = [str(int(10. ** e)) for e in ytick_exps]

# 4) plot
plt.figure(figsize=(6, 5))
palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

other_positions = []
metrics_lines = []  # reset once per subplot, before the model loop
fig, ax = plt.subplots(figsize=(6, 5))
for (label, key), color in zip(models.items(), palette):
    src = results
    xs = np.array([r[f'test_loss_{key}'][0] for r in src])
    ys = np.array([r[f'pehe_{key}'][0] for r in src])
    logx = np.log10(xs)
    logy = np.log10(ys)

    # fit slope in log–log
    slope, intercept, _, _, _ = linregress(logx, logy)

    # regplot: scatter + line + 95% CI
    sns.regplot(
        x=logx, y=logy,
        scatter_kws={'alpha': 0.7, 'label': label, 'color': color},
        line_kws={'color': color},
        ci=95, ax=ax
    )

    ax.grid()

    sns.regplot(
        x=logx, y=logy,
        scatter_kws={'alpha': 0.7, 'label': label, 'color': color},
        line_kws={'color': color},
        ci=95, ax=axs[i_dataset]
    )

    r_val = pearsonr(logx, logy)[0]
    metrics_lines.append((f"r={r_val:.2f}", color))

    axs[i_dataset].grid()

    # annotate slope near 75th percentile point
    x_pos = np.percentile(logx, 75) + 0.2
    y_pos = intercept + slope * x_pos

    # if x_pos, y_pos is too close to a previous one, move the position slightly to right and up
    while True:
        is_close = False
        for other_x, other_y in other_positions:
            if (abs(x_pos - other_x) + abs(y_pos - other_y)) ** 2 < 0.1:  # threshold for closeness
                is_close = True
                break
        if is_close:
            x_pos += 0.2
            y_pos += 0.1
        else:
            break
    other_positions.append((x_pos, y_pos))

    ax.text(
        x_pos, y_pos,
        # f'slope={slope:.2f}',
        f'{slope:.2f}',
        color=color,
        ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )

    axs[i_dataset].text(
        x_pos, y_pos,
        # f'slope={slope:.2f}',
        f'{slope:.2f}',
        color=color,
        ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )

    # draw colored lines bottom-right, in the same order as the legend
    x_anchor = 0.99
    y_start = 0.15  # bottom margin inside the axes
    line_step = 0.045  # vertical spacing in axes coords; reduce if you have many models

    for i, (txt, col) in enumerate(metrics_lines):
        axs[i_dataset].text(
            x_anchor, y_start - i * line_step, txt,
            transform=axs[i_dataset].transAxes,
            ha='right', va='bottom',
            fontsize=9, color=col
        )

# 5) set ticks & labels in original scale
ax.set_xticks(xtick_exps)
ax.set_xticklabels(xtick_labels)
ax.set_yticks(ytick_exps)
ax.set_yticklabels(ytick_labels)

axs[i_dataset].set_xticks(xtick_exps)
axs[i_dataset].set_xticklabels(xtick_labels)
axs[i_dataset].set_yticks(ytick_exps)
axs[i_dataset].set_yticklabels(ytick_labels)

# 6) labels, title, legend
ax.set_xlabel('Test loss')
ax.set_ylabel('PEHE')
ax.set_title(f'PEHE vs Test loss for dataset {dataset}')
# plt.legend(loc='best')

# 6) custom legend
handles = [
    Line2D([0], [0],
           marker='o', color=color, linestyle='-',
           markersize=6, linewidth=1, label=label)
    for color, label in zip(palette, models.keys())
]
ax.legend(handles=handles, loc='best')

axs[0].legend(handles=handles, loc='best')
# axs[i_dataset].set_title(dataset)
axs[i_dataset].set_xlabel(f"Test loss \n {dataset.replace('_', ' ')}")

fig.tight_layout()
fig.savefig(f'../figures/all_pehe_vs_test_loss_{dataset}.png', bbox_inches='tight', dpi=200)
fig.show()
plt.close(fig)

# Collect all PEHEs (log-space) to set robust Y limits (like your test-loss plot)
logy_all_c = []
for key in models.values():
    src = results
    ys = np.array([r[f'pehe_{key}'][0] for r in src])
    logy_all_c.append(np.log10(ys))
logy_all_c = np.concatenate(logy_all_c)

# Robust y-range via percentiles (95% CI like before)
ci = 95
p1y_c, p99y_c = np.percentile(logy_all_c, [100 - ci, ci])
ly_min_c, ly_max_c = np.floor(p1y_c), np.ceil(p99y_c)
ytick_exps_c = np.arange(int(ly_min_c), int(ly_max_c) + 1)
ytick_labels_c = [str(int(10. ** e)) for e in ytick_exps_c]

# Per-dataset, per-model plot
fig_c, ax_c = plt.subplots(figsize=(6, 5))
palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

other_positions_c = []
metrics_lines_c = []

for (label, key), color in zip(models.items(), palette):
    src = results
    xs = np.array([compute_complexity(r) for r in src])  # <-- complexity on X
    ys = np.array([r[f'pehe_{key}'][0] for r in src])  # <-- PEHE on Y
    logy = np.log10(ys)

    # Fit slope in (x, log10(y)) space, to match your log–log style on Y
    slope, intercept, _, _, _ = linregress(xs, logy)

    # Regression + scatter with 95% CI
    sns.regplot(
        x=xs, y=logy,
        scatter_kws={'alpha': 0.7, 'label': label, 'color': color},
        line_kws={'color': color},
        ci=95, ax=ax_c
    )
    ax_c.grid()

    # Mirror onto the multi-panel canvas
    sns.regplot(
        x=xs, y=logy,
        scatter_kws={'alpha': 0.7, 'label': label, 'color': color},
        line_kws={'color': color},
        ci=95, ax=axs_c[i_dataset]
    )
    axs_c[i_dataset].grid()

    # Pearson r in the same transformed space as the fit
    r_val = pearsonr(xs, logy)[0]
    metrics_lines_c.append((f"r={r_val:.2f}", color))

    # Annotate slope near the 75th percentile x
    x_pos = np.percentile(xs, 75) + 0.2
    y_pos = intercept + slope * x_pos

    # De-conflict label overlap
    while True:
        is_close = False
        for ox, oy in other_positions_c:
            if (abs(x_pos - ox) + abs(y_pos - oy)) ** 2 < 0.1:
                is_close = True
                break
        if is_close:
            x_pos += 0.2
            y_pos += 0.1
        else:
            break
    other_positions_c.append((x_pos, y_pos))

    ax_c.text(
        x_pos, y_pos, f'{slope:.2f}',
        color=color, ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )
    axs_c[i_dataset].text(
        x_pos, y_pos, f'{slope:.2f}',
        color=color, ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )

    for i, (txt, col) in enumerate(metrics_lines_c):
        ax_c.text(
            x_anchor, y_start - i * line_step, txt,
            transform=ax_c.transAxes, ha='right', va='bottom',
            fontsize=9, color=col
        )
        axs_c[i_dataset].text(
            x_anchor, y_start - i * line_step, txt,
            transform=axs_c[i_dataset].transAxes, ha='right', va='bottom',
            fontsize=9, color=col
        )

# Axis ticks & labels
# X: integer complexity values
all_x = []
for key in models.values():
    src = results
    all_x.extend([compute_complexity(r) for r in src])
x_min, x_max = min(all_x), max(all_x)
x_ticks = np.arange(x_min, x_max + 1, 1)

ax_c.set_xticks(x_ticks)
axs_c[i_dataset].set_xticks(x_ticks)

# Y: show ticks in original PEHE scale (labels are powers of 10)
ax_c.set_yticks(ytick_exps_c)
ax_c.set_yticklabels(ytick_labels_c)
axs_c[i_dataset].set_yticks(ytick_exps_c)
axs_c[i_dataset].set_yticklabels(ytick_labels_c)

# Labels, title, legend
ax_c.set_xlabel('Complexity')
ax_c.set_ylabel('PEHE')
ax_c.set_title(f'PEHE vs Complexity for dataset {dataset}')

handles_c = [
    Line2D([0], [0], marker='o', color=color, linestyle='-',
           markersize=6, linewidth=1, label=label)
    for color, label in zip(palette, models.keys())
]
ax_c.legend(handles=handles_c, loc='best')
axs_c[0].legend(handles=handles_c, loc='best')

axs_c[i_dataset].set_xlabel(f"Complexity \n {dataset.replace('_', ' ')}")

fig_c.tight_layout()
fig_c.savefig(f'../figures/all_pehe_vs_complexity_{dataset}.png', bbox_inches='tight', dpi=200)
fig_c.show()
plt.close(fig_c)

logy_all_l = []
for key in models.values():
    src =  results
    ys = np.array([r[f'pehe_{key}'][0] for r in src])
    logy_all_l.append(np.log10(ys))
logy_all_l = np.concatenate(logy_all_l)

ci = 95
p1y_l, p99y_l = np.percentile(logy_all_l, [100 - ci, ci])
ly_min_l, ly_max_l = np.floor(p1y_l), np.ceil(p99y_l)
ytick_exps_l = np.arange(int(ly_min_l), int(ly_max_l) + 1)
ytick_labels_l = [str(int(10. ** e)) for e in ytick_exps_l]

fig_l, ax_l = plt.subplots(figsize=(6, 5))
palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

other_positions_l = []
metrics_lines_l = []

for (label, key), color in zip(models.items(), palette):
    src = results
    xs = np.array([num_layers_from_hidden_dims(r['hidden_dims']) for r in src])  # <-- #layers on X
    ys = np.array([r[f'pehe_{key}'][0] for r in src])  # <-- PEHE on Y
    logy = np.log10(ys)

    # Linear fit in (x, log10(y)) space (discrete x, log-scaled y)
    slope, intercept, _, _, _ = linregress(xs, logy)

    # Regression + scatter with 95% CI
    sns.regplot(
        x=xs, y=logy,
        scatter_kws={'alpha': 0.7, 'label': label, 'color': color},
        line_kws={'color': color},
        ci=95, ax=ax_l
    )
    ax_l.grid()

    # Mirror onto the multi-panel canvas
    sns.regplot(
        x=xs, y=logy,
        scatter_kws={'alpha': 0.7, 'label': label, 'color': color},
        line_kws={'color': color},
        ci=95, ax=axs_l[i_dataset]
    )
    axs_l[i_dataset].grid()

    # Pearson r in the same transformed space
    r_val = pearsonr(xs, logy)[0]
    metrics_lines_l.append((f"r={r_val:.2f}", color))

    # Annotate slope near 75th percentile x (works fine for discrete x too)
    x_pos = 1
    y_pos = intercept + slope * x_pos

    # De-conflict overlap
    while True:
        is_close = False
        for ox, oy in other_positions_l:
            if (abs(x_pos - ox) + abs(y_pos - oy)) ** 2 < 0.1:
                is_close = True
                break
        if is_close:
            x_pos += 0.1
            y_pos += 0.05
        else:
            break
    other_positions_l.append((x_pos, y_pos))

    ax_l.text(
        x_pos, y_pos, f'{slope:.2f}',
        color=color, ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )
    axs_l[i_dataset].text(
        x_pos, y_pos, f'{slope:.2f}',
        color=color, ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )

    for i, (txt, col) in enumerate(metrics_lines_l):
        ax_l.text(
            x_anchor, y_start - i * line_step, txt,
            transform=ax_l.transAxes, ha='right', va='bottom',
            fontsize=9, color=col
        )
        axs_l[i_dataset].text(
            x_anchor, y_start - i * line_step, txt,
            transform=axs_l[i_dataset].transAxes, ha='right', va='bottom',
            fontsize=9, color=col
        )

# X ticks are discrete {0,1,2}
ax_l.set_xticks([0, 1, 2])
axs_l[i_dataset].set_xticks([0, 1, 2])

# Y ticks show original PEHE magnitudes (powers of 10)
ax_l.set_yticks(ytick_exps_l)
ax_l.set_yticklabels(ytick_labels_l)
axs_l[i_dataset].set_yticks(ytick_exps_l)
axs_l[i_dataset].set_yticklabels(ytick_labels_l)

# Labels, title, legend
ax_l.set_xlabel('Number of layers')
ax_l.set_ylabel('PEHE')
ax_l.set_title(f'PEHE vs #Layers for dataset {dataset}')

handles_l = [
    Line2D([0], [0], marker='o', color=color, linestyle='-',
           markersize=6, linewidth=1, label=label)
    for color, label in zip(palette, models.keys())
]
ax_l.legend(handles=handles_l, loc='best')
axs_l[0].legend(handles=handles_l, loc='best')

axs_l[i_dataset].set_xlabel(f"#Layers \n {dataset.replace('_', ' ')}")

fig_l.tight_layout()
fig_l.savefig(f'../figures/all_pehe_vs_layers_{dataset}.png', bbox_inches='tight', dpi=200)
fig_l.show()
plt.close(fig_l)

# configure the figure for all datasets
# fig_all_datasets.suptitle('PEHE vs Test loss for all datasets', fontsize=16)
axs[0].set_ylabel('PEHE')
# fig_all_datasets.supxlabel('Test loss')
fig_all_datasets.tight_layout()
# save fig
fig_all_datasets.savefig(f'../figures/all_datasets_pehe_vs_test_loss.png', bbox_inches='tight', dpi=200)
plt.close(fig_all_datasets)

axs_c[0].set_ylabel('PEHE')
fig_all_complexity.tight_layout()
fig_all_complexity.savefig('../figures/all_datasets_pehe_vs_complexity.png', bbox_inches='tight', dpi=200)
plt.close(fig_all_complexity)

axs_l[0].set_ylabel('PEHE')
fig_all_layers.tight_layout()
fig_all_layers.savefig('../figures/all_datasets_pehe_vs_layers.png', bbox_inches='tight', dpi=200)
plt.close(fig_all_layers)
