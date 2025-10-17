import os

import numpy as np
import pandas as pd
import sympy as sp
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from typing import Union, Tuple

from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.projections.polar import PolarAxes
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.projections import register_projection
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def radar_factory(num_vars, frame='circle'):  # Adapted from https://stackoverflow.com/questions/52910187/how-to-make-a-polygon-radar-spider-chart-in-python
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'

        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, fontsize=14):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=fontsize)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def plot_sorted_variances(x_train, x_test, binary, delta_train, delta_test, n_logits, args, dataset):
    variances_train = [d.var(axis=0) for d in delta_train]
    variances_test = [d.var(axis=0) for d in delta_test]
    for i in range(n_logits):
        plt.figure(figsize=(8, 9))
        idxs_train = np.argsort(variances_train[i])[::-1]  # Sort by training variance

        # plt.plot(variances_train[i].values[idxs_train], label=f'train')
        # plt.plot(variances_test[i].values[idxs_train], label=f'test')

        labels = variances_train[i].index[idxs_train]
        x = np.arange(len(labels))

        # Barras para train y test (desplazadas para no solaparse)
        width = 0.4
        plt.bar(x - width / 2, variances_train[i].values[idxs_train], width=width, label='Train')
        plt.bar(x + width / 2, variances_test[i].values[idxs_train], width=width, label='Test')


        plt.title(f'Variance of delta values for logit {i}', fontsize=22)
        plt.xlabel('Feature', fontsize=20)
        plt.ylabel('Variance', fontsize=20)
        plt.legend(loc='best', fontsize=18)
        # Add the feature names
        plt.xticks(ticks=np.arange(len(variances_train[i])),
                   labels=variances_train[i].index[idxs_train],
                   rotation=90, fontsize=18)
        plt.yticks(fontsize=18)
        # Ensure that ticks are not cut off
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(args['results_folder'], dataset, f'variances_{i}.png'), bbox_inches='tight', dpi=600)
        plt.close()

    for i in range(n_logits):
        logit = 1 if binary else i
        for feat in x_train.columns:
            if variances_train[i][feat] > 1e-6:  # Only plot the features with a variance above a threshold
                plt.scatter(x_train[feat], delta_train[i][feat], label=f"{feat}_{logit}_train")
                plt.scatter(x_test[feat], delta_test[i][feat], label=f"{feat}_{logit}_test")

                # Plot the average delta values as well
                plt.plot(x_train[feat].unique(), delta_train[i][feat].mean() * np.ones_like(x_train[feat].unique()),
                         color='b', linestyle='-')
                plt.plot(x_test[feat].unique(), delta_test[i][feat].mean() * np.ones_like(x_test[feat].unique()),
                         color='r', linestyle='-')

                plt.title(f'Delta for {feat} and logit {logit}', fontsize=22)
                plt.xlabel(feat, fontsize=20)
                plt.ylabel('Delta', fontsize=20)
                plt.legend(loc='best', fontsize=18)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.savefig(os.path.join(args['results_folder'], dataset, f'delta_{feat}_logit{logit}.png'),
                            bbox_inches='tight', dpi=600)
                plt.close()  # Note: a higher delta means a higher risk


def plot_binary_explanation_plot(y_true, y_pred_proba, labels, threshold, outfile=None, title='Probability of positive class'):  # Added by Juan
    y_pred_proba = np.squeeze(np.array(y_pred_proba))

    # Umbralize the probabilities: the minimum probability is 0.01
    y_pred_proba = np.where(y_pred_proba < 0.01, 0.01, y_pred_proba)
    y_true = np.squeeze(np.array(y_true))
    fig, ax = plt.subplots()
    sort_idx = np.argsort(y_pred_proba)[::-1]  # Sort the patients by probability

    # Plot a bar diagram of the probability of each patient, where the color bar depends on the y_true value
    color_vals = [['r', 'g'][int(y)] for y in y_true[sort_idx]]
    ax.bar(range(len(y_pred_proba)), y_pred_proba[sort_idx], color=color_vals)
    ax.axhline(threshold, color='k', linestyle='--', label='Threshold')
    ax.set_yscale('log')  # Plot in log scale the vertical axis for better visualization
    ax.set_xlabel('Patient', fontsize=20)
    ax.set_ylabel('Log-Probability', fontsize=20)
    ax.set_title(title, fontsize=22)
    ax.tick_params(axis='both', labelsize=18)

    # Add the legent: red for the first label, green for the second label
    red_patch = mpatches.Patch(color='red', label=labels[0])
    green_patch = mpatches.Patch(color='green', label=labels[1])
    ax.legend(handles=[red_patch, green_patch], loc='best', fontsize=18)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile + '_explanation.png', bbox_inches='tight', dpi=600)
    plt.close()

def eval_expr_on_df(expr, df, constants=None):
    """
    Evaluate a SymPy expression on all rows of a pandas DataFrame.

    Parameters
    ----------
    expr : sympy.Expr
        The symbolic expression to evaluate.
    df : pandas.DataFrame
        Must contain columns named exactly as the symbols in `expr`.
    constants : dict, optional
        Mapping {symbol_name: value} for fixed parameters to substitute.

    Returns
    -------
    np.ndarray
        1D array with one value per row.
    """
    # Optional constant substitution (e.g., hyperparameters)
    if constants:
        expr = expr.subs({sp.Symbol(k): v for k, v in constants.items()})

    # Symbols required by the expression (deterministic order by name)
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    sym_names = [s.name for s in syms]

    # Sanity checks
    missing = [c for c in sym_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    # Build vectorized function and evaluate
    f = sp.lambdify(syms, expr, modules="numpy")
    args = [df[name].to_numpy() for name in sym_names]
    out = f(*args)

    # Ensure 1D ndarray
    return np.asarray(out).reshape(-1)

def get_patient_values(delta_formula, x_in):
    x = x_in.to_numpy()
    if isinstance(delta_formula, sp.Float):  # WE have a constant!
        delta = np.zeros((x.shape[0], x.shape[1] + 1))
        delta[:, -1] = float(delta_formula)  # There is only a constant term!
    else:
        delta = np.zeros(
            (x.shape[0], x.shape[1] + 1))  # One input per covariate, one extra output for the constant term
        for i in range(x.shape[0]):  # For each patient
            for fs in delta_formula.args:
                formula_sum_term = copy.deepcopy(fs)
                if isinstance(formula_sum_term, sp.Float):  # We have a constant!
                    delta[i, -1] = float(formula_sum_term)
                else:  # Since it is a KAAM, it depends on a single variable
                    assert len(formula_sum_term.free_symbols) == 1
                    variable_in_the_expresion = list(formula_sum_term.free_symbols)[0]
                    variable_index = x_in.columns.get_loc(str(variable_in_the_expresion))
                    delta[i, variable_index] += float(
                        formula_sum_term.subs(variable_in_the_expresion, x[i, variable_index]))
    delta = pd.DataFrame(delta, columns=x_in.columns.tolist() + ['const'])
    return delta

def get_delta(x_train, formula):
    '''
    get the delta values for each patient in x_train according to the formula
    x_train: pandas dataframe with the covariates
    formula: sympy expression with the formula
    returns: delta_formula, delta_train. Note that delta_train has as many elements as outputs the KAN has

    '''

    delta_formula = formula

    # Since the formula may have pruned variables, we keep only the variables that are present in the formula
    actual_vars = []

    for i, col in enumerate(x_train.columns):
        delta_formula = delta_formula.subs(sp.symbols(f'x_{i + 1}'), sp.symbols(col))

    actual_vars += [str(s) for s in delta_formula.free_symbols]
    actual_vars = list(set(actual_vars))  # Remove duplicates
    x_train = x_train[actual_vars]

    delta_train, delta_test = [], []

    d = get_patient_values(delta_formula, x_train)
    delta_train.append(d)

    return delta_formula, delta_train

def get_pehe(ite_pred, ite_real):  # Formula to get the PEHE metric
    return np.sqrt(np.mean((np.squeeze(ite_pred) - np.squeeze(ite_real)) ** 2))

def get_formula_values(formula, x_in):  # Expression that takes a sympy expression and a pandas dataframe and returns the values of the expression for each row of the dataframe
    x = x_in.to_numpy()
    col_names = x_in.columns.tolist()
    fval = np.zeros((x.shape[0], 1))
    for i in tqdm(range(x.shape[0])):
        f = formula.copy()
        for j in range(x.shape[1]):
            f = f.subs(col_names[j], x[i, j])
        fval[i] = float(f)
    return fval


def plot_pdp_delta(x:pd.DataFrame, formula, cols=None, subplots_grid=None, outcome='y', ids=None, colors=None, figsize=None):
    delta_formula, delta_matrix = get_delta(x, formula)
    delta_matrix = delta_matrix[0].iloc[:, :-1]  # Remove the constant term
    if cols is None:
        cols = delta_matrix.columns.tolist()
        cols = [col for col in x.columns if col in cols]

    index_cols = [i for i, col in enumerate(delta_matrix.columns) if col in cols]
    # delta_matrix = delta_matrix.iloc[index_cols, index_cols]

    if subplots_grid is not None:
        assert sum(subplots_grid) == delta_matrix.shape[0]
    else:
        if len(delta_matrix.columns)//2 ==0:
            subplots_grid = (len(delta_matrix.columns)/2, 2)
        elif len(delta_matrix.columns)//3 ==0:
            subplots_grid = (len(delta_matrix.columns)/3, 3)
        else:
            subplots_grid = (len(delta_matrix.columns), 1)

    if figsize is None:
        fig, axes = plt.subplots(subplots_grid[0], subplots_grid[1])
    else:
        fig, axes = plt.subplots(subplots_grid[0], subplots_grid[1], figsize=figsize)

    for i, var in enumerate(delta_matrix.columns):
        if subplots_grid[1]>1:
            ax = axes[i//subplots_grid[1], i%subplots_grid[1]]
        else:
            ax = axes[i]
        ax.set_xlabel(var)
        # ax.set_ylabel(r'$$\Delta$$' + outcome)
        ax.set_ylabel(f'Delta {outcome}')

        sorted_idx = np.argsort(x[var])
        ax.plot(x[var].values[sorted_idx], delta_matrix[var].values[sorted_idx], color='b')

        if ids is not None:
            colors = ['r', 'g', 'm', 'y', 'c']  if colors is None else colors
            # represent the points of the patients of patients_id
            for j, id in enumerate(ids):
                ax.scatter(x[var].values[id], delta_matrix[var].values[id], color=colors[j],
                           label=f'Patient {id}')

    return fig, axes


def plot_prp(x:pd.DataFrame, formula, ids, cols=None, colors=None):
    delta_formula, delta_matrix = get_delta(x, formula)
    delta_matrix = delta_matrix[0].iloc[:, :-1]  # Remove the constant term

    if cols is None:
        cols = delta_matrix.columns.tolist()
        cols = [col for col in x.columns if col in cols]

    theta = radar_factory(len(cols), frame='polygon')

    index_cols = [i for i, col in enumerate(delta_matrix.columns) if col in cols]
    delta_matrix = delta_matrix.iloc[index_cols, index_cols]
    fig, ax = plt.subplots(subplot_kw=dict(projection='radar'))
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])

    avg_delta = sum(np.mean(delta_matrix, axis=0)) * np.ones(len(cols))

    _ = ax.plot(theta, avg_delta, label='Avg', color='gray')
    ax.fill(theta, avg_delta, alpha=0.1, color='gray')

    avg_delta_ = delta_matrix.mean(axis=0).values[None, :]
    avg_matrix = np.repeat(avg_delta_, delta_matrix.shape[1], axis=0)

    for j, id in enumerate(ids):
        np.fill_diagonal(avg_matrix, delta_matrix.iloc[id].values)
        pat_diff = avg_matrix.sum(axis=1)
        _ = ax.plot(theta, pat_diff, label=f'Individual {id}', color=colors[j])
        ax.fill(theta, pat_diff, alpha=0.2, color=colors[j])

    ax.set_varlabels(cols)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    return fig, ax


def plot_ice(x, formula, ids=None, cols=None, hue=None, colors=None, centered=False, outcome='y', figsize=None):
    """
    Plot Individual Conditional Expectation (ICE) curves.

    Behavior:
    - If ids=None: plot all individuals. If hue is provided, color by hue and show hue legend.
    - If ids is specified: color each by given colors (or black) and show ID legend.
    - If hue is numeric with exactly two unique values, it is treated as categorical.
    """
    legend_flag = True
    if cols is None:
        cols = x.columns.tolist()
    if hue is not None and hue in cols:
        cols.remove(hue)

    # Determine individuals to plot
    if ids is None:
        ids = list(np.arange(x.shape[0]))
        legend_flag = False  # will show hue legend instead if hue is given

    alpha = 1
    hue_is_categorical = False
    color_dict = None
    sm = None
    show_colorbar = False

    # --- Decide how to color ---
    if colors is None:
        if hue is not None and (len(ids) == x.shape[0]):
            hue_vals = x[hue].values
            unique_vals = np.unique(hue_vals)
            is_numeric = np.issubdtype(x[hue].dtype, np.number)

            # Treat numeric with only two unique values as categorical
            if (not is_numeric) or (len(unique_vals) == 2):
                hue_is_categorical = True
                cmap = plt.cm.get_cmap('tab10')
                color_dict = {v: cmap(i) for i, v in enumerate(unique_vals)}
                colors = [color_dict[v] for v in hue_vals]
            else:
                norm = mpl.colors.Normalize(vmin=hue_vals.min(), vmax=hue_vals.max())
                cmap = plt.cm.viridis
                sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
                colors = [cmap(norm(v)) for v in hue_vals]
                show_colorbar = True
        else:
            colors = ['black'] * len(ids)
            alpha = 0.3
    else:
        assert len(ids) == len(colors)

    # --- Plotting ---
    if figsize is None:
        fig, axes = plt.subplots(len(cols), 1, squeeze=False)
    else:
        fig, axes = plt.subplots(len(cols), 1, squeeze=False, figsize=figsize)
    axes = axes.flatten()

    def _pred(df):
        return np.asarray(get_formula_values(formula, df)).reshape(-1)

    for i, feat in enumerate(cols):
        ax = axes[i]
        grid = np.sort(x[feat].unique())

        # Compute predictions along the grid
        outcomes = np.vstack([
            _pred(x.assign(**{feat: val}))
            for val in grid
        ])  # shape (len(grid), n_samples)

        # Centering
        if centered:
            baseline = _pred(x.assign(**{feat: x[feat].mean()}))
            outcomes_centered = outcomes - baseline[None, :]
        else:
            outcomes_centered = outcomes

        # Plot each individual's curve
        for j, idx in enumerate(ids):
            col_idx = idx  # x row index
            y_vals = outcomes_centered[:, col_idx]
            ax.plot(grid, y_vals, color=colors[col_idx], alpha=0.3 if centered else alpha,
                    label=(f'ID {idx}' if legend_flag else None))

            factual_x = x.iloc[idx][feat]
            if factual_x in grid:
                row = np.where(grid == factual_x)[0][0]
                ax.scatter(factual_x, y_vals[row], color=colors[col_idx], marker='o', s=60,
                           zorder=5, alpha=0.3)

        ax.set_xlabel(feat)
        ax.set_ylabel(outcome)
        ax.grid(True)
        if i == 0 and legend_flag:
            ax.legend(loc='best', fontsize=10)

    # --- Hue legend or colorbar (RIGHT SIDE) ---
    if hue is not None and (len(ids) == x.shape[0]) and not legend_flag:
        if hue_is_categorical:
            handles = [mpl.lines.Line2D([0], [0], color=color_dict[v], lw=3, label=str(v))
                       for v in color_dict]
            # Legend outside, right side
            fig.subplots_adjust(right=0.8)
            fig.legend(handles=handles, title=hue, loc='center left', bbox_to_anchor=(0.82, 0.5))
        elif show_colorbar and sm is not None:
            # Colorbar outside, right side
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(hue)

    plt.tight_layout(rect=[0, 0, 0.80, 1])  # leave space on right
    return fig, axes

def plot_interact_contour(x, formula: Union[str, sp.Expr], feat1: str, feat2: str,
                          outcome: str = 'y', grid_size: int = 20, figsize=None,
                          colormap='bwr', levels=30):
    """
    Plot three filled contour maps in a (1,3) subplot layout to interpret a SymPy formula
    over two selected covariates (feat1, feat2).

    Subplot 1: additive components only, i.e., terms that depend solely on feat1 OR solely on feat2.
               (g(x1) + g(x2))
    Subplot 2: pure interaction components between feat1 and feat2, i.e., terms that factor as
               f1(feat1) * f2(feat2) and involve NO other variables.
    Subplot 3: full formula restricted to terms that involve ONLY feat1 and/or feat2
               (drops any term that depends on variables outside {feat1, feat2}).

    Notes and assumptions:
    - `formula` must be a SymPy expression (or a string parsable by sympy.sympify) that is numerically
      evaluable via `sympy.lambdify` with NumPy. If you use undefined SymPy Functions (e.g., F(x)),
      you must provide them as numerically evaluable functions or rewrite the formula to standard
      NumPy-supported operations (sin, exp, **, etc.).
    - The function automatically identifies additive vs. interaction terms by analyzing free symbols
      and separability with sympy.separatevars.
    - Interaction terms include only products that can be separated as f1(feat1)*f2(feat2) with no
      dependence on other variables and BOTH variables present. Sums like sin(x1 + x2) are not
      considered interaction for subplot 2 (not separable), but they are included in subplot 3.

    Parameters
    ----------
    x : pandas.DataFrame
        DataFrame containing at least columns `feat1` and `feat2`. Used to set plotting ranges.
    formula : str or sympy.Expr
        The model formula as a SymPy expression or a string to be parsed by sympy.sympify.
    feat1 : str
        Name of the first covariate (must match the symbol name used in `formula`).
    feat2 : str
        Name of the second covariate (must match the symbol name used in `formula`).
    outcome : str, default 'y'
        (Not used in plotting; kept for signature compatibility.)
    grid_size : int, default 20
        Number of points per axis for the evaluation grid.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib Figure containing the three subplots.
    axes : np.ndarray
        Array of Axes objects for further customization if needed.
    """

    # --- 1) Parse/validate the formula and get the two symbols ---
    if isinstance(formula, str):
        expr = sp.sympify(formula)
    elif isinstance(formula, sp.Expr):
        expr = formula
    else:
        raise TypeError("`formula` must be a string or a SymPy expression.")

    s1 = sp.Symbol(feat1)
    s2 = sp.Symbol(feat2)

    # --- 2) Helper: top-level additive terms (without forcibly expanding products) ---
    # We only split by top-level addition; each `term` is an additive component.
    terms = sp.Add.make_args(expr)

    # --- 3) Classify terms into (a) additive-only, (b) separable interaction, (c) x1/x2-only total ---
    add_terms = []      # terms depending on {s1} only OR {s2} only
    inter_terms = []    # terms depending on both {s1,s2} AND separable as f1(s1)*f2(s2)
    total_terms = []    # terms whose free_symbols ⊆ {s1,s2} (everything with only x1/x2)

    for t in terms:
        fs = t.free_symbols
        # Keep any term that uses no symbols (a constant): it is independent of (s1,s2).
        # For this task we drop pure constants from plots 1 and 2, but include them in total if you wish.
        # We will EXCLUDE constants from additive and interaction (as per description), but INCLUDE in total.
        if not fs:
            total_terms.append(t)
            continue

        # Terms that involve only feat1/feat2 (no other variables)
        if fs.issubset({s1, s2}):
            total_terms.append(t)

            # Additive-only: depends on exactly one of the two symbols
            if fs == {s1} or fs == {s2}:
                add_terms.append(t)
            else:
                # Potential interaction: try to separate as f1(s1)*f2(s2)
                # separatevars(..., dict=True) tries to factor into symbol-specific pieces.
                sep = sp.separatevars(t, symbols=[s1, s2], dict=True, force=True)
                if isinstance(sep, dict) and s1 in sep and s2 in sep:
                    # Ensure no leftover factor depends on s1/s2 (sep['coeff'] should be constant w.r.t s1/s2)
                    coeff = sep.get('coeff', sp.Integer(1))
                    if coeff.free_symbols.isdisjoint({s1, s2}):
                        inter_terms.append(t)
        else:
            # Term depends on variables beyond {s1, s2} -> ignore for all three plots.
            continue

    # Build expressions for each subplot (use 0 if empty to keep lambdify happy)
    expr_add = sp.Add(*add_terms) if add_terms else sp.Integer(0)
    expr_inter = sp.Add(*inter_terms) if inter_terms else sp.Integer(0)
    expr_total = sp.Add(*total_terms) if total_terms else sp.Integer(0)

    # --- 4) Prepare evaluation grid from DataFrame ranges ---
    x1_min, x1_max = float(np.nanmin(x[feat1].values)), float(np.nanmax(x[feat1].values))
    x2_min, x2_max = float(np.nanmin(x[feat2].values)), float(np.nanmax(x[feat2].values))

    # Avoid degenerate ranges
    if x1_min == x1_max:
        x1_min, x1_max = x1_min - 1.0, x1_max + 1.0
    if x2_min == x2_max:
        x2_min, x2_max = x2_min - 1.0, x2_max + 1.0

    x1_lin = np.linspace(x1_min, x1_max, grid_size)
    x2_lin = np.linspace(x2_min, x2_max, grid_size)
    X1, X2 = np.meshgrid(x1_lin, x2_lin, indexing='xy')

    # --- 5) Lambdify the three expressions (NumPy backend) ---
    # If your expression contains custom/undefined functions, pass a modules dict to lambdify with numeric callables.
    f_add = sp.lambdify((s1, s2), expr_add, modules=['numpy'])
    f_inter = sp.lambdify((s1, s2), expr_inter, modules=['numpy'])
    f_total = sp.lambdify((s1, s2), expr_total, modules=['numpy'])

    # Safe numerical evaluation
    def _safe_eval(func, X1, X2):
        """Evaluate func on the grid and coerce to float array; safely replace nan/inf."""
        try:
            Z = func(X1, X2)
            Z = np.asarray(Z, dtype=float)
        except Exception as e:
            print(
                f"[plot_interact_contour] Warning: evaluation failed with error: {e}. Returning zeros for this panel.")
            Z = np.zeros_like(X1, dtype=float)

        # Replace NaN and inf values with finite ones for stable contouring
        finite_mask = np.isfinite(Z)
        if np.any(finite_mask):
            Z_min, Z_max = np.min(Z[finite_mask]), np.max(Z[finite_mask])
        else:
            Z_min, Z_max = 0.0, 0.0
        Z = np.nan_to_num(Z, nan=0.0, posinf=Z_max, neginf=Z_min)

        return Z

    Z_add = _safe_eval(f_add, X1, X2)
    Z_inter = _safe_eval(f_inter, X1, X2)
    Z_total = _safe_eval(f_total, X1, X2)

    vmin = np.min([np.nanmin(Z_add), np.nanmin(Z_inter), np.nanmin(Z_total)])
    vmax = np.max([np.nanmax(Z_add), np.nanmax(Z_inter), np.nanmax(Z_total)])

    # --- 6) Plotting ---
    if figsize is None:
        fig, axes = plt.subplots(1, 3, constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 3, constrained_layout=True, figsize=figsize)

    # Panel 1: additive only
    cs1 = axes[0].contourf(X1, X2, Z_add, levels=levels, cmap=colormap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Additive")
    axes[0].set_xlabel(feat1)
    axes[0].set_ylabel(feat2)
    # fig.colorbar(cs1, ax=axes[0], shrink=0.9)

    # Panel 2: pure interaction (use a diverging colormap to highlight sign changes)
    cs2 = axes[1].contourf(X1, X2, Z_inter, levels=levels, cmap=colormap, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Interaction")
    axes[1].set_xlabel(feat1)
    # axes[1].set_ylabel(feat2)
    axes[1].set_yticks([])
    # fig.colorbar(cs2, ax=axes[1], shrink=0.9)

    # Panel 3: full x1/x2-only part of the formula
    cs3 = axes[2].contourf(X1, X2, Z_total, levels=levels, cmap=colormap, vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Total")
    axes[2].set_xlabel(feat1)
    # axes[2].set_ylabel(feat2)
    axes[2].set_yticks([])
    fig.colorbar(cs3, ax=axes[2], shrink=0.9)


    # Uniform limits to make the three panels visually comparable (optional)
    # You can comment this block if you prefer independent color scaling.
    vmin = min(np.nanmin(Z_add), np.nanmin(Z_inter), np.nanmin(Z_total))
    vmax = max(np.nanmax(Z_add), np.nanmax(Z_inter), np.nanmax(Z_total))
    for ax, Z in zip(axes, [Z_add, Z_inter, Z_total]):
        for c in ax.collections:
            c.set_clim(vmin, vmax)

    return fig, axes

def add_rugplot(ax, values, color='k', height_frac=0.03, alpha=0.4):
    """Draw small vertical ticks along x-axis to show data density."""
    y_min, y_max = ax.get_ylim()
    rug_height = (y_max - y_min) * height_frac
    for v in values:
        ax.plot([v, v], [y_min, y_min + rug_height],
                color=color, alpha=alpha, lw=0.8, solid_capstyle='round')
    ax.set_ylim(y_min, y_max)  # restore limits

def add_histogram_under(ax, values, bins=30, height_frac=0.5, alpha=0.25, color='r'):
    """
    Draw a density histogram as a filled band at the bottom of the plot.
    The histogram is density-normalized and then scaled to a fraction of the
    current y-range so it remains readable regardless of the y-axis scale.
    No additional axis or scale is shown.
    """
    # Preserve current limits
    y_min, y_max = ax.get_ylim()
    band_height = (y_max - y_min) * height_frac

    counts, bin_edges = np.histogram(values, bins=bins, density=True)
    if counts.size == 0 or np.max(counts) == 0:
        return
    counts = counts / np.max(counts)  # normalize to [0, 1]

    # Build a filled step-like polygon hugging the bottom
    xs = [bin_edges[0]]
    ys = [y_min]
    for c, left, right in zip(counts, bin_edges[:-1], bin_edges[1:]):
        top = y_min + c * band_height
        xs.extend([left, right])
        ys.extend([top, top])
    xs.append(bin_edges[-1])
    ys.append(y_min)

    ax.fill(xs, ys, color=color, alpha=alpha, linewidth=0, zorder=0)
    ax.hlines(y_min, bin_edges[0], bin_edges[-1], color=color, alpha=alpha*0.9, lw=0.8)
    ax.set_ylim(y_min, y_max)  # restore limits

def plot_pdp_additive(x, formula: Union[str, sp.Expr], feat1: str, feat2: str, grid_size: int = 200, figsize=None,
                      rugplot=False, histogram=True):
    """
    Plot two Partial Dependence (additive-only) curves:
      - Left: sum of all additive terms that depend only on feat1 (g11(feat1))
      - Right: sum of all additive terms that depend only on feat2 (g21(feat2))

    Identification rule (strict additive):
      - A term contributes to g11 if its free symbols == {feat1}
      - A term contributes to g21 if its free symbols == {feat2}
      - Constants and any term involving both variables (interaction, separable or not) are excluded.

    Parameters
    ----------
    x : pandas.DataFrame
        DataFrame with columns feat1 and feat2 (used to set plotting ranges).
    formula : str or sympy.Expr
        SymPy expression or a parsable string.
    feat1, feat2 : str
        Names of the covariates (must match symbol names in `formula`).
    grid_size : int, default 200
        Number of points for each PDP curve.

    Returns
    -------
    fig, axes : Matplotlib Figure and Axes for further customization.
    """
    # --- Parse formula and symbols ---
    expr = sp.sympify(formula) if isinstance(formula, str) else formula
    s1, s2 = sp.Symbol(feat1), sp.Symbol(feat2)

    # --- Collect top-level additive terms ---
    terms = sp.Add.make_args(expr)
    terms_s1_only, terms_s2_only = [], []

    for t in terms:
        fs = t.free_symbols
        if fs == {s1}:
            terms_s1_only.append(t)
        elif fs == {s2}:
            terms_s2_only.append(t)
        # else: ignore constants and anything that involves other variables or both s1 and s2

    expr_g1 = sp.Add(*terms_s1_only) if terms_s1_only else sp.Integer(0)
    expr_g2 = sp.Add(*terms_s2_only) if terms_s2_only else sp.Integer(0)

    # --- Build evaluation grids from data ranges ---
    x1_min, x1_max = float(np.nanmin(x[feat1])), float(np.nanmax(x[feat1]))
    x2_min, x2_max = float(np.nanmin(x[feat2])), float(np.nanmax(x[feat2]))
    if x1_min == x1_max:
        x1_min, x1_max = x1_min - 1.0, x1_max + 1.0
    if x2_min == x2_max:
        x2_min, x2_max = x2_min - 1.0, x2_max + 1.0

    X1 = np.linspace(x1_min, x1_max, grid_size)
    X2 = np.linspace(x2_min, x2_max, grid_size)

    # --- Numeric evaluation helpers ---
    f_g1 = sp.lambdify(s1, expr_g1, modules=['numpy'])
    f_g2 = sp.lambdify(s2, expr_g2, modules=['numpy'])

    def _safe_eval_1d(func, X):
        """Evaluate 1D SymPy->NumPy function safely and sanitize NaN/Inf."""
        try:
            Y = np.asarray(func(X), dtype=float)
        except Exception as e:
            print(f"[plot_pdp_additive] Warning: evaluation failed ({e}). Returning zeros.")
            Y = np.zeros_like(X, dtype=float)
        finite = np.isfinite(Y)
        if np.any(finite):
            y_min, y_max = np.min(Y[finite]), np.max(Y[finite])
        else:
            y_min, y_max = 0.0, 0.0
        Y = np.nan_to_num(Y, nan=0.0, posinf=y_max, neginf=y_min)
        return Y

    Y1 = _safe_eval_1d(f_g1, X1)
    Y2 = _safe_eval_1d(f_g2, X2)

    # --- Plotting ---
    if figsize is None:
        fig, axes = plt.subplots(1, 2, constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    axes[0].plot(X1, Y1)
    axes[0].set_xlabel(feat1)
    axes[0].set_ylabel("Additive contribution")

    axes[1].plot(X2, Y2)
    axes[1].set_xlabel(feat2)
    # axes[1].set_ylabel("Additive contribution")
    if rugplot:
        add_rugplot(axes[0], x[feat1].values)
        add_rugplot(axes[1], x[feat2].values)

    if histogram:
        add_histogram_under(axes[0], x[feat1].values)
        add_histogram_under(axes[1], x[feat2].values)

    return fig, axes

def plot_contribution_maps(x,
                           formula: Union[str, sp.Expr],
                           feat1: str,
                           feat2: str,
                           grid_size: int = 40,
                           mode: str = "color",
                           figsize=None,
                           colormap='bwr'
                           ) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot contribution maps for y = g11(x1) + g21(x2) + g12(x1)*g22(x2) over (feat1, feat2).

    What it shows:
      - c1(x1)  = sum of all additive terms that depend only on feat1.
      - c2(x2)  = sum of all additive terms that depend only on feat2.
      - c12(x1,x2) = sum of all *pure* interactions separable as f1(feat1)*f2(feat2),
                     and involving NO other variables.

    Visualization:
      - mode="color" (default): three filled contour maps (c1, c2, c12) using a shared 'bwr' colormap
        and one shared colorbar, so magnitudes are comparable.
      - mode="size": three scatter maps using marker size to encode |contribution| and color for sign.

    Notes:
      - Terms that involve variables beyond {feat1, feat2} are ignored.
      - Non-separable cross terms (e.g., sin(x1 + x2)) are NOT included in c12 by design.

    Parameters
    ----------
    x : pandas.DataFrame
        Must contain columns feat1 and feat2; only used to infer plotting ranges.
    formula : str or sympy.Expr
        SymPy expression or a string parseable by sympy.sympify.
    feat1, feat2 : str
        Variable names consistent with the symbols used in `formula`.
    grid_size : int
        Number of points per axis for the evaluation grid.
    mode : {"color", "size"}
        Visualization mode: "color" (contourf) or "size" (scatter sized by |value|).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of Axes
    """
    # --- Parse and identify terms ---
    expr = sp.sympify(formula) if isinstance(formula, str) else formula
    s1, s2 = sp.Symbol(feat1), sp.Symbol(feat2)
    terms = sp.Add.make_args(expr)

    terms_s1_only, terms_s2_only, terms_inter = [], [], []
    for t in terms:
        fs = t.free_symbols
        if not fs:
            # Pure constants do not contribute to the decomposition we want to show.
            continue
        if not fs.issubset({s1, s2}):
            # Drop terms that use variables beyond (s1, s2).
            continue

        if fs == {s1}:
            terms_s1_only.append(t)
        elif fs == {s2}:
            terms_s2_only.append(t)
        else:
            # Candidate interaction: must be separable f1(s1)*f2(s2) with constant coeff.
            sep = sp.separatevars(t, symbols=[s1, s2], dict=True, force=True)
            if isinstance(sep, dict) and s1 in sep and s2 in sep:
                coeff = sep.get('coeff', sp.Integer(1))
                if coeff.free_symbols.isdisjoint({s1, s2}):
                    terms_inter.append(t)

    expr_c1  = sp.Add(*terms_s1_only) if terms_s1_only else sp.Integer(0)
    expr_c2  = sp.Add(*terms_s2_only) if terms_s2_only else sp.Integer(0)
    expr_c12 = sp.Add(*terms_inter)   if terms_inter   else sp.Integer(0)

    # --- Build grid from data ranges ---
    x1_min, x1_max = float(np.nanmin(x[feat1].values)), float(np.nanmax(x[feat1].values))
    x2_min, x2_max = float(np.nanmin(x[feat2].values)), float(np.nanmax(x[feat2].values))
    if x1_min == x1_max:
        x1_min, x1_max = x1_min - 1.0, x1_max + 1.0
    if x2_min == x2_max:
        x2_min, x2_max = x2_min - 1.0, x2_max + 1.0

    x1_lin = np.linspace(x1_min, x1_max, grid_size)
    x2_lin = np.linspace(x2_min, x2_max, grid_size)
    X1, X2 = np.meshgrid(x1_lin, x2_lin, indexing='xy')

    # --- Lambdify and safe evaluation ---
    f_c1  = sp.lambdify((s1, s2), expr_c1,  modules=['numpy'])  # c1 is independent of s2; pass both for signature symmetry
    f_c2  = sp.lambdify((s1, s2), expr_c2,  modules=['numpy'])  # c2 is independent of s1
    f_c12 = sp.lambdify((s1, s2), expr_c12, modules=['numpy'])

    def _safe_eval(func, X1, X2):
        try:
            Z = np.asarray(func(X1, X2), dtype=float)
        except Exception as e:
            print(f"[plot_contribution_maps] Warning: evaluation failed ({e}). Returning zeros.")
            Z = np.zeros_like(X1, dtype=float)
        finite = np.isfinite(Z)
        if np.any(finite):
            zmin, zmax = np.min(Z[finite]), np.max(Z[finite])
        else:
            zmin, zmax = 0.0, 0.0
        Z = np.nan_to_num(Z, nan=0.0, posinf=zmax, neginf=zmin)
        return Z

    C1  = _safe_eval(f_c1,  X1, X2)  # will repeat along columns
    C2  = _safe_eval(f_c2,  X1, X2)  # will repeat along rows
    C12 = _safe_eval(f_c12, X1, X2)

    # --- Figure and plotting ---
    if figsize is None:
        fig, axes = plt.subplots(1, 3, constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    cmap = 'bwr' if colormap is None else colormap

    if mode == "color":
        # Shared symmetric scale around zero aids interpretation of signed effects
        vmax = np.max(np.abs([C1, C2, C12]))
        vmin = -vmax


        panels = [
            (C1,  f"Contribution g1({feat1})"),
            (C2,  f"Contribution g2({feat2})"),
            (C12, f"Interaction c12({feat1},{feat2}) = g11({feat1}) * g22({feat2})")
        ]
        contour_sets = []
        for i, (ax, (Z, title)) in enumerate(zip(axes, panels)):
            cs = ax.contourf(X1, X2, Z, levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel(feat1)
            if i == 0:
                ax.set_ylabel(feat2)
            else:
                # ax.set_ylabel(feat2)
                ax.set_yticks([])
            contour_sets.append(cs)
        fig.colorbar(contour_sets[-1], ax=axes, orientation='vertical', fraction=0.03, pad=0.04)

    elif mode == "size":
        # Marker size encodes |value|; color encodes sign using a diverging colormap.
        vmax = np.max(np.abs([C1, C2, C12]))
        if vmax == 0:
            vmax = 1.0
        panels = [
            (C1,  f"Contribution g1({feat1})"),
            (C2,  f"Contribution g2({feat2})"),
            (C12, f"Contribution c12({feat1},{feat2})= g11({feat1}) * g22({feat2})")
        ]
        for ax, (Z, title) in zip(axes, panels):
            # Flatten grids for scatter
            xx = X1.ravel()
            yy = X2.ravel()
            zz = Z.ravel()
            sizes = 200.0 * (np.abs(zz) / vmax) + 5.0  # ensure visible minimum size
            sc = ax.scatter(xx, yy, s=sizes, c=zz, cmap=cmap, vmin=-vmax, vmax=vmax, edgecolor='none')
            ax.set_title(title)
            ax.set_xlabel(feat1)
            # ax.set_ylabel(feat2)
            ax.set_yticks([])
        # Single shared colorbar for sign (magnitude is implicit in size)
        fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.03, pad=0.04)
    else:
        raise ValueError("mode must be 'color' or 'size'")

    # fig.suptitle(f"Contribution maps over ({feat1}, {feat2})", fontsize=13)
    return fig, axes

def plot_derivatives(x,
                     formula: Union[str, sp.Expr],
                     feat1: str,
                     feat2: str,
                     grid_size: int = 40,
                     colormap='bwr',
                     figsize=None,
                     levels=30,
                     ) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot three filled contour maps (1x3) of derivatives over (feat1, feat2):

      1) ∂y/∂x1
      2) ∂y/∂x2
      3) ∂²y/(∂x1 ∂x2)   (mixed partial)

    The formula may contain additional variables, but only terms whose free symbols
    are a subset of {feat1, feat2} are kept (others are dropped) to avoid requiring
    values for unrelated variables during numerical evaluation.

    Parameters
    ----------
    x : pandas.DataFrame
        Must contain columns feat1 and feat2; used to infer plotting ranges.
    formula : str or sympy.Expr
        SymPy expression or parsable string.
    feat1, feat2 : str
        Variable names consistent with the symbols used in `formula`.
    grid_size : int
        Number of points per axis for the evaluation grid.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of Axes
    """
    # --- Parse and restrict the expression to (feat1, feat2) terms only ---
    expr = sp.sympify(formula) if isinstance(formula, str) else formula
    s1, s2 = sp.Symbol(feat1), sp.Symbol(feat2)

    # Keep only additive terms whose free symbols ⊆ {s1, s2}
    terms = sp.Add.make_args(expr)
    kept_terms = [t for t in terms if t.free_symbols.issubset({s1, s2})]
    expr_xy = sp.Add(*kept_terms) if kept_terms else sp.Integer(0)

    # --- Symbolic derivatives ---
    d_y_dx1 = sp.diff(expr_xy, s1)
    d_y_dx2 = sp.diff(expr_xy, s2)
    d2_y_dx1dx2 = sp.diff(expr_xy, s1, s2)

    # --- Build evaluation grid from DataFrame ranges ---
    x1_min, x1_max = float(np.nanmin(x[feat1].values)), float(np.nanmax(x[feat1].values))
    x2_min, x2_max = float(np.nanmin(x[feat2].values)), float(np.nanmax(x[feat2].values))
    if x1_min == x1_max:
        x1_min, x1_max = x1_min - 1.0, x1_max + 1.0
    if x2_min == x2_max:
        x2_min, x2_max = x2_min - 1.0, x2_max + 1.0

    x1_lin = np.linspace(x1_min, x1_max, grid_size)
    x2_lin = np.linspace(x2_min, x2_max, grid_size)
    X1, X2 = np.meshgrid(x1_lin, x2_lin, indexing='xy')

    # --- Lambdify and safe evaluation ---
    f_dx1 = sp.lambdify((s1, s2), d_y_dx1, modules=['numpy'])
    f_dx2 = sp.lambdify((s1, s2), d_y_dx2, modules=['numpy'])
    f_d2  = sp.lambdify((s1, s2), d2_y_dx1dx2, modules=['numpy'])

    def _safe_eval(func, X1, X2):
        """Evaluate func(X1, X2) robustly; broadcast constants; sanitize NaN/Inf."""
        try:
            Z = func(X1, X2)
            # Broadcast scalars / 0-D / 1-D to the grid shape
            if np.isscalar(Z) or np.ndim(Z) == 0:
                Z = float(Z)
                Z = np.full(X1.shape, Z, dtype=float)
            else:
                Z = np.asarray(Z, dtype=float)
                if Z.shape != X1.shape:
                    # Attempt broadcasting to the grid shape
                    Z = np.broadcast_to(Z, X1.shape).astype(float)
        except Exception as e:
            print(f"[plot_derivatives] Warning: evaluation failed ({e}). Returning zeros.")
            Z = np.zeros_like(X1, dtype=float)

        finite = np.isfinite(Z)
        if np.any(finite):
            zmin, zmax = np.min(Z[finite]), np.max(Z[finite])
        else:
            zmin, zmax = 0.0, 0.0
        Z = np.nan_to_num(Z, nan=0.0, posinf=zmax, neginf=zmin)
        return Z

    Z_dx1 = _safe_eval(f_dx1, X1, X2)
    Z_dx2 = _safe_eval(f_dx2, X1, X2)
    Z_d2  = _safe_eval(f_d2,  X1, X2)

    # --- Shared symmetric color scale for interpretability ---
    vmax = max(
        np.nanmax(Z_dx1),
        np.nanmax(Z_dx2),
        np.nanmax(Z_d2),
    )
    vmin = min(np.nanmin(Z_dx1), np.nanmin(Z_dx2), np.nanmin(Z_d2))

    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    cmap = 'bwr' if colormap is None else colormap

    # --- Plotting ---
    if figsize is None:
        fig, axes = plt.subplots(1, 3, constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    panels = [
        (Z_dx1, r"$\partial y/\partial x_1$"),
        (Z_dx2, r"$\partial y/\partial x_2$"),
        (Z_d2,  r"$\partial^2 y/(\partial x_1 \partial x_2)$"),
    ]

    contour_sets = []
    for i, (ax, (Z, title)) in enumerate(zip(axes, panels)):
        cs = ax.contourf(X1, X2, Z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        cs.set_clim(vmin, vmax)
        ax.set_title(title)
        ax.set_xlabel(feat1)
        if i == 0:
            ax.set_ylabel(feat2)
        else:
            ax.set_yticks([])
        contour_sets.append(cs)

    for ax in axes:
        for c in ax.collections:
            c.set_clim(vmin, vmax)

    # Single shared colorbar
    # Create a normalization and colormap for the color mapping
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    sm.set_array([])  # dummy array so colorbar() works
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.03, pad=0.04)
    # cbar = fig.colorbar(contour_sets[-2], ax=axes, orientation='vertical', fraction=0.03, pad=0.04)

    return fig, axes