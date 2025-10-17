# import pytest
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from utils.utils_representation import (plot_pdp_delta, plot_prp, plot_ice, plot_interact_contour, plot_pdp_additive,
                                        plot_contribution_maps, plot_derivatives)
'''
def test_plot_pdp_prp_ice():
    # Create synthetic dataframe
    np.random.seed(42)
    n = 20
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.uniform(-2, 2, n)
    x3 = np.random.normal(2, 0.5, n)
    x4 = np.random.binomial(1, 0.3, n)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4})

    # Create sympy formula: 2*x1 + sin(x2) + 0.5*x3**2
    x1_sym, x2_sym, x3_sym, x4_sym = sp.symbols('x1 x2 x3 x4')
    formula = 2*x1_sym + sp.sin(x2_sym) + 0.5*x3_sym**2 + x4_sym # include interaction with binary x4

    y = 2*x1 + np.sin(x2) + 0.5*x3**2 + x4 +np.random.normal(0, 0.1, n)
    # Test plot_prp (use first 3 ids)
    ids = [0, 1, 2]
    colors = ['r', 'g', 'b', 'c']
    # Test plot_pdp
    fig_pdp, axes_pdp = plot_pdp_delta(df, formula, ids=ids, cols=['x1', 'x2', 'x3', 'x4'], colors = colors)
    # fig_pdp.savefig('test_plot_pdp.png')
    plt.show()
    plt.close(fig_pdp)

    colors = ['r', 'g', 'b']
    fig_prp, ax_prp = plot_prp(df, formula, ids=ids, cols=['x1', 'x2', 'x3', 'x4'], colors=colors)
    plt.show()
    # fig_prp.savefig('test_plot_prp.png')
    plt.close(fig_prp)

    # Test plot_ice
    colors = ['r', 'g', 'b']
    fig_ice, axes_ice = plot_ice(df, formula, ids=ids, cols=['x1', 'x2', 'x3', 'x4'], colors=colors)
    plt.show()
    plt.close(fig_ice)

    # Test plot_ice
    colors = ['r', 'g', 'b']
    fig_ice, axes_ice = plot_ice(df, formula, ids=ids, cols=['x2'], colors=colors)
    plt.show()
    plt.close(fig_ice)

    # Test plot_ice
    colors = ['r', 'g', 'b']
    fig_ice, axes_ice = plot_ice(df, formula, ids=ids, centered=True, cols=['x2'], colors=colors)
    plt.show()
    plt.close(fig_ice)

    # Test plot_ice
    fig_ice, axes_ice = plot_ice(df, formula, cols=['x2'])
    plt.show()
    plt.close(fig_ice)

    # Test plot_ice
    fig_ice, axes_ice = plot_ice(df, formula, cols=['x2'], centered=True)
    plt.show()
    plt.close(fig_ice)

    # Test plot_ice
    fig_ice, axes_ice = plot_ice(df, formula, cols=['x2'], hue = 'x4')
    plt.show()
    plt.close(fig_ice)

    # Test plot_ice
    fig_ice, axes_ice = plot_ice(df, formula, cols=['x2'], hue = 'x4', centered=True)
    plt.show()
    plt.close(fig_ice)

    # Test plot_ice
    fig_ice, axes_ice = plot_ice(df, formula, cols=['x3'], hue = 'x2')
    plt.show()
    plt.close(fig_ice)
'''
def test_plot_interact():
    # create a dataframe with 2 features
    # in which there are multiplications
    np.random.seed(42)

    n = 30
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.uniform(-2, 2, n)
    df = pd.DataFrame({'x1': x1, 'x2': x2})
    x1_sym, x2_sym = sp.symbols('x1 x2')
    formula = 2*x1_sym * x2_sym**3 + x1_sym**2 - x2_sym**3 + x2_sym/2

    plot_interact_contour(df, formula, 'x1', 'x2', grid_size=30, figsize=(12, 4), levels=100)
    plt.show()
    plt.close()

    plot_pdp_additive(df, formula, 'x1', 'x2')
    plt.show()
    plt.close()

    plot_contribution_maps(df, formula, 'x1', 'x2', grid_size=30, figsize=(12, 4))
    plt.show()
    plt.close()

    plot_derivatives(df, formula, 'x1', 'x2', grid_size=30, figsize=(12, 4), colormap='PiYG', levels=100)
    plt.show()
    plt.close()

if __name__ == "__main__":
    # test_plot_pdp_prp_ice()
    test_plot_interact()
    print("plot_pdp and plot_prp ran successfully and figures were saved.")

