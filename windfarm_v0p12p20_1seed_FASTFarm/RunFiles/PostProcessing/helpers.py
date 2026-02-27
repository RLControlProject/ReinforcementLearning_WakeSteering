import numpy as np
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def generate_layout_grid(num_points_x, num_points_y, spacing_x, spacing_y):
    """
    Generate a grid of layout points with arbitrary numbers of points and spacing in each direction.

    Args:
        num_points_x (int): Number of points in the x direction.
        num_points_y (int): Number of points in the y direction.
        spacing_x (float): Spacing between points in the x direction.
        spacing_y (float): Spacing between points in the y direction.

    Returns:
        list of tuples: List of layout points represented as (x, y) coordinates.
    """
    layout_grid = []
    for i in range(num_points_x):
        for j in range(num_points_y):
            x = i * spacing_x
            y = j * spacing_y
            layout_grid.append((x, y))
    return layout_grid

def create_net_arch(neurons_per_layer, num_layers):
    """
    Create a neural network architecture with the specified number of layers and neurons per layer.

    Args:
        neurons_per_layer (int): Number of neurons per layer.
        num_layers (int): Number of layers in the neural network.

    Returns:
        list: List representing the neural network architecture.
    """
    return [neurons_per_layer] * num_layers

def generate_layout_grid(num_points_x, num_points_y, spacing_x, spacing_y):
    """
    Generate a grid of layout points with arbitrary numbers of points and spacing in each direction.

    Args:
        num_points_x (int): Number of points in the x direction.
        num_points_y (int): Number of points in the y direction.
        spacing_x (float): Spacing between points in the x direction.
        spacing_y (float): Spacing between points in the y direction.

    Returns:
        list of tuples: List of layout points represented as (x, y) coordinates.
    """
    layout_grid = []
    for i in range(num_points_x):
        for j in range(num_points_y):
            x = i * spacing_x
            y = j * spacing_y
            layout_grid.append((x, y))
    return layout_grid
    
# Shows single plot
def saveplotsingle(fig, axis, filepath, ls=1.5, fs=12): #20
    thisfont = 'DejaVu Sans'
    plt.rcParams.update({'font.size': fs})
    #### Get handles and print plot ####
    handles, labels = axis.get_legend_handles_labels()
    #fig.legend(handles, labels, bbox_to_anchor=(ls, 0.8),frameon=False,prop={'size': fs})
    #axis.grid(b=True, which='major', color='#999999', linestyle='-', alpha=0.5)
    #axis.minorticks_on()
    #axis.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
    #for tick in axis.get_xticklabels():
    #    tick.set_fontname(thisfont)
    #for tick in axis.get_yticklabels():
    #    tick.set_fontname(thisfont)
    fig.tight_layout()
    print('Saving: ',filepath)
    plt.savefig(filepath, bbox_inches='tight')

# Shows single plot
def saveplotsingle_noaxis(fig, filepath, ls=1.5, fs=12): #20
    thisfont = 'DejaVu Sans'
    # plt.rcParams.update({'font.size': fs})
    #### Get handles and print plot ####
    # handles, labels = axis.get_legend_handles_labels()
    #fig.legend(handles, labels, bbox_to_anchor=(ls, 0.8),frameon=False,prop={'size': fs})
    #axis.grid(b=True, which='major', color='#999999', linestyle='-', alpha=0.5)
    #axis.minorticks_on()
    #axis.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
    #for tick in axis.get_xticklabels():
    #    tick.set_fontname(thisfont)
    #for tick in axis.get_yticklabels():
    #    tick.set_fontname(thisfont)
    # fig.tight_layout()
    print('Saving: ',filepath)
    fig.savefig(filepath, bbox_inches='tight')
