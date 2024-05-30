import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def seaborn_plot(df, ax, title, xticks, mse=True, ylabel=False, symlog=True):
    x_values = np.log(xticks)  # Manually set x-axis values in log scale
    
    df_long = pd.melt(df, var_name='Method', value_name='Value')
    df_long['x'] = np.tile(x_values, len(df.columns))

    markers = ['o', 's', 'D', '^', 'v', 'X']
    method_marker_dict = {method: markers[i] for i, method in enumerate(df.columns)}
    
    for method, marker in method_marker_dict.items():
        subset = df_long[df_long['Method'] == method[0]]
        sns.scatterplot(data=subset, x='x', y='Value', label=method[0], marker=marker,s=100, ax=ax, legend=False)
        sns.lineplot(data=subset, x='x', y='Value', linewidth=2, ax=ax, legend=False)

    if mse:
        ax.set_yscale('log')
        ax.set_title(title)
        if ylabel:
            ax.set_ylabel('MSE')
        else:
            ax.set_ylabel('')
    else:
        if symlog:
            ax.set_yscale('symlog')
            
        ax.set_xlabel('No. of trajectories')
        if ylabel:
            ax.set_ylabel('Bias')
        else:
            ax.set_ylabel('')
    ax.set_xticklabels(xticks)
    ax.set_xticks(x_values)
    ax.grid(True)

def plot_helper(df, title, xticks=None, xlabel="x", ylabel="y"):
    plt.figure(figsize=(6, 4))
    if xticks is None:
        xticks = np.arange(df.shape[0])
        set_xtick = False
    else:
        set_xtick = True
    for column in df.columns:
        #plt.scatter(xticks, df[column], label=column[0])
        plt.plot(xticks, df[column], linestyle='-', label=column[0])
    if set_xtick:
        plt.xticks(xticks)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.legend()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()