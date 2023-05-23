import re
import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
matplotlib.rcParams.update({'font.size': 20})
from matplotlib.ticker import MaxNLocator

# latex text rendering, from https://stackoverflow.com/a/8384685
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']


DMC = [
    'acrobot-swingup', 'reacher-hard', 'finger-turn_hard', 'hopper-hop', 'fish-swim', 'cheetah-run', 'walker-run',
    'quadruped-run', 'swimmer-swimmer15', 'humanoid-run', 'dog-run'
]

MuJoCo = ['Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4']

MetaWorld = [
    'metaworld_button-press-v2', 'metaworld_door-open-v2', 'metaworld_drawer-close-v2',
    'metaworld_drawer-open-v2', 'metaworld_peg-insert-side-v2', 'metaworld_pick-place-v2',
    'metaworld_push-v2', 'metaworld_reach-v2', 'metaworld_window-open-v2', 'metaworld_window-close-v2',
    'metaworld_basketball-v2', 'metaworld_dial-turn-v2', 'metaworld_sweep-into-v2', 'metaworld_hammer-v2',
    'metaworld_assembly-v2'
]

Box2D = ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3']

envs = {
    'DMC': DMC,
    'MuJoCo': MuJoCo,
    'MetaWorld': MetaWorld,
    'Box2D': Box2D
}

path_lists = [
    # benchmark
    '../doublegumbelqlearning-results/results/DoubleGum0_5',
    '../doublegumbelqlearning-results/results/DoubleGum0_1',
    '../doublegumbelqlearning-results/results/DoubleGum',
    '../doublegumbelqlearning-results/results/DoubleGum-0_1',
    '../doublegumbelqlearning-results/results/DoubleGum-0_5'
]

pretty_display = {
    '-0.5'         : '../doublegumbelqlearning-results/results/DoubleGum0_5',
    '-0.1'         : '../doublegumbelqlearning-results/results/DoubleGum0_1',
    '0.0 (Default)': '../doublegumbelqlearning-results/results/DoubleGum',
    '0.1'          : '../doublegumbelqlearning-results/results/DoubleGum-0_1',
    '0.5'          : '../doublegumbelqlearning-results/results/DoubleGum-0_5',
}

algorithms = list(pretty_display.keys())


def _decorate_axis(ax, wrect=10, hrect=10):
    # from https://github.com/google-research/rliable/blob/46f250777f69313f813026f9d6e1cc9d4b298e2d/rliable/plot_utils.py#L70
    """Helper function for decorating plots."""
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    # ax.tick_params(length=0.1, width=0.1)
    # ax.spines['left'].set_position(('outward', hrect))
    # ax.spines['bottom'].set_position(('outward', wrect))
    # ax['xzero'].set_axisline_style("-|>")
    # ax['yzero'].set_axisline_style("-|>")

    ax.grid(True, alpha=0.2)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # phi = (1 + np.sqrt(5)) / 2
    phi = 1.3
    ax.set_aspect(1. / (phi * ax.get_data_ratio()), adjustable='box')
    return ax


def make_legend(ax, algorithms=None):
    color_palette = sns.color_palette('colorblind', n_colors=len(algorithms))
    colors        = dict(zip(algorithms, color_palette))
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for algorithm in algorithms:
        ax.plot(0, 0, color=colors[algorithm], label=algorithm)
    # ax.legend(loc='center', ncol=len(algorithms), fontsize=30)
    leg = ax.legend(loc='center', fontsize=30)

    # from https://stackoverflow.com/a/48296983
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)


def save_fig(fig, env):
    folder    = 'plotting/plots/3adjust'
    file_name = f'{folder}/{env}.pdf'
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(file_name, format='pdf', bbox_inches='tight')
    print(file_name)



def listfiles(path, name):
    return glob.glob(f'./{path}/**/*{name}*.txt', recursive=True)


for env in envs:
    height = 1
    width  = 4
    for position, name in enumerate(envs[env]):
        x, y = int(position / width), position % width
        if y == 0:
            fig, axes = plt.subplots(height, width, figsize=(22, 4.5))

        colors = sns.color_palette('colorblind', n_colors=len(algorithms))

        shown = False
        rot_path_lists = path_lists[1:] + path_lists[:1]
        rot_colors     = colors[1:] + colors[:1]
        for path, color in zip(rot_path_lists, rot_colors):
            paths = listfiles(path, name)
            results = []
            for files in paths:
                result_file = np.loadtxt(files)
                result      = np.array(result_file)
                results.append(result.tolist())

            new_results = []
            min_length = min([len(r) for r in results])
            for r, f in zip(results, files):
                new_results.append(r)
            new_results = [r[:min_length] for r in new_results]

            results         = np.array(new_results)
            proportiontocut = 0.25
            samples         = 12
            cut             = int(proportiontocut * samples)

            results.sort(0)
            stdev = results.std(0)

            results = results[cut:-cut]

            means = results.mean(0)
            index = (np.arange(results.shape[1]) / 100).tolist()

            axes[y].plot(index, means, color=color, linewidth=3)
            axes[y].set_title(name.replace('metaworld_', ''), fontsize=30)
            phi = (np.sqrt(5) - 1) / 2
            alpha = phi / len(path_lists)
            axes[y].fill_between(index, np.array(means)-stdev, np.array(means)+stdev, alpha=alpha, color=color)
            _decorate_axis(axes[y])

        if y == 3:
            shown = True

            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            # plt.xlabel('Timesteps (in millions)', fontsize=30)
            plt.ylabel('IQM Return\n', fontsize=30)
            plt.tight_layout()
            save_fig(plt, f'{env}_{x}')
            # plt.show()

        else:
            shown = False

    if not shown:
        _decorate_axis(axes[y])
        make_legend(axes[y+1], algorithms=algorithms)
        for i in range(y+2, width):
            _decorate_axis(axes[i])
            axes[i].set_frame_on(False)
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('Timesteps (in millions)', fontsize=30)
        plt.ylabel('IQM Return\n', fontsize=30)
        plt.tight_layout()
        save_fig(plt, f'{env}_{x}')
        # plt.show()
