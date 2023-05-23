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


path_lists = [
    f'../doublegumbelqlearning-results/logging',
]


def listfiles(path, name):
    return glob.glob(f'./{path}/**/*{name}*/*.npy', recursive=True)



envs = [
    'quadruped-run',
]


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

    phi = (1 + np.sqrt(5)) / 2
    ax.set_aspect(1. / (phi * ax.get_data_ratio()), adjustable='box')
    return ax


def make_legend(ax, algorithms=None):
    color_palette = sns.color_palette('colorblind', n_colors=len(algorithms))
    colors        = dict(zip(algorithms, color_palette))
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for algorithm in algorithms:
        plt.plot(0, 0, color=colors[algorithm], label=algorithm)
    # ax.legend(loc='center', ncol=len(algorithms), fontsize=30)
    ax.legend(loc='center', fontsize=30)
    return ax


def save_fig(fig, name):
    folder    = 'plotting/plots/'
    file_name = f'{folder}/{name}.pdf'
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(file_name, format='pdf', bbox_inches='tight')
    print(file_name)


for env in envs:
    height, width = 1, 3
    fig, axes = plt.subplots(height, width, figsize=(22, 6.5))

    for pl in path_lists:
        # from https://stackoverflow.com/a/33159707
        files = listfiles(pl, env)
        files = listfiles(os.path.dirname(files[0]), '')
        files.sort(key=lambda f: int(re.sub('\D', '', f)))

        f = files[-1]
        colors = sns.color_palette('colorblind', n_colors=4)

        aux        = np.load(f, allow_pickle=True)
        target     = aux.item().get('target_Q').flatten()
        mean       = aux.item().get('online_Q').flatten()
        std        = aux.item().get('online_std').flatten()


        def plot_histogram(data, index):
            emp_mean            = data.mean()
            emp_std             = data.std()
            emp_logistic_spread = emp_std * np.sqrt(3) / np.pi
            emp_spread          = emp_std * np.sqrt(6) / np.pi
            emp_loc             = emp_mean + np.euler_gamma * emp_spread

            spread = 3
            low    = emp_mean - spread*emp_std
            high   = emp_mean + spread*emp_std
            axes[index].set_xlim(low, high)

            bins = int((data.max() - data.min()) / (high - low) * 100)
            bins = axes[index].hist(data, bins, density=True, color='silver', histtype='barstacked')[1]

            normal_pdf = 1 / (emp_std * np.sqrt(2 * np.pi)) * np.exp(-.5 * ((bins - emp_mean) / emp_std) ** 2)
            axes[index].plot(bins, normal_pdf, color=colors[1], linewidth=3)

            z = (bins - emp_loc) / emp_spread
            gumbel_pdf = (1 / emp_spread) * np.exp(-z - np.exp(-z))
            axes[index].plot(bins, gumbel_pdf, color=colors[2], linewidth=3)

            z            = (bins - emp_mean) / emp_logistic_spread
            logistic_pdf = np.exp(- z) / (emp_logistic_spread * (1 + np.exp(-z)) ** 2)
            axes[index].plot(bins, logistic_pdf, color=colors[0], linewidth=3)
            _decorate_axis(axes[index])


        def plot_nll(ax):
            files = listfiles(pl, env)
            set_of_seeds = list((set([os.path.dirname(f) for f in files])))

            hetero_data_array = []
            homo_data_array = []

            for file in set_of_seeds:
                files = listfiles(file, '')
                files.sort(key=lambda f: int(re.sub('\D', '', f)))
                files = files[1:]

                hetero_data = []
                homo_data = []
                for f in files:
                    aux = np.load(f, allow_pickle=True)
                    # print(aux.item().keys()) # 'online_Q', 'online_std', 'target_Q', 'target_std'

                    target = aux.item().get('target_Q').flatten()
                    mean = aux.item().get('online_Q').flatten()
                    std = aux.item().get('online_std').flatten()

                    # target   = aux.item().get('target').flatten()
                    # loc      = aux.item().get('online_loc').flatten()
                    # spread   = aux.item().get('online_spread').flatten()
                    # mean     = aux.item().get('online_mean').flatten()
                    # std      = aux.item().get('online_std').flatten()

                    # hetero_datapoint = (target - loc) / spread
                    # hetero_datapoint = (target - mean) / std

                    hetero_datapoint = (target - mean) / std
                    homo_datapoint = target - mean

                    hetero_data.append(hetero_datapoint)
                    homo_data.append(homo_datapoint)

                hetero_data_array.append(hetero_data)
                homo_data_array.append(homo_data)

            # print(np.array(hetero_data_array).shape)
            # print(np.array(homo_data_array).shape)

            hetero_data = np.array(hetero_data_array)
            homo_data = np.array(homo_data_array)

            mean = hetero_data.mean()
            std = hetero_data.std()
            spread = std * np.sqrt(6) / np.pi
            loc = mean - np.euler_gamma * spread
            logistic_spread = std * np.sqrt(3) / np.pi

            z = (hetero_data - loc) / spread
            gumbel_nll = np.mean(np.log(spread) + z + np.exp(-z), axis=-1)

            z = (hetero_data - mean) / logistic_spread
            logistic_nll = np.mean(z + np.log(logistic_spread * (1 + np.exp(-z)) ** 2), axis=-1)

            normal_nll = np.mean(np.log(std * np.sqrt(2 * np.pi)) + .5 * ((hetero_data - mean) / std) ** 2, axis=-1)

            homo_nll = np.mean(np.log(std * np.sqrt(2 * np.pi)) + .5 * ((homo_data - mean) / std) ** 2, axis=-1)

            gumbel_nll_mean = gumbel_nll.mean(0).squeeze()
            gumbel_nll_std = gumbel_nll.std(0).squeeze()
            logistic_nll_mean = logistic_nll.mean(0).squeeze()
            logistic_nll_std = logistic_nll.std(0).squeeze()
            normal_nll_mean = normal_nll.mean(0).squeeze()
            normal_nll_std = normal_nll.std(0).squeeze()
            homo_nll_mean = homo_nll.mean(0).squeeze()
            homo_nll_std = homo_nll.std(0).squeeze()

            lower_lnnl = logistic_nll_mean - logistic_nll_std
            upper_lnnl = logistic_nll_mean + logistic_nll_std

            lower_nnnl = normal_nll_mean - normal_nll_std
            upper_nnnl = normal_nll_mean + normal_nll_std

            index = ((1 + np.arange(99)) / 100).tolist()

            phi = (np.sqrt(5) - 1) / 2
            alpha = phi / 2

            ax.plot(index, normal_nll_mean, color=colors[1], linewidth=3)
            ax.fill_between(index, y1=lower_nnnl, y2=upper_nnnl, color=colors[1], alpha=alpha)

            ax.plot(index, logistic_nll_mean, color=colors[0], linewidth=3)
            ax.fill_between(index, y1=lower_lnnl, y2=upper_lnnl, color=colors[0], alpha=alpha)

            # ax.plot(index, gumbel_nll_mean, color=colors[3], linewidth=2)
            # ax.fill_between(index, y1=lower_nnnl, y2=upper_nnnl, color=colors[3], alpha=0.2)

            _decorate_axis(ax)


        index = 0
        plot_histogram(target - mean, index)
        axes[index].set_xlabel('TD Error', fontsize=30)
        axes[index].set_ylabel('Frequency Density', fontsize=30)
        # from https://stackoverflow.com/a/18346779
        axes[index].text(0.05, 1.06, 'a', transform=axes[index].transAxes, fontsize=40, va='top')

        index = 1
        plot_histogram((target - mean) / std, index)
        axes[index].set_xlabel('Standardized TD Error', fontsize=30)
        axes[index].set_ylabel('Frequency Density', fontsize=30)
        axes[index].text(0.05, 1.06, 'b', transform=axes[index].transAxes, fontsize=40, va='top')

        index = 2
        plot_nll(axes[index])
        axes[index].set_xlabel('Timesteps (in millions)', fontsize=30)
        axes[index].set_ylabel('Negative Log-Likelihood', fontsize=30)
        axes[index].text(0.05, 1.06, 'c', transform=axes[index].transAxes, fontsize=40, va='top')


# From https://stackoverflow.com/a/24229589
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.plot(0, 0, color='silver' , label='Empirical (Histogram)')
plt.plot(0, 0, color=colors[0], label=r'\textbf{Logistic (expected)}')
plt.plot(0, 0, color=colors[1], label='Normal')
plt.plot(0, 0, color=colors[2], label='Gumbel')
leg = plt.legend(loc='upper center', ncol=4, fontsize=30, bbox_to_anchor=(0.5, -.25))

# from https://stackoverflow.com/a/48296983
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)


plt.tight_layout()
save_fig(plt, 'histograms')
plt.show()
