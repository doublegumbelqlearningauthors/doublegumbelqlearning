"""
Lovingly pastiched from https://colab.research.google.com/drive/1a0pSD-1tWhMmeJeeoyZM1A-HCW3yf1xR
"""
import matplotlib.pyplot as plt
import numpy as np
import glob
from rliable import library as rly
from rliable import metrics
import seaborn as sns

### here begins my code

from matplotlib.ticker import MaxNLocator
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

# latex text rendering, from https://stackoverflow.com/a/8384685
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']

from lower_bounds import lower_bounds


domains_to_envs = {
    'DMC'      : ['acrobot-swingup', 'reacher-hard', 'finger-turn_hard', 'hopper-hop', 'fish-swim', 'cheetah-run', 'walker-run', 'quadruped-run', 'swimmer-swimmer15', 'humanoid-run', 'dog-run'],
    'MuJoCo'   : ['Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4'],
    'MetaWorld': ['metaworld_button-press-v2', 'metaworld_door-open-v2', 'metaworld_drawer-close-v2', 'metaworld_drawer-open-v2', 'metaworld_peg-insert-side-v2', 'metaworld_pick-place-v2', 'metaworld_push-v2', 'metaworld_reach-v2', 'metaworld_window-open-v2', 'metaworld_window-close-v2', 'metaworld_basketball-v2', 'metaworld_dial-turn-v2', 'metaworld_sweep-into-v2', 'metaworld_hammer-v2', 'metaworld_assembly-v2'],
    'Box2D'    : ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3']
}



jaxrl_path_lists = []

path_lists = [
    # benchmark
    '../doublegumbelqlearning-results/continuous/DoubleGum',
    '../doublegumbelqlearning-results/continuous/DDPG',
    '../doublegumbelqlearning-results/continuous/TD3',
    '../doublegumbelqlearning-results/continuous/MoG-DDPG'
]

pretty_display = {
    r'\textbf{DoubleGum, $c=0$ (Ours)}': '../doublegumbelqlearning-results/continuous/DoubleGum',
    'DDPG'                             : '../doublegumbelqlearning-results/continuous/DDPG',
    'TD3'                              : '../doublegumbelqlearning-results/continuous/TD3',
    'MoG-DDPG'                         : '../doublegumbelqlearning-results/continuous/MoG-DDPG',
}

algorithms = list(pretty_display.keys())


def jaxrl_listfiles(path, name):
    return glob.glob(f'./{path}/**/{name}/**/*.txt', recursive=True)

def listfiles(path, name):
    return glob.glob(f'./{path}/**/*{name}*.txt', recursive=True)

def load_path_name(path, name):
    if path in jaxrl_path_lists:
        paths = jaxrl_listfiles(path, name)
    else:
        paths = listfiles(path, name)
    results = []
    for files in paths:
        result_file = np.loadtxt(files)
        result      = np.array(result_file)
        if path in jaxrl_path_lists:
            result = result.T[1]
        results.append(result.tolist())

    try:
        new_results = []
        min_length = min([len(r) for r in results])
        for r, f in zip(results, files):
            new_results.append(r)
        new_results = [r[:min_length] for r in new_results]
        results     = np.array(new_results)

    except:
        print(path, name)

    return results



baseline_path_lists = [
    '../doublegumbelqlearning-results/continuous/DDPG',
    '../doublegumbelqlearning-results/continuous/TD3',
]


def dmc_max_score(env):
    return 1000.


def baseline_max_score(env):
    # read all baseline files and select maximum value at last timestep
    maxes = []
    for bpl in baseline_path_lists:
        results = load_path_name(bpl, env)
        maxes.append(np.amax(results))
    return max(maxes)


def metaworld_max_score(env):
    return 10000.


def robosuite_max_score(env):
    return 500.


def box2d_max_score(env):
    # https://github.com/openai/gym/blob/dcd185843a62953e27c2d54dc8c2d647d604b635/gym/envs/box2d/bipedal_walker.py#L109-L110
    if env == 'BipedalWalker-v3':
        return 300.
    elif env == 'BipedalWalkerHardcore-v3':
        return 300.
    return 0.


domains_to_norm_scores = {
    'DMC'      : dmc_max_score,
    'MuJoCo'   : baseline_max_score,
    'MetaWorld': metaworld_max_score,
    'RoboSuite': robosuite_max_score,
    'Box2D'    : box2d_max_score
}


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
    ax.grid(True, alpha=0.2)

    # https://stackoverflow.com/a/34919251
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # phi = (1 + np.sqrt(5)) / 2
    phi = 1.3
    ax.set_aspect(1. / (phi * ax.get_data_ratio()), adjustable='box')
    return ax


def plot_sample_efficiency(ax, frames, point_estimates, interval_estimates, algorithms=None):
    color_palette = sns.color_palette('colorblind', n_colors=len(algorithms))
    colors        = dict(zip(algorithms, color_palette))
    # https://stackoverflow.com/a/29498853
    for algorithm in algorithms[1:] + algorithms[:1]:
        metric_values = point_estimates[algorithm]
        lower, upper  = interval_estimates[algorithm]
        ax.plot(
            frames,
            metric_values,
            color=colors[algorithm],
            # marker=kwargs.pop('marker', 'o'),
            linewidth=2,
            label=algorithm
        )
        phi = (np.sqrt(5) - 1) / 2
        alpha = phi / len(path_lists)
        ax.fill_between(frames, y1=lower, y2=upper, color=colors[algorithm], alpha=alpha)
    return _decorate_axis(ax)


def make_legend(ax, algorithms=None):
    color_palette = sns.color_palette('colorblind', n_colors=len(algorithms))
    colors        = dict(zip(algorithms, color_palette))
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for algorithm in algorithms:
        plt.plot(0, 0, color=colors[algorithm], label=algorithm)
    ax.legend(loc='center', ncol=len(algorithms))
    return ax


def save_fig(fig, name):
    file_name = '{}.pdf'.format(name)
    fig.savefig(file_name, format='pdf', bbox_inches='tight')
    print(file_name)


width = 4
height = 1
# fig, ax = plt.subplots(figsize=(7, 4.5))
# fig, axes = plt.subplots(height, width, figsize=(22, 3.5))
# fig, axes = plt.subplots(height, width, figsize=(22, 4.5))
fig, axes = plt.subplots(height, width, figsize=(22, 6))

for position, domain in enumerate(domains_to_envs):
    score_dict = {}
    for path in jaxrl_path_lists + path_lists:
        aggregate_results = []
        for env in domains_to_envs[domain]:
            max_score = domains_to_norm_scores[domain](env)
            min_score = lower_bounds[env]
            results   = load_path_name(path, env)
            results   = (results - min_score)  / (max_score - min_score)
            # print([len(r) for r in results], env)
            aggregate_results.append(results)

        shortest                  = min([min(len(a) for a in ar) for ar in aggregate_results])
        # print(shortest)
        cleaned_aggregate_results = []
        for ar in aggregate_results:
            seed_results = []
            for a in ar:
                seed_results.append(a[:shortest])
            cleaned_aggregate_results.append(seed_results)

        aggregate_results = np.array(cleaned_aggregate_results)
        # print(domain)
        # print(path)
        aggregate_results = aggregate_results.swapaxes(0, 1)
        # aggregate_results = aggregate_results[:, :, -1]

        score_dict[path] = aggregate_results



    # def aggregate(score_dict):
    #     """
    #     :param score_dict: key -> array_shape (seeds, tasks)
    #     :return: aggregate_scores, aggregate_interval_estimates
    #     """
    #     IQM    = lambda x: metrics.aggregate_iqm(x)
    #     OG     = lambda x: metrics.aggregate_optimality_gap(x, 1.0)
    #     MEAN   = lambda x: metrics.aggregate_mean(x)
    #     MEDIAN = lambda x: metrics.aggregate_median(x)
    #
    #     aggregate_func = lambda x: np.array([MEDIAN(x), IQM(x), MEAN(x), OG(x)])
    #     return rly.get_interval_estimates(score_dict, aggregate_func, reps=50000)
    #
    #
    # aggregate_scores, aggregate_interval_estimates = aggregate(score_dict)
    #
    # colors = zip(aggregate_scores.keys(), plt.rcParams["axes.prop_cycle"].by_key()["color"])
    # fig, axes = plot_utils.plot_interval_estimates(
    #     aggregate_scores,
    #     aggregate_interval_estimates,
    #     metric_names = ['Median', 'IQM', 'Mean', 'Optimality Gap'],
    #     algorithms=path_lists,
    #     xlabel_y_coordinate=-0.16,
    #     xlabel='Human Normalized Score')
    # plt.show()
    # save_fig(fig, 'mujoco')

    for pd in pretty_display:
        score_dict[pd] = score_dict.pop(pretty_display[pd])


    frames = np.arange(100)
    iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm, reps=2000)
    # iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm, reps=20)

    x, y = int(position / width), position % width

    plot_sample_efficiency(axes[y], frames/100, iqm_scores, iqm_cis, algorithms=algorithms)
    axes[y].set_title(domain, fontsize=30)


# From https://stackoverflow.com/a/53172335
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Timesteps (in millions)', fontsize=30)
plt.ylabel('IQM Normalized Score', fontsize=30)

# From https://stackoverflow.com/a/24229589
color_palette = sns.color_palette('colorblind', n_colors=len(algorithms))
colors = dict(zip(algorithms, color_palette))
for algorithm in algorithms:
    plt.plot(0, 0, color=colors[algorithm], label=algorithm)
leg = plt.legend(loc='upper center', ncol=len(algorithms), fontsize=30, bbox_to_anchor=(0.5, -0.25))

# from https://stackoverflow.com/a/48296983
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)

plt.tight_layout()
save_fig(plt, 'per_suite_IQM')
plt.show()
