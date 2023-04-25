from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import sys
from modt.utils import compute_hypervolume, compute_sparsity, check_dominated, undominated_indices
from copy import deepcopy



def visualize(rollout_logs, logsdir, cur_step):

    n_obj = rollout_logs['n_obj']
    dataset_min_prefs = rollout_logs['dataset_min_prefs']
    dataset_max_prefs = rollout_logs['dataset_max_prefs']
    dataset_min_raw_r = rollout_logs['dataset_min_raw_r']
    dataset_max_raw_r = rollout_logs['dataset_max_raw_r']
    dataset_min_final_r = rollout_logs['dataset_min_final_r']
    dataset_max_final_r = rollout_logs['dataset_max_final_r']
    target_returns = rollout_logs['target_returns']
    target_prefs = rollout_logs['target_prefs']
    rollout_unweighted_raw_r = rollout_logs['rollout_unweighted_raw_r']
    rollout_weighted_raw_r = rollout_logs['rollout_weighted_raw_r']
    rollout_original_raw_r = rollout_logs['rollout_original_raw_r']
    
    


    
    indices_wanted = undominated_indices(rollout_unweighted_raw_r, tolerance=0.05)
    n_points = len(indices_wanted)
    edge_colors = ['royalblue' if i in indices_wanted else 'r' for i in range(rollout_unweighted_raw_r.shape[0])]
    face_colors = ['none' for i in range(rollout_unweighted_raw_r.shape[0])]


    hv = compute_hypervolume(rollout_original_raw_r) # this automatically ignores the dominated points
    indices_wanted_strict = undominated_indices(rollout_original_raw_r, tolerance=0)
    front_return_batch = rollout_original_raw_r[indices_wanted_strict]
    sparsity = compute_sparsity(front_return_batch)
    


    fig, axes = plt.subplots(n_obj, 3, constrained_layout=True, figsize=(12, 8))
    axes = axes.flatten()
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    sns.despine()
    cur_ax = 0

    # obj0 vs obj1, unweighted
    if n_obj == 2:
        axes[cur_ax].scatter(
            rollout_original_raw_r[:, 0],
            rollout_original_raw_r[:, 1],
            label=f"hv: {hv:.3e}\npts: {n_points}\nsp: {np.round(sparsity, 2)}",
            facecolors=face_colors,
            edgecolors=edge_colors,
        )
        axes[cur_ax].set_xlim([0, max(rollout_original_raw_r[:, 0]) * 1.05])
        axes[cur_ax].set_ylim([0, max(rollout_original_raw_r[:, 1]) * 1.05])
        axes[cur_ax].set_title(f'Obj 0 vs Obj 1')
        axes[cur_ax].set(xlabel="Obj 0", ylabel="Obj 1")
        axes[cur_ax].legend(loc='center left')
        cur_ax += 1
    # change to 3d pareto front     
    elif n_obj == 3:
        axes[cur_ax].remove()
        axes[cur_ax] = fig.add_subplot(n_obj, 3, cur_ax+1, projection="3d")
        axes[cur_ax].scatter(
            rollout_original_raw_r[:, 0],
            rollout_original_raw_r[:, 1],
            rollout_original_raw_r[: ,2],
            label=f"hv: {hv:.3e}\npts: {n_points}\nsp: {np.round(sparsity, 2)}",
            facecolors=face_colors,
            edgecolors=edge_colors,
        )
        axes[cur_ax].set_xlim3d([0, max(rollout_original_raw_r[:, 0]) * 1.05])
        axes[cur_ax].set_ylim3d([0, max(rollout_original_raw_r[:, 1]) * 1.05])
        axes[cur_ax].set_zlim3d([0, max(rollout_original_raw_r[:, 2]) * 1.05])
        axes[cur_ax].set_title(f'Obj 1 vs. Obj 2 vs. Obj 3')
        axes[cur_ax].set(xlabel="Obj 1", ylabel="Obj 2", zlabel="Obj 3")
        axes[cur_ax].legend(loc='lower center')
        cur_ax += 1


    rollout_ratio = rollout_original_raw_r / np.sum(rollout_original_raw_r, axis=1, keepdims=True)
    axes[cur_ax].scatter(
        target_prefs[:, 0],
        rollout_ratio[:, 0],
        label="MODT",
        facecolors=face_colors,
        edgecolors=edge_colors,
    )
    axes[cur_ax].axvline(
        x = dataset_min_prefs[0],
        ls="--",
    )
    axes[cur_ax].axvline(
        x = dataset_max_prefs[0],
        ls="--",
    )
    axes[cur_ax].set_xlim([-0.05, 1.05])
    axes[cur_ax].set_ylim([-0.05, 1.05])
    axes[cur_ax].set_title(f'Preference 0: Target vs. Achieved')
    axes[cur_ax].set(xlabel="target pref0", ylabel="achieved pref0")
    lims = [
        np.min([axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]),  # min of both axes
        np.max([axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]),  # max of both axes
    ]
    axes[cur_ax].plot(lims, lims, label="oracle", alpha=0.75, zorder=0)
    axes[cur_ax].legend(loc='upper left')
    cur_ax += 1

    
    axes[cur_ax].scatter(
        target_prefs[:, 1],
        rollout_ratio[:, 1],
        label="MODT",
        facecolors=face_colors,
        edgecolors=edge_colors,
    )
    axes[cur_ax].axvline(
        x = dataset_min_prefs[1],
        ls="--",
    )
    axes[cur_ax].axvline(
        x = dataset_max_prefs[1],
        ls="--",
    )
    axes[cur_ax].set_xlim([-0.05, 1.05])
    axes[cur_ax].set_ylim([-0.05, 1.05])
    axes[cur_ax].set_title(f'Preference 1: Target vs. Achieved')
    axes[cur_ax].set(xlabel="target pref1", ylabel="achieved pref1")
    lims = [
        np.min([axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]),  # min of both axes
        np.max([axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]),  # max of both axes
    ]
    axes[cur_ax].plot(lims, lims, label="oracle", alpha=0.75, zorder=0)
    axes[cur_ax].legend(loc='upper left')
    cur_ax += 1

    # need 1 more graph
    if n_obj == 3:
        axes[cur_ax].scatter(
            target_prefs[:, 2],
            rollout_ratio[:, 2],
            label="MODT",
            facecolors=face_colors,
            edgecolors=edge_colors,
        )
        axes[cur_ax].axvline(
            x = dataset_min_prefs[2],
            ls="--",
        )
        axes[cur_ax].axvline(
            x = dataset_max_prefs[2],
            ls="--",
        )
        axes[cur_ax].set_xlim([-0.05, 1.05])
        axes[cur_ax].set_ylim([-0.05, 1.05])
        axes[cur_ax].set_title(f'Preference 2: Target vs. Achieved')
        axes[cur_ax].set(xlabel="target pref2", ylabel="achieved pref2")
        lims = [
            np.min([axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]),  # min of both axes
            np.max([axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]),  # max of both axes
        ]
        axes[cur_ax].plot(lims, lims, label="oracle", alpha=0.75, zorder=0)
        axes[cur_ax].legend(loc='upper left')
        cur_ax += 1

    # rtg0 vs return0, rtg1 vs return1, ... (all are weighted)

    using_mo_rtg = False if len(target_returns.shape) == 1 else True
    if using_mo_rtg:
        axes[cur_ax].scatter(
            target_returns[:, 0],
            rollout_weighted_raw_r[:, 0],
            facecolors=face_colors,
            edgecolors=edge_colors,
            label="MODT"
        )
        axes[cur_ax].set_xlim([-5, np.max(target_returns[:, 0]) * 1.05])
        axes[cur_ax].set_ylim([-5, np.max(rollout_weighted_raw_r[:, 0]) * 1.05])
        axes[cur_ax].set(xlabel="target obj0", ylabel="achieved obj0")
        lims = [
            np.min([axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]),  # min of both axes
            np.max([axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]),  # max of both axes
        ]
        axes[cur_ax].plot(lims, lims, label="oracle", alpha=0.75, zorder=0)
        axes[cur_ax].axvline(
            x = dataset_min_raw_r[0],
            ls="--",
        )
        axes[cur_ax].axvline(
            x = dataset_max_raw_r[0],
            ls="--",
        )
        axes[cur_ax].legend(loc='upper left')
        axes[cur_ax].set_title(f'Weighted Obj 0: Target vs. Achieved')
        cur_ax += 1

        axes[cur_ax].scatter(
            target_returns[:, 1],
            rollout_weighted_raw_r[:, 1],
            facecolors=face_colors,
            edgecolors=edge_colors,
            label="MODT"
        )
        axes[cur_ax].set_xlim([-5, np.max(target_returns[:, 1]) * 1.05])
        axes[cur_ax].set_ylim([-5, np.max(rollout_weighted_raw_r[:, 1]) * 1.05])
        axes[cur_ax].set(xlabel="target obj1", ylabel="achieved obj1")
        axes[cur_ax].legend(loc='upper left')
        lims = [
            np.min([axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]),  # min of both axes
            np.max([axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]),  # max of both axes
        ]
        axes[cur_ax].plot(lims, lims, label="oracle", alpha=0.75, zorder=0)
        axes[cur_ax].axvline(
            x = dataset_min_raw_r[1],
            ls="--",
        )
        axes[cur_ax].axvline(
            x = dataset_max_raw_r[1],
            ls="--",
        )
        axes[cur_ax].legend(loc='upper left')
        axes[cur_ax].set_title(f'Weighted Obj 1: Target vs. Achieved')
        cur_ax += 1

        if n_obj == 3:
            axes[cur_ax].scatter(
                target_returns[:, 2],
                rollout_weighted_raw_r[:, 2],
                facecolors=face_colors,
                edgecolors=edge_colors,
                label="MODT"
            )
            axes[cur_ax].set_xlim([-5, np.max(target_returns[:, 2]) * 1.05])
            axes[cur_ax].set_ylim([-5, np.max(rollout_weighted_raw_r[:, 2]) * 1.05])
            axes[cur_ax].set(xlabel="target obj2", ylabel="achieved obj2")
            axes[cur_ax].legend(loc='upper left')
            lims = [
                np.min([axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]),  # min of both axes
                np.max([axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]),  # max of both axes
            ]
            axes[cur_ax].plot(lims, lims, label="oracle", alpha=0.75, zorder=0)
            axes[cur_ax].axvline(
                x = dataset_min_raw_r[2],
                ls="--",
            )
            axes[cur_ax].axvline(
                x = dataset_max_raw_r[2],
                ls="--",
            )
            axes[cur_ax].legend(loc='upper left')
            axes[cur_ax].set_title(f'Weighted Obj 2: Target vs. Achieved')
            cur_ax += 1
    else:

        rollout_final_r = np.sum(rollout_weighted_raw_r, axis=1)
        axes[cur_ax].scatter(
            target_returns,
            rollout_final_r,
            facecolors=face_colors,
            edgecolors=edge_colors,
            label="MODT"
        )
        axes[cur_ax].set_xlim([-5, np.max(target_returns) * 1.05])
        axes[cur_ax].set_ylim([-5, np.max(rollout_final_r) * 1.05])
        axes[cur_ax].set(xlabel="target final reward", ylabel="achieved final reward")
        lims = [
            np.min([axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]),  # min of both axes
            np.max([axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]),  # max of both axes
        ]
        axes[cur_ax].plot(lims, lims, label="oracle", alpha=0.75, zorder=0)
        axes[cur_ax].axvline(
            x = dataset_min_final_r,
            ls="--",
        )
        axes[cur_ax].axvline(
            x = dataset_max_final_r,
            ls="--",
        )
        axes[cur_ax].legend(loc='upper left')
        axes[cur_ax].set_title(f'Final Reward: Target vs. Achieved')
        cur_ax += 1
        

    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.savefig(f'{logsdir}/step={cur_step}_plots.png')
    plt.close()





    def visualize_pareto_front_all_envs():
        pass
    

if __name__ =="__main__":
    pass