import numpy as np
import torch

class Fn:
    
    def __init__(self, evaluator, num_eval_episodes, target_reward, target_pref):
        self.evaluator = evaluator
        self.num_eval_episodes = num_eval_episodes
        self.scale = self.evaluator.scale
        self.target_reward = target_reward # this can be mo, or weighted
        self.target_pref = target_pref
        self.pref_dim = evaluator.pref_dim
        self.rtg_dim = evaluator.rtg_dim
    
    def __call__(self, model, cur_step):
        
        target_pref = self.target_pref
        target_return = self.target_reward / self.scale
        returns = np.zeros(shape=(self.num_eval_episodes))
        lengths = np.zeros(shape=(self.num_eval_episodes))
        raw_returns = np.zeros(shape=(self.num_eval_episodes, self.pref_dim))
        weighted_raw_returns = np.zeros(shape=(self.num_eval_episodes, self.pref_dim))
        all_cum_r_original = np.zeros(shape=(self.num_eval_episodes, self.pref_dim))
        for eval_ep in range(self.num_eval_episodes):
            with torch.no_grad():
                ret, length, raw_ret, weighted_raw_ret, cum_r_original = self.evaluator(
                    model,
                    target_return=target_return, # this could be mo, or weighted
                    target_pref=target_pref,
                    cur_step=cur_step
                )
            returns[eval_ep] = ret
            raw_returns[eval_ep, :] = raw_ret
            weighted_raw_returns[eval_ep, :] = weighted_raw_ret
            all_cum_r_original[eval_ep, :] = cum_r_original
            lengths[eval_ep] = length
            
        
        returns *= self.scale
        raw_returns *= self.scale
        target_reward = np.round(self.target_reward, decimals=0) # round to int each entry
        # info for weighted return
        infos = {
            f'total_return_mean/rtg_{target_reward}_pref_{target_pref}': np.mean(returns),
            f'length_mean/rtg_{target_reward}_pref_{target_pref}': np.mean(lengths),
        }
        return infos, returns, raw_returns, weighted_raw_returns, all_cum_r_original


class EvalEpisode:
    
    def __init__(self, evaluator, num_eval_episodes, max_each_obj_traj, rtg_scale, lrModels, use_max_rtg):
        self.evaluator = evaluator
        self.num_eval_episodes = num_eval_episodes
        self.max_each_obj_traj = max_each_obj_traj
        self.rtg_scale = rtg_scale
        self.lrModels = lrModels
        self.rtg_dim = evaluator.rtg_dim
        self.use_max_rtg = use_max_rtg
        
    # returns a list of eval_fn function
    # single_obj target_reward
    def __call__(self, pref_set):
        

        adjusted_target_rewards = []
        n_obj = pref_set.shape[1]
        for pref in pref_set:
            adjusted_target_rewards.append(np.array([self.lrModels[i].predict(pref.reshape(-1, n_obj))[0] for i in range(n_obj)]))
        adjusted_target_rewards = np.array(adjusted_target_rewards)

        # 1-dim rtg uses dot product of raw_reward & pref
        if self.rtg_dim == 1:
            adjusted_target_rewards = np.array([np.dot(adjusted_target_rewards[i], pref_set[i]) for i in range(pref_set.shape[0])])
        # multi-dim rtg uses multiplied element-wise product
        else:
            adjusted_target_rewards = np.multiply(adjusted_target_rewards, pref_set)
        
        if not self.use_max_rtg:
            fns = [Fn(evaluator=self.evaluator,
                num_eval_episodes=self.num_eval_episodes,
                target_reward=adjusted_target_rewards[i] * self.rtg_scale,
                target_pref=target_pref * self.rtg_scale) for i, target_pref in enumerate(pref_set)]
        else:
            adjusted_target_rewards = np.multiply(self.max_each_obj_traj, pref_set)
            fns = [Fn(evaluator=self.evaluator,
                num_eval_episodes=self.num_eval_episodes,
                target_reward=adjusted_target_rewards[i] * self.rtg_scale,
                target_pref=target_pref * self.rtg_scale) for i, target_pref in enumerate(pref_set)]
        return fns