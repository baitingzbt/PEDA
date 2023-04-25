import numpy as np
import torch
import time
from tqdm import tqdm
from modt.training.visualizer import visualize
class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        get_batch,
        loss_fn,
        dataset_min_prefs,
        dataset_max_prefs,
        dataset_min_raw_r,
        dataset_max_raw_r,
        dataset_min_final_r,
        dataset_max_final_r,
        scheduler=None,
        eval_fns=[],
        max_iter=0,
        n_steps_per_iter=0,
        eval_only=False,
        concat_rtg_pref=0,
        concat_act_pref=0,
        logsdir='./'
        
    ):
        self.model = model
        self.optimizer = optimizer
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        # for plotting purposes
        self.dataset_min_prefs = dataset_min_prefs
        self.dataset_max_prefs = dataset_max_prefs
        self.dataset_min_raw_r = dataset_min_raw_r # weighted
        self.dataset_max_raw_r = dataset_max_raw_r
        self.dataset_min_final_r = dataset_min_final_r
        self.dataset_max_final_r = dataset_max_final_r
        self.scheduler = scheduler
        self.eval_fns = eval_fns
        self.max_iter = max_iter
        self.n_steps_per_iter = n_steps_per_iter
        self.eval_only = eval_only
        self.concat_rtg_pref = concat_rtg_pref
        self.concat_act_pref = concat_act_pref
        self.logsdir = logsdir
        self.diagnostics = dict()
        self.start_time = time.time()

    def train_iteration(self, ep):
        train_losses = []
        logs = dict()
        train_start = time.time()
        if not self.eval_only:
            self.model.train()
            for ite in tqdm(range(self.n_steps_per_iter), disable=True):
                train_loss = self.train_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()
                    

        logs['time/training'] = time.time() - train_start
        eval_start = time.time()
        self.model.eval()
        cur_step = (ep+1) * self.n_steps_per_iter


        set_final_return, set_unweighted_raw_return, set_weighted_raw_return, set_cum_r_original = [], [], [], []
        for eval_fn in self.eval_fns:
            
            outputs, final_returns, unweighted_raw_returns, weighted_raw_returns, cum_r_original = eval_fn(self.model, cur_step)
            set_final_return.append(np.mean(final_returns, axis=0))
            set_unweighted_raw_return.append(np.mean(unweighted_raw_returns, axis=0))
            set_weighted_raw_return.append(np.mean(weighted_raw_returns, axis=0))
            set_cum_r_original.append(np.mean(cum_r_original, axis=0))
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v


        rollout_unweighted_raw_r = np.array(set_unweighted_raw_return)
        rollout_weighted_raw_r = np.array(set_weighted_raw_return)
        rollout_original_raw_r = np.array(set_cum_r_original)
        target_prefs = np.array([eval_fn.target_pref for eval_fn in self.eval_fns])
        target_returns = np.array([eval_fn.target_reward for eval_fn in self.eval_fns]) # target returns are weighted

        
        
        n_obj = self.model.pref_dim
        # rollout_ratio = rollout_original_raw_r / np.sum(rollout_original_raw_r, axis=1, keepdims=True)
        rollout_logs = {
            'n_obj': n_obj,
            'target_prefs': target_prefs,
            'target_returns': target_returns,
            'dataset_min_prefs': self.dataset_min_prefs,
            'dataset_max_prefs': self.dataset_max_prefs,
            'dataset_min_raw_r': self.dataset_min_raw_r,
            'dataset_max_raw_r': self.dataset_max_raw_r,
            'dataset_min_final_r': self.dataset_min_final_r,
            'dataset_max_final_r': self.dataset_max_final_r,
            'rollout_unweighted_raw_r': rollout_unweighted_raw_r,
            'rollout_weighted_raw_r': rollout_weighted_raw_r, # for finding [achieved return vs. target return]
            'rollout_original_raw_r': rollout_original_raw_r, # unnormalized raw_r, for calculating roll-out ratio
        }
        
        visualize(rollout_logs, self.logsdir, cur_step)
        
        if not self.eval_only:
            cur_step = (ep+1) * self.n_steps_per_iter
            log_file_name = f'{self.logsdir}/step={cur_step}.txt'
            with open(log_file_name, 'a') as f:
                f.write(f"\n\n\n------------------> epoch: {ep} <------------------")
                f.write(f"\nloss = {np.mean(train_losses)}")
                for k in self.diagnostics:
                    f.write(f"\n{k} = {self.diagnostics[k]}")
            
            logs['time/total'] = time.time() - self.start_time
            logs['time/evaluation'] = time.time() - eval_start
            logs['training/train_loss_mean'] = np.mean(train_losses)
            logs['training/train_loss_std'] = np.std(train_losses)

            for k in self.diagnostics:
                logs[k] = self.diagnostics[k]
        return logs, rollout_logs


    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch()
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)
        
        
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
