from copy import deepcopy
import numpy as np
import torch
from modt.evaluation import Evaluator
from collections import defaultdict

class EvaluatorDT(Evaluator):
    
        
    def __call__(self, model, target_return, target_pref, cur_step):

        model.eval()
        model.to(device=self.device)
        # add a little variance to make sure eval results are not by luck
        # target_pref += np.random.normal(loc=0.0, scale=0.001, size=target_pref.shape)
        # target_pref = target_pref / np.sum(target_pref)
        # target_return += np.random.normal(loc=0.0, scale=0.001, size=target_return.shape)

        with torch.no_grad():
            init_target_return = deepcopy(target_return)

            init_target_pref = deepcopy(target_pref)
            
            state_mean = torch.from_numpy(self.state_mean).to(device=self.device, dtype=torch.float32)
            state_std = torch.from_numpy(self.state_std).to(device=self.device, dtype=torch.float32)
            
            seed = np.random.randint(0, 10000)
            self.eval_env.seed(seed) # fixed seeding in evaluation to visualize
            state_np = self.eval_env.reset()

            state_np = np.concatenate((state_np, np.tile(init_target_pref, self.concat_state_pref)), axis=0)
            
            state_tensor = torch.from_numpy(state_np).to(device=self.device, dtype=torch.float32).reshape(1, self.state_dim)
            state_tensor = torch.clip((state_tensor - state_mean) / state_std, -10, 10)
            states = state_tensor
            
            # if self.mode == 'noise':
            #     state = state + np.random.normal(0, 0.1, size=state.shape).astype(np.float32)
            
            actions = torch.zeros((0, self.act_dim), device=self.device, dtype=torch.float32)
            # prefs = torch.zeros((0, self.pref_dim), device=self.device, dtype=torch.float32)

            # prefs_to_go = torch.from_numpy(target_pref).to(device=self.device, dtype=torch.float32).reshape(1, self.pref_dim)
            pref_np = np.array(target_pref)
            pref_tensor = torch.from_numpy(pref_np).reshape(1, self.pref_dim).to(device=self.device, dtype=torch.float32)
            prefs = pref_tensor
            
            target_return = torch.tensor(target_return, device=self.device, dtype=torch.float32).reshape(1, self.rtg_dim)
            timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)
            
            episode_return_eval, episode_length_eval = 0, 0
            unweighted_raw_reward_cumulative_eval = np.zeros(shape=(self.pref_dim), dtype=np.float32)
            unweighted_raw_reward_cumulative_model = np.zeros(shape=(self.pref_dim), dtype=np.float32)
            
            cum_r_original = np.zeros(shape=(self.pref_dim), dtype=np.float32)
            for t in range(self.max_ep_len):
                # add padding
                actions = torch.cat([actions, torch.zeros((1, self.act_dim), device=self.device)], dim=0)

                action = model.get_action(
                    states.to(dtype=torch.float32),
                    actions.to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    prefs.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
                actions[-1] = action
                action = action.detach().cpu().numpy()
                action = np.multiply(action, self.act_scale)

                state_np, _, done, info = self.eval_env.step(action)
                
                
                # eval: for return, don't process any data, NO clipping, NO rewriting, etc.
                # model: for auto-reg rollout, process data
                if self.normalize_reward:
                    unweighted_raw_reward_eval = (info['obj'] - self.min_each_obj_step) / (self.max_each_obj_step - self.min_each_obj_step) / self.scale
                    unweighted_raw_reward_model = np.clip((info['obj'] - self.min_each_obj_step) / (self.max_each_obj_step - self.min_each_obj_step), 0, 1) / self.scale
                else:
                    unweighted_raw_reward_eval = info['obj'] / self.scale
                    unweighted_raw_reward_model = info['obj'] / self.scale
                    
                cum_r_original += info['obj']
                
                final_reward_eval = np.dot(init_target_pref, unweighted_raw_reward_eval)
                final_reward_model = np.dot(init_target_pref, unweighted_raw_reward_model)
                weighted_raw_reward_eval = np.multiply(init_target_pref, unweighted_raw_reward_eval)
                weighted_raw_reward_model = np.multiply(init_target_pref, unweighted_raw_reward_model)
                unweighted_raw_reward_cumulative_eval += unweighted_raw_reward_eval
                unweighted_raw_reward_cumulative_model += unweighted_raw_reward_model
                
                state_np = np.concatenate((state_np, np.tile(init_target_pref, self.concat_state_pref)), axis=0)
                state_tensor = torch.from_numpy(state_np).to(device=self.device, dtype=torch.float32).reshape(1, self.state_dim)
                state_tensor = torch.clip((state_tensor - state_mean) / state_std, -10, 10)
                states = torch.cat([states, state_tensor], dim=0)
                prefs = torch.cat([prefs, pref_tensor], dim=0)

                

                unweighted_raw_reward_model = torch.from_numpy(np.array(unweighted_raw_reward_model)).to(device=self.device).reshape(1, self.pref_dim)
                weighted_raw_reward_model = torch.from_numpy(np.array(weighted_raw_reward_model)).to(device=self.device).reshape(1, self.pref_dim)

                
                if self.rtg_dim == 1:
                    pred_return = target_return[-1] - final_reward_model
                else:
                    pred_return = target_return[-1] - weighted_raw_reward_model
                target_return = torch.cat([target_return, pred_return.reshape(1, self.rtg_dim)], dim=0)
                timesteps = torch.cat([timesteps, torch.ones((1, 1), device=self.device, dtype=torch.long) * (t+1)], dim=1)

                # MODT: find final reward through dot product
                episode_return_eval += final_reward_eval
                episode_length_eval += 1

                if done:
                    break

            target_ret_scaled_back = np.round(init_target_return * self.scale, 3) # this is normalized
            weighted_raw_reward_cumulative_eval = np.round(np.multiply(unweighted_raw_reward_cumulative_eval * self.scale, init_target_pref), 3)
            unweighted_raw_return_cumulative_eval = np.round(unweighted_raw_reward_cumulative_eval * self.scale, 3)
            total_return_scaled_back_eval = np.round(np.sum(weighted_raw_reward_cumulative_eval), 3)
            if not self.eval_only:
                log_file_name = f'{self.logsdir}/step={cur_step}.txt'
                with open(log_file_name, 'a') as f:
                    f.write(f"\ntarget return: {target_ret_scaled_back} ------------> {weighted_raw_reward_cumulative_eval}\n")
                    f.write(f"target pref: {np.round(init_target_pref, 3)} ------------> {np.round(cum_r_original / np.sum(cum_r_original), 3)}\n")
                    f.write(f"\tunweighted raw returns: {unweighted_raw_return_cumulative_eval}\n")
                    f.write(f"\tweighted raw return: {weighted_raw_reward_cumulative_eval}\n")
                    f.write(f"\tweighted final return: {total_return_scaled_back_eval}\n")
                    f.write(f"\tlength: {episode_length_eval}\n")
            
            # self.decide_save_video(np.multiply(actions.detach().cpu().numpy(), self.act_scale), raw_rewards_cumulative, init_target_return, init_target_pref, seed)
            return episode_return_eval, episode_length_eval, unweighted_raw_reward_cumulative_eval, weighted_raw_reward_cumulative_eval, cum_r_original