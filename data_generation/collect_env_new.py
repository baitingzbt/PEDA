import os, sys
from morl.hypervolume import InnerHyperVolume
def compute_hypervolume(ep_objs_batch):
    n = len(ep_objs_batch[0])
    HV = InnerHyperVolume(np.zeros(n))
    return HV.compute(ep_objs_batch)

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals/baselines'))
sys.path.append(os.path.join(base_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.model import Policy

import pickle
import torch
import gym
import mujoco_py
import numpy as np
import environments
import argparse
from tqdm import tqdm
from collections import defaultdict
from morl.sample import Sample
from morl.population_2d import Population

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='MO-Hopper-v2')
parser.add_argument('--collect_type', type=str, default="amateur")
parser.add_argument('--preference_type', type=str, default="uniform")
parser.add_argument('--num_traj', type=int, default=10000)
parser.add_argument('--data_path', type=str, default="data_collected")
parser.add_argument('--p_bar', type=bool, default=False)
args = parser.parse_args()

env_names = ["MO-Ant-v2", "MO-HalfCheetah-v2", "MO-Hopper-v2", "MO-Humanoid-v2", "MO-Swimmer-v2", "MO-Walker2d-v2"]
deterministic_transition = [True, True, True, True, True, True]
deterministic_initial = [True, True, True, False, True, True]
env_names_and_infos = {
    "MO-Ant-v2": {
        "base_name": "Ant-v2",
        "max_task": 69,
        "num_processes": 1,
        "base_kwargs": {'layernorm' : False},
        "obj_num": 2,
    },
    "MO-HalfCheetah-v2": {
        "base_name": "HalfCheetah-v2",
        "max_task": 233,
        "num_processes": 1,
        "base_kwargs": {'layernorm' : False},
        "obj_num": 2,
    },
    "MO-Hopper-v2": {
        "base_name": "Hopper-v2",
        "max_task": 95,
        "num_processes": 1,
        "base_kwargs": {'layernorm' : False},
        "obj_num": 2,
    },
    "MO-Humanoid-v2": {
        "base_name": "Humanoid-v2",
        "max_task": 354,
        "num_processes": 1,
        "base_kwargs": {'layernorm' : False},
        "obj_num": 2,
    },
    "MO-Swimmer-v2": {
        "base_name": "Swimmer-v2",
        "max_task": 173,
        "num_processes": 1,
        "base_kwargs": {'layernorm' : False},
        "obj_num": 2,
    },
    "MO-Walker2d-v2": {
        "base_name": "Walker2d-v2",
        "max_task": 364,
        "num_processes": 1,
        "base_kwargs": {'layernorm' : False},
        "obj_num": 2,
    },
    "MO-Hopper-v3": {
        "base_name": "Hopper-v3",
        "max_task": 2444,
        "num_processes": 1,
        "base_kwargs": {'layernorm': False},
        "obj_num": 3,
    }
}

def collect_helper(args, all_datas):
    data_path = f"{args.data_path}/{args.env_name}"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    filename = f"{data_path}/{args.env_name}_{args.num_traj}_new{args.collect_type}_{args.preference_type}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(all_datas, f)

def generate_one_alpha(pref_dim, alpha_type):
    assert alpha_type in ['wide', 'narrow']
    _low = 0
    _high = 10e6
    alpha = np.random.uniform(low=_low, high=_high, size=pref_dim) if alpha_type == 'wide' \
        else np.random.uniform(low=_high/3, high=_high*2/3, size=pref_dim)
    return alpha

def dirichlet_preference(pref_dim, pref_range, n_alpha=1, n_pref_each_alpha=1):
    # sanity check
    assert pref_range in ['wide', 'narrow']
    pref_dim = int(pref_dim)
    n_alpha = int(n_alpha)
    n_pref_each_alpha = int(n_pref_each_alpha)
    # find alpha as uniform prior
    alphas = np.array([generate_one_alpha(pref_dim=pref_dim, alpha_type=pref_range) for _ in range(n_alpha)])
    # find preference using alpha + dirichlet
    preferences = np.zeros(shape=(n_alpha*n_pref_each_alpha, pref_dim))
    for i, alpha in enumerate(alphas):
        start = n_pref_each_alpha*i
        end = n_pref_each_alpha*(i+1)
        tmp = np.random.dirichlet(alpha, size=n_pref_each_alpha)
        tmp = tmp / np.linalg.norm(tmp, ord=1, axis=1, keepdims=True)
        preferences[start:end,:] = tmp
    return preferences

default_dirichlet_wide = lambda pref_dim, n_pref: dirichlet_preference(pref_dim, "wide", n_pref, 1)
defualt_dirichlet_narrow = lambda pref_dim, n_pref: dirichlet_preference(pref_dim, "narrow", n_pref, 1)

def uniform_preference(pref_dim, n_pref):
    k = np.array([np.random.exponential(scale=1.0, size=pref_dim) for _ in range(n_pref)])
    return k / np.sum(k, axis=1)[:, np.newaxis]

class PrefDist:
    
    def __init__(self, preference_type):
        if preference_type == "uniform":
            self.pref_func = uniform_preference
        elif preference_type == "wide":
            self.pref_func = default_dirichlet_wide
        elif preference_type == "narrow":
            self.pref_func = defualt_dirichlet_narrow
    
    def __call__(self, pref_dim, n_pref):
        return self.pref_func(pref_dim, n_pref)
        
class RejectSampling:
    
    def __init__(self, preference_dist, min_each_obj, max_each_obj, n_obj):
        self.preference_dist = preference_dist
        self.min_each_obj = min_each_obj
        self.max_each_obj = max_each_obj
        self.n_obj = n_obj

    def validate_preference(self, preference):
        for i, p in enumerate(preference):
            if p < self.min_each_obj[i] or p > self.max_each_obj[i]:
                return False # not valid
        return True # valid

    def get_preferences(self, n_pref_wanted):
        preferences = np.zeros(shape=(n_pref_wanted, n_obj))
        cur = 0
        while cur < n_pref_wanted:
            one_pref = self.preference_dist(n_obj, 1)[0]
            if self.validate_preference(one_pref):
                preferences[cur, :] = one_pref
                cur += 1
        return preferences

class GetPolicyId:
    
    def __init__(self, objs_normalized):
        self.objs_normalized = objs_normalized

    def __call__(self, sampled_preferences):
        policy_ids = np.zeros(shape=sampled_preferences.shape[0])
        for i, pref in enumerate(sampled_preferences):
            priorities = np.sum(np.square(self.objs_normalized - pref), axis=1) # sum of squared difference each row
            policy_ids[i] = np.argmin(priorities)
        return policy_ids
        # assume preference is valid here after reject-sampling
        

def update_item(datas, name, values):
    datas[name].append(values)

def eval_collect(args, samples, n_obj):
    
    all_datas = [{} for _ in range(args.num_traj)]
    eval_env = gym.make(args.env_name)
    objs_normalized = np.array([samp.objs_normalized for samp in samples])
    min_each_obj = np.min(objs_normalized, axis=0)
    max_each_obj = np.max(objs_normalized, axis=0)
    obj_range = max_each_obj - min_each_obj
    preference_dist = PrefDist(args.preference_type)
    
    if args.preference_type == "narrow":
        min_each_obj += obj_range / 3
        max_each_obj -= obj_range / 3
        
    if args.preference_type == "uniform":
        reject_sampler = RejectSampling(preference_dist, min_each_obj, max_each_obj, n_obj)
        sampled_preferences = reject_sampler.get_preferences(args.num_traj)
    else:
        sampled_preferences = preference_dist(n_obj, args.num_traj) * obj_range + min_each_obj
        sampled_preferences = sampled_preferences / np.sum(sampled_preferences, axis=1, keepdims=True)
        
    # PGMORL has a group of single-obj policies
    # here we try to decide the best single-obj to use for rollout
    getPolicyId = GetPolicyId(objs_normalized)
    policy_ids = getPolicyId(sampled_preferences)
    with torch.no_grad():

        for i in tqdm(range(args.num_traj), disable=(not args.p_bar)):
            
            
            preference = sampled_preferences[i, :]
            datas = defaultdict(list)
            
            policy_id = int(policy_ids[i])
            ob_rms = samples[policy_id].env_params['ob_rms']
            policy = samples[policy_id].actor_critic
            '''
            --------------------- EXPLANATION ---------------------
            we use RANDOMIZED environment reset instead of fixed reset value as in PGMORL
            this helps to diversify environment trajectories, so we need to give a seed
            '''
            eval_env.seed(i)
            obs = eval_env.reset()
            done = False
            ep_raw_reward = np.zeros(n_obj)
            
            while not done:
                
                update_item(datas, "observations", np.array(obs))
                # reload normalizing value used when training behavioral policy
                ob_norm = np.clip((obs - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)
                
                # below should match in PEDA Appendix C: https://openreview.net/pdf?id=Ki4ocDm364
                if args.collect_type == "amateur":
                    if args.env_name in ['MO-Ant-v2', 'MO-Hopper-v2', 'MO-Hopper-v3', 'MO-Walker2d-v2', 'MO-HalfCheetah-v2']:
                        if np.random.uniform(0, 1) < 0.35:
                            action = policy.act(torch.Tensor(ob_norm).double().unsqueeze(0), None, None, deterministic=True)[1]
                        else:
                            action_deterministic = policy.act(torch.Tensor(ob_norm).double().unsqueeze(0), None, None, deterministic=True)[1]
                            action = action_deterministic * np.random.uniform(low=0.35, high=1.65)
                    elif args.env_name in ['MO-Swimmer-v2']:
                        if np.random.uniform(0, 1) < 0.35:
                            action = np.array([eval_env.action_space.sample()])
                        else:
                            action_deterministic = policy.act(torch.Tensor(ob_norm).double().unsqueeze(0), None, None, deterministic=True)[1]
                            action = action_deterministic * np.random.uniform(low=0.35, high=1.65)
                else: # expert
                    action = policy.act(torch.Tensor(ob_norm).double().unsqueeze(0), None, None, deterministic=True)[1]
                

                if args.env_name in ["MO-Hopper-v2", "MO-Hopper-v3"]:
                    action = np.clip(action, [-2, -2, -4], [2, 2, 4])
                else:
                    action = np.clip(action, -1, 1)
                
                next_obs, _, done, info = eval_env.step(action)
                
                raw_reward = info['obj']
                # reward = np.dot(preference, raw_reward)
                
                
                update_item(datas, "actions", np.array(action[0]))
                # update_item(datas, "rewards", np.array(reward))
                update_item(datas, "next_observations", np.array(next_obs))
                update_item(datas, 'preference', np.array(preference))
                update_item(datas, "terminals", np.array(done))
                update_item(datas, "raw_rewards", np.array(raw_reward))
                
                obs = next_obs
                ep_raw_reward += raw_reward

            for k, v in datas.items():
                datas[k] = np.array(v)
                
            all_datas[i] = datas

    eval_env.close()
    return all_datas


def eval_sample_hv(args, samples, eval_per_sample):
    
    with torch.no_grad():
        eval_out = np.zeros(shape=(len(samples), args.n_obj))
        for i, sample in enumerate(tqdm(samples, disable=(not args.p_bar))):
            policy = sample.actor_critic
            ob_rms = sample.env_params['ob_rms']
            

            all_this_sample = []
            for eval in range(eval_per_sample):
                env = gym.make(args.env_name)
                env.seed(np.random.randint(0, 100000))
                unweighted_raw_return = np.zeros(args.n_obj)
                done = False
                obs = env.reset()
                while not done:
                    ob_norm = np.clip((obs - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)
                    action = policy.act(torch.Tensor(ob_norm).double().unsqueeze(0), None, None, deterministic=True)[1]
                    next_obs, _, done, info = env.step(action)
                    unweighted_raw_return += info['obj']
                    obs = next_obs
                all_this_sample.append(unweighted_raw_return)

            this_sample_median_ret = np.median(all_this_sample, axis=0)
            eval_out[i, :] = this_sample_median_ret
    hv = compute_hypervolume(eval_out)
    print(f"Env = {args.env_name}; data = {args.collect_type}_{args.preference_type}; hv = {hv:.3e}")


if __name__ == "__main__":
    
    env_name = args.env_name
    env_info = env_names_and_infos[env_name]
    base_env = gym.make(env_name)
    folder = f"Precomputed_Results/{env_info['base_name']}/final"
    
    objs_path = f"{folder}/objs.txt"
    with open(objs_path, 'r') as f:
        lines = f.read().splitlines()
        objs_original = np.array([[float(v) for v in l.split(",")] for l in lines])
    objs_normalized = objs_original / np.sum(objs_original, axis=1, keepdims=True)
    
    n_obj = env_info['obj_num']
    args.n_obj = n_obj
    samples = []
    for task in tqdm(range(env_info['max_task'] + 1), disable=(not args.p_bar)):
        
        
        policy_path = f"{folder}/EP_policy_{task}.pt"
        actor_critic = Policy(action_space=base_env.action_space,
                              obs_shape=base_env.observation_space.shape,
                              base_kwargs=env_info['base_kwargs'],
                              obj_num=n_obj)
        actor_critic.eval()
        actor_critic.load_state_dict(torch.load(policy_path))
        actor_critic.to("cpu").double()
        
        ob_rms_path = f"{folder}/EP_env_params_{task}.pkl"
        with open(ob_rms_path, 'rb') as f:
            ob = pickle.load(f)

        # default as given in PGMORL, only as a place-holder, these params doesn't
        # matter for collection
        agent = algo.PPO(
            actor_critic,
            None,
            10,
            32,
            0.5,
            0,
            lr=3e-4,
            eps=1e-5,
            max_grad_norm=None
        )
        
        sample = Sample(ob, actor_critic, agent)
        sample.objs = objs_original[task]
        sample.objs_normalized = objs_normalized[task]
        samples.append(sample)

    # also place holder here, these args are only needed for training
    # not for simulation
    args.pbuffer_num = 1
    args.pbuffer_size = 1
    
    '''
    optional: check the quality of behavioral policy ckpt (as rolled out again)
    '''
    # eval_sample_hv(args, samples, 5)

    all_datas = eval_collect(args, samples, n_obj)
    collect_helper(args, all_datas)