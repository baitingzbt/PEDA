from copy import deepcopy
from logging import warning
import numpy as np
import torch
import wandb
import argparse
import pickle
import sys
import os
import gym
from gym.spaces.box import Box
import environments # import to register environments for multi-objective
from math import isclose
from modt.evaluation.evaluate_episodes import EvalEpisode
from modt.training.loader import GetBatch
from sklearn.linear_model import LinearRegression
from torch import nn
from state_norm_params import state_norm_params # we use normalization parameter for states from the behavioral policy
import random

isCloseToOne = lambda x: isclose(x, 1, rel_tol=1e-12)
def pref_grid(n_obj, max_prefs=None, min_prefs=None, granularity=5):
    max_prefs = np.ones(n_obj) if max_prefs is None else max_prefs
    min_prefs = np.zeros(n_obj) if min_prefs is None else min_prefs
    grid = np.array([x/granularity for x in range(granularity+1)])
    prefs = [[]]
    grid = tuple(grid)
    for _ in range(n_obj):
        prefs = [x+[y] for x in prefs for y in grid if sum(x+[y]) < 1 or isCloseToOne(sum(x+[y]))]
    prefs = np.array([p for p in prefs if isCloseToOne(sum(p))])
    for i in range(n_obj):
        prefs[:, i] = prefs[:, i] * (max_prefs[i] - min_prefs[i]) + min_prefs[i]
    prefs = prefs / np.sum(prefs, axis=1, keepdims=True)
    return prefs

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def experiment(
    variant
):
    env_name = variant['env']
    dataset = variant['dataset']
    device = variant['device']
    log_to_wandb = variant['log_to_wandb']
    model_type = variant['model_type']
    mode = variant['mode']
    concat_state_pref = variant['concat_state_pref']
    concat_rtg_pref = variant['concat_rtg_pref']
    concat_act_pref = variant['concat_act_pref']
    use_obj = variant['use_obj']
    percent_dt = variant['percent_dt']
    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    warmup_steps = variant['warmup_steps']
    normalize_reward = variant['normalize_reward']
    mo_rtg = variant['mo_rtg']
    eval_only = variant['eval_only']
    return_loss = variant['return_loss']
    pref_loss = variant['pref_loss']
    num_steps_per_iter = int(variant["num_steps_per_iter"])
    max_iters = int(variant["max_iters"])
    optimizer_name = variant['optimizer']
    eval_context_length = variant['eval_context_length']
    rtg_scale = variant['rtg_scale']
    granularity = variant['granularity']
    use_max_rtg = variant['use_max_rtg']
    
    if model_type == 'dt':
        from modt.training.seq_trainer import SequenceTrainer as Trainer
        from modt.evaluation.evaluator_dt import EvaluatorDT as Evaluator
        from modt.models.decision_transformer import DecisionTransformer as Model
    elif model_type == 'bc':
        from modt.training.act_trainer import ActTrainer as Trainer
        from modt.evaluation.evaluator_bc import EvaluatorBC as Evaluator
        from modt.models.mlp_bc import MLPBCModel as Model
    elif model_type == 'rvs':
        # from pytorch_lightning import Trainer
        from modt.training.rvs_trainer import RVSTrainer as Trainer
        from modt.evaluation.evaluator_rvs import EvaluatorRVS as Evaluator
        from rvs.src.rvs.policies import RvS as Model
    
    if optimizer_name == "adam":
        from torch.optim import AdamW as Optimizer
    elif optimizer_name == "lamb":
        from modt.models.lamb import Lamb as Optimizer
    
    
    ckptdir = variant['dir'] + '/ckpt'
    logsdir = variant['dir'] + '/logs'
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)
    if not os.path.exists(logsdir):
        os.makedirs(logsdir)

    env = gym.make(env_name)
    act_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    reward_size = env.obj_dim
    pref_dim = reward_size
    rtg_dim = pref_dim if mo_rtg else 1
    scale = 100
    max_ep_len = 500
    if not normalize_reward:
        scale *= 10
    
    # if using multiple dataset, load all at once
    dataset_paths = [f"data/{env_name}/{env_name}_50000_{d}.pkl" for d in dataset]
    trajectories = []
    for data_path in dataset_paths:
        with open(data_path, 'rb') as f:
            trajectories.extend(pickle.load(f))

    states, traj_lens, returns, returns_mo, preferences = [], [], [], [], []
    min_each_obj_step = np.min(np.vstack([np.min(traj['raw_rewards'], axis=0) for traj in trajectories]), axis=0)
    max_each_obj_step = np.max(np.vstack([np.max(traj['raw_rewards'], axis=0) for traj in trajectories]), axis=0)

    for traj in trajectories:
        if concat_state_pref != 0:
            traj['observations'] = np.concatenate((traj['observations'], np.tile(traj['preference'], concat_state_pref)), axis=1)
            
        if normalize_reward:
            traj['raw_rewards'] = (traj['raw_rewards'] - min_each_obj_step) / (max_each_obj_step - min_each_obj_step)
        
        traj['rewards'] = np.sum(np.multiply(traj['raw_rewards'], traj['preference']), axis=1)
        states.append(traj['observations'])
        traj_lens.append(len(traj['observations']))
        returns.append(traj['rewards'].sum())
        returns_mo.append(traj['raw_rewards'].sum(axis=0))
        preferences.append(traj['preference'][0, :])
        
    traj_lens, returns, returns_mo, states, preferences = np.array(traj_lens), np.array(returns), np.array(returns_mo), np.array(states), np.array(preferences)

    if not isCloseToOne(percent_dt):
        num_traj_wanted = int(percent_dt * len(trajectories))
        indices_wanted = np.unique(np.argpartition(returns_mo, -num_traj_wanted, axis=0)[-num_traj_wanted:])
        trajectories = np.array([trajectories[i] for i in indices_wanted])
        traj_lens = traj_lens[indices_wanted]
        returns = returns[indices_wanted]
        returns_mo = returns_mo[indices_wanted, :]
        states = states[indices_wanted]
        preferences = preferences[indices_wanted, :]
        

    states = np.concatenate(states, axis=0)
    state_mean = state_norm_params[env_name]["mean"]
    state_std = np.sqrt(state_norm_params[env_name]["var"])
    state_mean = np.concatenate((state_mean, np.zeros(concat_state_pref * pref_dim)))
    state_std = np.concatenate((state_std, np.ones(concat_state_pref * pref_dim)))
    state_dim += pref_dim * concat_state_pref
        
    
    lrModels = [LinearRegression() for _ in range(pref_dim)]
    for obj, lrModel in enumerate(lrModels):
        lrModel.fit(preferences.reshape((-1, pref_dim)), returns_mo[:, obj])
    # all experiments use pre-cashed expert_uniform models
    with open(f"lr_models/{env_name}_expert_uniform.pkl", 'rb') as f:
        lrModels = pickle.load(f)
    
    max_prefs = np.max(preferences, axis=0)
    min_prefs = np.min(preferences, axis=0)
    if concat_act_pref == 0 and concat_rtg_pref == 0 and concat_state_pref == 0 and model_type == "bc":
        granularity = 1
    prefs = pref_grid(pref_dim, granularity=granularity)
    print('=' * 50)
    print(f'Starting new experiment: {env_name} {"_".join(dataset)}')
    print(f'{len(traj_lens)} trajectories, {sum(traj_lens)} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    sorted_inds = np.argsort(returns)  # lowest to highest
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    get_batch = GetBatch(
        batch_size=batch_size,
        # RvS conditions on future avg return, always until the end of traj
        max_len=K if model_type != 'rvs' else 1,
        max_ep_len=max_ep_len,
        num_trajectories=len(traj_lens),
        p_sample=p_sample,
        trajectories=trajectories,
        sorted_inds=sorted_inds,
        state_dim=state_dim,
        act_dim=act_dim,
        pref_dim=pref_dim,
        rtg_dim=rtg_dim,
        state_mean=state_mean,
        state_std=state_std,
        scale=scale,
        device=device,
        act_low = np.array(env.action_space.low),
        act_high = np.array(env.action_space.high),
        avg_rtg = bool(model_type == "rvs"), # RvS conditions on future avg return
        use_obj = use_obj,
        concat_state_pref = concat_state_pref
    )

    video_dir = variant['dir'] + f'/{model_type}_eval_videos'
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
        
    evaluator = Evaluator(
        env_name, state_dim, act_dim, pref_dim, rtg_dim,
        max_ep_len=max_ep_len,
        scale=scale,
        state_mean=state_mean,
        state_std=state_std,
        min_each_obj_step=min_each_obj_step,
        max_each_obj_step=max_each_obj_step,
        act_scale=np.array(env.action_space.high),
        use_obj=use_obj,
        concat_state_pref=concat_state_pref,
        concat_rtg_pref=concat_rtg_pref,
        concat_act_pref=concat_act_pref,
        normalize_reward=normalize_reward,
        video_dir=video_dir,
        device=device,
        mode=mode,
        logsdir=logsdir,
        eval_only=eval_only
    )
    # this simply returns a list of lists of callable function objects
    # each is initialized with the specific evaluator, and init-pref + init-rtg
    eval_episodes = EvalEpisode(
        evaluator=evaluator,
        num_eval_episodes=num_eval_episodes,
        max_each_obj_traj=np.max(returns_mo, axis=0),
        rtg_scale=rtg_scale,
        lrModels=lrModels,
        use_max_rtg=use_max_rtg
    )
    
    if model_type in ['dt', 'bc']:
        model = Model(
            state_dim=state_dim,
            act_dim=act_dim,
            pref_dim=pref_dim,
            rtg_dim=rtg_dim,
            hidden_size=variant['embed_dim'],
            max_length=K,
            eval_context_length=eval_context_length,
            max_ep_len=max_ep_len,
            act_scale=torch.from_numpy(np.array(env.action_space.high)),
            use_pref=variant['use_pref_predict_action'],
            concat_state_pref=concat_state_pref,
            concat_rtg_pref=concat_rtg_pref,
            concat_act_pref=concat_act_pref,
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout']
        ).to(device=device)
    elif model_type == "rvs":
        # change dimension for concatenating preference
        # we don't really use anything in the obs space other than dimension
        observation_space_place_holder = Box(
            low=np.zeros(state_dim),
            high=np.ones(state_dim),
        )
        model = Model(
            observation_space=observation_space_place_holder,
            action_space=env.action_space,
            state_dim=state_dim,
            act_dim=act_dim,
            pref_dim=pref_dim,
            rtg_dim=rtg_dim,
            hidden_size=variant['embed_dim'],
            depth=variant['n_layer'],
            learning_rate=variant['learning_rate'],
            batch_size=batch_size,
            activation_fn=nn.ReLU,
            dropout_p=variant['dropout'],
            unconditional_policy=False,
            reward_conditioning=True,
            env_name=env_name,
        ).to(device=device)
        model.state_dim = state_dim
        model.act_dim = act_dim
        model.pref_dim = pref_dim
        model.rtg_dim = rtg_dim
    
    optimizer = Optimizer(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )

    if variant['ckpt'] != '':
        print(f'[Info] Loading ckpt from {variant["ckpt"]}')
        ckpt = torch.load(variant['ckpt'])
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps+1)/warmup_steps, 1)
    )
    # default version only trains on action loss
    if (not pref_loss) and (not return_loss):
        loss_fn = lambda s_hat, a_hat, r_hat, pref_hat, s, a, r, pref: \
            torch.mean((a_hat - a) ** 2)
    # alternatively, can train on predicting preference
    elif (not pref_loss) and return_loss:
        loss_fn = lambda s_hat, a_hat, r_hat, pref_hat, s, a, r, pref: \
            torch.mean((a_hat - a) ** 2) + torch.mean((r_hat - r) ** 2)
    elif pref_loss and (not return_loss):
        loss_fn = lambda s_hat, a_hat, r_hat, pref_hat, s, a, r, pref: \
            torch.mean((a_hat - a) ** 2) + torch.mean((pref_hat - pref) ** 2)
    else:
        loss_fn = lambda s_hat, a_hat, r_hat, pref_hat, s, a, r, pref: \
            torch.mean((a_hat - a) ** 2) + torch.mean((r_hat - r) ** 2) + torch.mean((pref_hat - pref) ** 2)
    
    

    max_raw_r = np.multiply(np.max(returns_mo, axis=0), max_prefs) # based on weighted values
    min_raw_r = np.multiply(np.min(returns_mo, axis=0), min_prefs)
    max_final_r = np.max(returns)
    min_final_r = np.min(returns)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=loss_fn,
        dataset_min_prefs=min_prefs,
        dataset_max_prefs=max_prefs,
        dataset_min_raw_r=min_raw_r,
        dataset_max_raw_r=max_raw_r,
        dataset_min_final_r=min_final_r,
        dataset_max_final_r=max_final_r,
        eval_fns=eval_episodes(pref_set=prefs), # this return a list (of lists) of eval_fns
        max_iter=max_iters,
        n_steps_per_iter=num_steps_per_iter,
        eval_only=eval_only,
        concat_rtg_pref=concat_rtg_pref,
        concat_act_pref=concat_act_pref,
        logsdir=logsdir
    )
    
    
    for iter in range(max_iters):

        step = int((iter+1) * num_steps_per_iter)
        logs, rollout_logs = trainer.train_iteration(ep=iter)
        
        # save rollout results, later we can use these and don't need to rollout again
        filename = f'{logsdir}/step={step}_rollout.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(rollout_logs, f)
        
        if eval_only:
            break
        
        # save model
        filename = f'{ckptdir}/step={step}.ckpt'
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename)
        
        
        # save to wandb
        if log_to_wandb:
            wandb.log(logs)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MO-Hopper-v2')
    parser.add_argument('--dataset', type=str, nargs='+', default=['expert_highh'])
    parser.add_argument('--data_mode', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt, bc, rvs
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--n_layer', type=int, default=3) # lamb's default should be 4
    parser.add_argument('--n_head', type=int, default=1) # lamb's default should be 4
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-3)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--num_steps_per_iter', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dir', type=str, default='test_dir')
    parser.add_argument('--log_to_wandb', type=bool, default=False)
    parser.add_argument('--wandb_group', type=str, default='none')
    parser.add_argument('--use_obj', type=int, default=-1) # decay to only 1-obj scenario. -1 default means nothing is decayed
    parser.add_argument('--percent_dt', type=float, default=1) # make DT to only use top% of data, default would be 99%
    parser.add_argument('--use_pref_predict_action', type=bool, default=False)
    parser.add_argument('--concat_state_pref', type=int, default=0)
    parser.add_argument('--concat_rtg_pref', type=int, default=0)
    parser.add_argument('--concat_act_pref', type=int, default=0)
    parser.add_argument('--normalize_reward', type=bool, default=False)
    parser.add_argument('--mo_rtg', type=bool, default=False)
    parser.add_argument('--eval_only', type=bool, default=False)
    parser.add_argument('--return_loss', type=bool, default=False)
    parser.add_argument('--pref_loss', type=bool, default=False)
    parser.add_argument('--optimizer', type=str, default="adam") # adam, lamb
    parser.add_argument('--eval_context_length', type=int, default=5)
    parser.add_argument('--rtg_scale', type=float, default=1)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--granularity', type=int, default=1)
    parser.add_argument('--use_max_rtg', type=bool, default=False)
    args = parser.parse_args()
    

    seed = args.seed if args.seed is not None else np.random.randint(0, 100000)
    seed_everything(seed=seed)
    
    dataset_name = '_'.join(args.dataset)
    
    args.run_name = f"{args.dir}/{args.env}/{dataset_name}/K={args.K}/mo_rtg={args.mo_rtg}/rtg_scale={int(args.rtg_scale * 100)}/norm_rew={args.normalize_reward}/concat_state_pref={args.concat_state_pref}/concat_rtg_pref={args.concat_rtg_pref}/concat_act_pref={args.concat_act_pref}/percent={args.percent_dt}/batch={args.batch_size}/dim={args.embed_dim}/layers={args.n_layer}/obj={args.use_obj}/use_pref={args.use_pref_predict_action}/return_loss={args.return_loss}/pref_loss={args.pref_loss}/optim={args.optimizer}/seed={seed}"

    if args.log_to_wandb:
        wandb.init(
            project=args.wandb_group,
            entity="baitingz",
            name=args.run_name
        )
    
    args.dir = args.run_name
    experiment(variant=vars(args))
