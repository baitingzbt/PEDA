import gym
import numpy as np
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder

class Evaluator:
    
    def __init__(
        self,
        env_name,
        state_dim,
        act_dim,
        pref_dim,
        rtg_dim,
        max_ep_len,
        scale,
        state_mean,
        state_std,
        min_each_obj_step,
        max_each_obj_step,
        act_scale,
        use_obj,
        concat_state_pref,
        concat_rtg_pref,
        concat_act_pref,
        normalize_reward,
        video_dir,
        device,
        mode,
        eval_only, 
        logsdir
    ):
        self.env_name=env_name
        self.eval_env = gym.make(env_name)
        self.eval_env.reset()
        self.state_dim=state_dim
        self.act_dim=act_dim
        self.pref_dim=pref_dim
        self.rtg_dim=rtg_dim
        self.max_ep_len=max_ep_len
        self.scale=scale
        self.state_mean=state_mean
        self.state_std=state_std
        self.min_each_obj_step=min_each_obj_step
        self.max_each_obj_step=max_each_obj_step
        self.act_scale=act_scale if act_scale is not None else np.ones(shape=act_dim)
        self.use_obj=use_obj
        self.concat_state_pref=concat_state_pref
        self.concat_rtg_pref=concat_rtg_pref
        self.concat_act_pref=concat_act_pref
        self.normalize_reward=normalize_reward
        self.video_dir=video_dir
        self.device=device
        self.mode=mode
        self.logsdir=logsdir
        self.eval_only=eval_only
    
    def decide_save_video(self, actions, raw_rewards_cumulative, init_target_return, init_target_pref, seed):

        self.eval_env.seed(seed)
        self.eval_env.reset()

        ratio = raw_rewards_cumulative / np.sum(raw_rewards_cumulative)
        videoRecorder = VideoRecorder(self.eval_env.env, \
            f"{self.video_dir}/outputRaw={np.round(raw_rewards_cumulative * self.scale)}_outputRatio={np.round(ratio, 2)}_inputRtg={np.round(init_target_return * self.scale)}_inputPref={np.round(init_target_pref, 2)}.mp4"
        )
        for a in actions:
            self.eval_env.env.step(a)
            videoRecorder.capture_frame()
        videoRecorder.close()
        self.eval_env.reset()
    
        
    def __call__(self):
        raise NotImplementedError
