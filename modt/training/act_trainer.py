import numpy as np
import torch
from modt.training.trainer import Trainer


class ActTrainer(Trainer):

    def train_step(self):
        states, actions, raw_return, rtg, timesteps, attention_mask, pref = self.get_batch()
        
        action_target = torch.clone(actions)
        action_preds = self.model.forward(states)

        act_dim = self.get_batch.act_dim
        action_preds = action_preds.reshape(-1, act_dim)
        action_target = action_target[:,-1].reshape(-1, act_dim)

        # only action loss 
        loss = self.loss_fn(
            None, action_preds, None, None,
            None, action_target, None, None,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
        return loss.detach().cpu().item()
