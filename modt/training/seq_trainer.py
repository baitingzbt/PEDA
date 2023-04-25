import numpy as np
import torch
from modt.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, raw_return, rtg, timesteps, attention_mask, pref = self.get_batch()
        rtg = rtg[:, :-1]
        
        action_target = torch.clone(actions)
        return_target = torch.clone(raw_return)
        pref_target = torch.clone(pref)

        
        if self.concat_rtg_pref != 0:
            rtg = torch.cat((rtg, torch.cat([pref] * self.concat_rtg_pref, dim=2)), dim=2)
        if self.concat_act_pref != 0:
            actions = torch.cat((actions, torch.cat([pref] * self.concat_act_pref, dim=2)), dim=2)

        
        action_preds, return_preds, pref_preds = self.model.forward(
            states, actions, rtg, pref, timesteps, attention_mask=attention_mask
        )

        act_dim = self.get_batch.act_dim
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        pref_dim = self.get_batch.pref_dim
        return_preds = return_preds.reshape(-1, pref_dim)[attention_mask.reshape(-1) > 0]
        return_target = return_target.reshape(-1, pref_dim)[attention_mask.reshape(-1) > 0]
        
        pref_preds = pref_preds.reshape(-1, pref_dim)[attention_mask.reshape(-1) > 0]
        pref_target = pref_target.reshape(-1, pref_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, return_preds, pref_preds,
            None, action_target, return_target, pref_target,
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
            self.diagnostics['training/return_error'] = torch.mean((return_preds - return_target) ** 2).detach().cpu().item()
            self.diagnostics['training/pref_error'] = torch.mean((pref_preds - pref_target) ** 2).detach().cpu().item()
            
        return loss.detach().cpu().item()
