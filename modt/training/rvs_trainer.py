from audioop import avg
import numpy as np
import torch
from modt.training.trainer import Trainer


class RVSTrainer(Trainer):

    def train_step(self):
        states, actions, raw_return, avg_rtg, timesteps, attention_mask, pref = self.get_batch()
        states = torch.squeeze(states)
        actions = torch.squeeze(actions)
        avg_rtg = torch.squeeze(avg_rtg)
        if len(avg_rtg.shape) == 1:
            avg_rtg = torch.unsqueeze(avg_rtg, dim=-1)

        states = torch.cat((states, avg_rtg), dim=1)

        loss = self.model.training_step(
            (states, actions),
            batch_idx=0 # doesn't matter in source code
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()