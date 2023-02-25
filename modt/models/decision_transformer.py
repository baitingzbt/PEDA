from logging import warning
import numpy as np
import torch
import torch.nn as nn

import transformers
from modt.models.model import TrajectoryModel
from modt.models.trajectory_gpt2 import GPT2Model


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        pref_dim,
        rtg_dim,
        hidden_size,
        act_scale,
        use_pref=False,
        concat_state_pref=0,
        concat_rtg_pref=0,
        concat_act_pref=0,
        max_length=None,
        eval_context_length=None,
        max_ep_len=4096,
        action_tanh=True,
        *args,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, pref_dim, max_length=max_length)
        
        self.hidden_size = hidden_size
        self.use_pref = use_pref
        self.act_scale = act_scale
        self.concat_state_pref = concat_state_pref
        self.concat_rtg_pref = concat_rtg_pref
        self.concat_act_pref = concat_act_pref
        self.eval_context_length = eval_context_length
        
        self.rtg_dim = rtg_dim + concat_rtg_pref * pref_dim
        self.act_dim = act_dim + concat_act_pref * pref_dim
        
        self.init_temperature=0.1
        
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        # return and preference should have the same dimension for linear preference
        self.embed_return = torch.nn.Linear(self.rtg_dim, hidden_size)
        self.embed_pref = torch.nn.Linear(self.pref_dim, hidden_size, bias=False)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        
        # note: we don't predict return nor pref for the paper
        # but you can try to see if training these jointly can improve stability
        self.predict_return = nn.Linear(hidden_size * 2, self.pref_dim)
        self.predict_pref = nn.Sequential(
            *([nn.Linear(hidden_size * 2, self.pref_dim), nn.Softmax(dim=2)])
        )


    def forward(self, states, actions, returns_to_go, pref, timesteps, attention_mask=None):
 
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)
        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        
        
        # time embeddings are treated similar to positional embeddings
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings



        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        
        
        stacked_inputs = self.embed_ln(stacked_inputs)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        
        x = transformer_outputs['last_hidden_state']


        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), actions (2), or pref (3); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions, return and pref predictions not used in model 
        return_preds = self.predict_return(torch.cat((x[:, 1], x[:, 2]), dim=2)) 
        pref_preds = self.predict_pref(torch.cat((x[:, 0], x[:, 2]), dim=2))
        # as default, we don't have a separate pref head, they are concat to other inputs
        # default self.use_pref = False
        # when using pref as a separate head, you can try adding pref head output to predict action
        if self.use_pref:
            # predict next action given state and preference
            action_preds = self.predict_action(torch.cat((x[:, 1], x[:, 3]), dim=2))
        else:
            action_preds = self.predict_action(x[:, 1])
        return action_preds, return_preds, pref_preds

    def get_action(self, states, actions, returns_to_go, pref, timesteps, **kwargs):

        if self.concat_rtg_pref != 0:
            returns_to_go = torch.cat((returns_to_go, torch.cat([pref] * self.concat_rtg_pref, dim=1)), dim=1)
        if self.concat_act_pref != 0:
            actions = torch.cat((actions, torch.cat([pref] * self.concat_act_pref, dim=1)), dim=1)

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, self.rtg_dim)
        pref = pref.reshape(1, -1, self.pref_dim)
        timesteps = timesteps.reshape(1, -1)
        
        if self.eval_context_length is not None:
            states = states[:, -self.eval_context_length:]
            actions = actions[:, -self.eval_context_length:]
            returns_to_go = returns_to_go[:, -self.eval_context_length:]
            pref = pref[:, -self.eval_context_length:]
            timesteps = timesteps[:, -self.eval_context_length:]
            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.eval_context_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)

            states = torch.cat(
                [torch.zeros((states.shape[0], self.eval_context_length - states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)

            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.eval_context_length - actions.shape[1], self.act_dim), device=actions.device), actions],
                dim=1).to(dtype=torch.float32)

            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.eval_context_length - returns_to_go.shape[1], self.rtg_dim), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            
            pref = torch.cat(
                [torch.zeros((pref.shape[0], self.eval_context_length - pref.shape[1], self.pref_dim), device=pref.device), pref],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.eval_context_length - timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        action_preds, return_preds, pref_preds = self.forward(
            states, actions, returns_to_go, pref, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0, -1]
