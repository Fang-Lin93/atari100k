
import torch
from torch import nn

import typing  # used for type hint
from typing import List  # used for type hint


class NetworkOutput(typing.NamedTuple):
    # output format of the model
    value: float
    value_prefix: float
    policy_logits: List[float]
    hidden_state: List[float]
    reward_hidden: object


class BaseNet(nn.Module):
    def __init__(self, inverse_value_transform, inverse_reward_transform, lstm_hidden_size):
        """Base Network
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        inverse_value_transform: Any
            A function that maps value supports into value scalars
        inverse_reward_transform: Any
            A function that maps reward supports into value scalars
        lstm_hidden_size: int
            dim of lstm hidden
        """
        super(BaseNet, self).__init__()
        self.inverse_value_transform = inverse_value_transform
        self.inverse_reward_transform = inverse_reward_transform
        self.lstm_hidden_size = lstm_hidden_size

    def prediction(self, state):
        raise NotImplementedError

    def representation(self, obs_history):
        raise NotImplementedError

    def dynamics(self, state, reward_hidden, action):
        raise NotImplementedError

    def initial_inference(self, obs) -> NetworkOutput:
        num = obs.size(0)

        state = self.representation(obs)
        actor_logit, value = self.prediction(state)

        if not self.training:
            # if not in training, obtain the scalars of the value/reward
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            state = state.detach().cpu().numpy()
            actor_logit = actor_logit.detach().cpu().numpy()
            # zero initialization for reward (value prefix) hidden states
            reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size).detach().cpu().numpy(),
                             torch.zeros(1, num, self.lstm_hidden_size).detach().cpu().numpy())
        else:
            # zero initialization for reward (value prefix) hidden states
            # TODO
            # reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size).to('cuda'),
            #                  torch.zeros(1, num, self.lstm_hidden_size).to('cuda'))
            reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size),
                             torch.zeros(1, num, self.lstm_hidden_size))

        return NetworkOutput(value, [0. for _ in range(num)], actor_logit, state, reward_hidden)

    def recurrent_inference(self, hidden_state, reward_hidden, action) -> NetworkOutput:
        state, reward_hidden, value_prefix = self.dynamics(hidden_state, reward_hidden, action)
        actor_logit, value = self.prediction(state)

        if not self.training:
            # if not in training, obtain the scalars of the value/reward
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            value_prefix = self.inverse_reward_transform(value_prefix).detach().cpu().numpy()
            state = state.detach().cpu().numpy()
            reward_hidden = (reward_hidden[0].detach().cpu().numpy(), reward_hidden[1].detach().cpu().numpy())
            actor_logit = actor_logit.detach().cpu().numpy()

        return NetworkOutput(value, value_prefix, actor_logit, state, reward_hidden)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


def renormalize(tensor, first_dim=1):
    # normalize the tensor (states)
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min) / (max - min)

    return flat_tensor.view(*tensor.shape)