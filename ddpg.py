import torch
import utils

class DDPG(object):
    def __init__(self, q_net, target_q_net, policy_net, target_policy_net, tau, device='cpu'):
        self._q_net = q_net
        self._target_q_net = target_q_net
        self._policy_net = policy_net
        self._target_policy_net = target_policy_net

        self._device = device

        self._discount = 0.95
        self._policy_learning_rate = 1e-3
        self._q_learning_rate = 1e-3
        self._target_update_tau = tau

        self._policy_optimizer, self._q_optimizer = get_optimizer(policy=self._policy_net, value=self._q_net)

        soft_update_from_to(
            source=self._policy_net ,
            target=self._target_policy_net,
            tau=1.0
        )
        soft_update_from_to(
            source=self._q_net ,
            target=self._target_q_net,
            tau=1.0
        )

    def train(self, replay):
        batch_size = 64
        batch = replay.random_batch(batch_size)
        batch = utils.batch_to_torch(batch, device=self._device)
        rewards = batch['rewards']
        terminals = batch['terminals']
        states = batch['observations']
        actions = batch['actions']
        states_t = batch['next_observations']

        # ''''

        # predict Q': s' -> |ActorNet'| -> a' -> | QNet'| -> Q'
        actions_dot = self._target_policy_net(states_t)
        Q_dot = self._target_q_net(states_t,actions_dot)

        # predict Q: (s,a) ->  | QNet| -> Q
        Q = self._q_net(states,actions)

        # compute loss
        optimizer = torch.optim.Adam(self._q_net.parameters(), lr=self._q_learning_rate)
        criterion = torch.nn.MSELoss()
        loss = criterion(Q,rewards+self._discount*Q_dot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update action net
        optimizer = torch.optim.Adam(self._policy_net.parameters(), lr=self._policy_learning_rate)
        optimizer.zero_grad()
        policy_loss = -self._q_net(states, self._policy_net(states))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        optimizer.step()

        # ''''

        # Target network updates
        soft_update_from_to(
            source=self._policy_net ,
            target=self._target_policy_net,
            tau=self._target_update_tau
        )
        soft_update_from_to(
            source=self._q_net ,
            target=self._target_q_net,
            tau=self._target_update_tau
        )

def get_optimizer(policy, value):
  policy_optimizer = torch.optim.Adam( list(policy.parameters()), lr=1e-4)
  value_optimizer = torch.optim.Adam(list(value.parameters()), lr=1e-3, weight_decay = 1e-3)

  return policy_optimizer, value_optimizer

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
