import torch
from torch import nn, Tensor


def compute_returns(self, rewards, values, dones, last_values, gamma, lam):
    advantage = 0
    returns = torch.zeros_like(values)
    for step in reversed(range(self.num_transitions_per_env)):
        if step == self.num_transitions_per_env - 1:
            next_values = last_values
        else:
            next_values = values[step + 1]
        next_is_not_terminal = 1.0 - dones[step].float()
        delta = (
            rewards[step] + next_is_not_terminal * gamma * next_values - values[step]
        )
        advantage = delta + next_is_not_terminal * gamma * lam * advantage
        returns[step] = advantage + values[step]

    # Compute and normalize the advantages
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


class RolloutStorage(nn.Module):
    class Transition:
        def __init__(self):
            self.actor_obs = None
            self.critic_obs = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    class MultiTransitions:
        def __init__(self, keys):
            self.keys = keys
            for key in keys:
                setattr(self, "transition_" + key, RolloutStorage.Transition())

        def clear(self):
            for key in self.keys:
                getattr(self, "transition_" + key).clear()

    def __init__(self, num_envs, num_transitions_per_env, device="cpu"):

        super().__init__()

        self.device = device

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        # self.saved_hidden_states_a = None
        # self.saved_hidden_states_c = None

        self.step = 0
        self.stored_keys = list()

    def register_key(self, key: str, shape=(), dtype=torch.float):
        # This class was partially copied from https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/agents/utils/data_utils.py
        assert not hasattr(self, key), key
        assert isinstance(shape, (list, tuple)), "shape must be a list or tuple"
        buffer = torch.zeros(
            (self.num_transitions_per_env, self.num_envs) + shape,
            dtype=dtype,
            device=self.device,
        )
        self.register_buffer(key, buffer, persistent=False)
        self.stored_keys.append(key)

    def add_transitions(
        self, transition: Transition, transition_sym: Transition = None
    ):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        if transition_sym is not None:
            self.update_key(
                "actor_obs",
                torch.cat((transition.actor_obs, transition_sym.actor_obs), dim=0),
            )
            self.update_key(
                "critic_obs",
                torch.cat((transition.critic_obs, transition_sym.critic_obs), dim=0),
            )
            self.update_key(
                "actions",
                torch.cat((transition.actions, transition_sym.actions), dim=0),
            )
            self.update_key(
                "rewards",
                torch.cat(
                    (
                        transition.rewards.view(-1, 1),
                        transition_sym.rewards.view(-1, 1),
                    ),
                    dim=0,
                ),
            )
            self.update_key(
                "dones",
                torch.cat(
                    (transition.dones.view(-1, 1), transition_sym.dones.view(-1, 1)),
                    dim=0,
                ),
            )
            self.update_key(
                "values", torch.cat((transition.values, transition_sym.values), dim=0)
            )
            self.update_key(
                "actions_log_prob",
                torch.cat(
                    (
                        transition.actions_log_prob.view(-1, 1),
                        transition_sym.actions_log_prob.view(-1, 1),
                    ),
                    dim=0,
                ),
            )
            self.update_key(
                "action_mean",
                torch.cat((transition.action_mean, transition_sym.action_mean), dim=0),
            )
            self.update_key(
                "action_sigma",
                torch.cat(
                    (transition.action_sigma, transition_sym.action_sigma), dim=0
                ),
            )
        else:
            self.update_key("actor_obs", transition.actor_obs)
            self.update_key("critic_obs", transition.critic_obs)
            self.update_key("actions", transition.actions)
            self.update_key("rewards", transition.rewards.view(-1, 1))
            self.update_key("dones", transition.dones.view(-1, 1))
            self.update_key("values", transition.values)
            self.update_key("actions_log_prob", transition.actions_log_prob.view(-1, 1))
            self.update_key("action_mean", transition.action_mean)
            self.update_key("action_sigma", transition.action_sigma)
        self.increment_step()

    def add_multi_transitions(self, transitions: MultiTransitions):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        for key in transitions.keys:
            transition = getattr(transitions, "transition" + "_" + key)
            self.update_key("actor_obs", transition.actor_obs)
            self.update_key("critic_obs", transition.critic_obs)
            self.update_key("actions_" + key, transition.actions)
            self.update_key("rewards_" + key, transition.rewards.view(-1, 1))
            self.update_key("dones_" + key, transition.dones.view(-1, 1))
            self.update_key("values_" + key, transition.values)
            self.update_key(
                "actions_log_prob_" + key, transition.actions_log_prob.view(-1, 1)
            )
            self.update_key("action_mean_" + key, transition.action_mean)
            self.update_key("action_sigma_" + key, transition.action_sigma)
        self.increment_step()

    def increment_step(self):
        self.step += 1

    def update_key(self, key: str, data: Tensor):
        # This class was partially copied from https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/agents/utils/data_utils.py
        assert not data.requires_grad
        assert self.step < self.num_transitions_per_env, "Rollout buffer overflow"
        getattr(self, key)[self.step].copy_(data)

    def batch_update_data(self, key: str, data: Tensor):
        # This class was partially copied from https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/agents/utils/data_utils.py
        assert not data.requires_grad
        getattr(self, key)[:] = data
        # self.store_dict[key] += self.total_sum()

    def _save_hidden_states(self, hidden_states):
        assert NotImplementedError
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = (
            hidden_states[0]
            if isinstance(hidden_states[0], tuple)
            else (hidden_states[0],)
        )
        hid_c = (
            hidden_states[1]
            if isinstance(hidden_states[1], tuple)
            else (hidden_states[1],)
        )

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(
                    self.observations.shape[0], *hid_a[i].shape, device=self.device
                )
                for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(
                    self.observations.shape[0], *hid_c[i].shape, device=self.device
                )
                for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def get_statistics(self):
        raise NotImplementedError
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (
                flat_dones.new_tensor([-1], dtype=torch.int64),
                flat_dones.nonzero(as_tuple=False)[:, 0],
            )
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def query_key(self, key: str):
        assert hasattr(self, key), key
        return getattr(self, key)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        _buffer_dict = {
            key: getattr(self, key)[:].flatten(0, 1) for key in self.stored_keys
        }

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                _batch_buffer_dict = {
                    key: _buffer_dict[key][batch_idx] for key in self.stored_keys
                }
                yield _batch_buffer_dict
