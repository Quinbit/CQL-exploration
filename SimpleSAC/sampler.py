import numpy as np
import torch

class StepSampler(object):

    def __init__(self, env, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._current_observation = self.env.reset()

    def sample(self, policy, n_steps, deterministic=False, replay_buffer=None):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for _ in range(n_steps):
            self._traj_steps += 1
            observation = self._current_observation
            action = policy(
                np.expand_dims(observation, 0), deterministic=deterministic
            )[0, :]
            next_observation, reward, done, _ = self.env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)

            if replay_buffer is not None:
                replay_buffer.add_sample(
                    observation, action, reward, next_observation, done
                )

            self._current_observation = next_observation

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                self._current_observation = self.env.reset()

        return dict(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )

    @property
    def env(self):
        return self._env

class OurSampler(object):

    def __init__(self, env, max_traj_length=1000, qf1=None, qf2=None, device='cuda'):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._current_observation = self.env.reset()
        self.qf1 = qf1
        self.qf2 = qf2
        self.device = device

    def sample(self, policy, n_steps, deterministic=False, replay_buffer=None):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        action_q_diffs = []

        for _ in range(n_steps):

            self._traj_steps += 1
            observation = self._current_observation
            action = policy(
                np.expand_dims(observation, 0), deterministic=deterministic
            )[0, :]

            observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
            qf1_vals = self.qf1(observation_tensor, action_tensor)
            qf2_vals = self.qf2(observation_tensor, action_tensor)
            action_q_diff = torch.abs(qf1_vals - qf2_vals)

            next_observation, reward, done, _ = self.env.step(action)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)
            action_q_diffs.append(action_q_diff.item())

            if replay_buffer is not None:
                replay_buffer.add_sample(
                    observation, action, reward, next_observation, done
                )

            self._current_observation = next_observation

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                self._current_observation = self.env.reset()

        d = dict(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )
        action_q_diffs = np.array(action_q_diffs)
        indices = np.argsort(action_q_diffs)
        k = len(actions) // 10 # take ~10 %
        top_k_indices = indices[-1 * k :]
        for k in d.keys():
            d[k] = d[k][top_k_indices]
        return d

    @property
    def env(self):
        return self._env


class EnsembleSampler(object):

    def __init__(self, env, max_traj_length=1000, qf1=None, qf2=None, device='cuda'):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._current_observation = self.env.reset()
        self.qf1 = qf1
        self.qf2 = qf2
        self.device = device

    def sample(self, policy, n_steps, deterministic=False, replay_buffer=None):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        action_q_ensembles = []

        for _ in range(n_steps):

            self._traj_steps += 1
            observation = self._current_observation
            action = policy(
                np.expand_dims(observation, 0), deterministic=deterministic
            )[0, :]

            observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
            action_q_ensemble_vals = []
            for i in range(10):
                qf1_vals = self.qf1(observation_tensor, action_tensor)
                qf2_vals = self.qf2(observation_tensor, action_tensor)
                action_q_ensemble = (qf1_vals.item() + qf2_vals.item()) / 2
                action_q_ensemble_vals.append(action_q_ensemble)
            action_q_ensemble = np.std(action_q_ensemble_vals)

            next_observation, reward, done, _ = self.env.step(action)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)
            action_q_ensembles.append(action_q_ensemble.item())

            if replay_buffer is not None:
                replay_buffer.add_sample(
                    observation, action, reward, next_observation, done
                )

            self._current_observation = next_observation

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                self._current_observation = self.env.reset()

        d = dict(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )
        action_q_ensembles = np.array(action_q_ensembles)
        indices = np.argsort(action_q_ensembles)
        k = len(actions) // 10 # take ~10 %
        top_k_indices = indices[-1 * k :]
        for k in d.keys():
            d[k] = d[k][top_k_indices]
        return d

    @property
    def env(self):
        return self._env


class TrajSampler(object):

    def __init__(self, env, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env

    def sample(self, policy, n_trajs, deterministic=False, replay_buffer=None, display=False):
        trajs = []
        for _ in range(n_trajs):
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []

            observation = self.env.reset()

            for _ in range(self.max_traj_length):
                action = policy(
                    np.expand_dims(observation, 0), deterministic=deterministic
                )[0, :]
                next_observation, reward, done, _ = self.env.step(action)
                if display:
                    self.env.render()
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_observations.append(next_observation)

                if replay_buffer is not None:
                    replay_buffer.add_sample(
                        observation, action, reward, next_observation, done
                    )

                observation = next_observation

                if done:
                    break

            trajs.append(dict(
                observations=np.array(observations, dtype=np.float32),
                actions=np.array(actions, dtype=np.float32),
                rewards=np.array(rewards, dtype=np.float32),
                next_observations=np.array(next_observations, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
            ))

        return trajs

    @property
    def env(self):
        return self._env
