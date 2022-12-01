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

<<<<<<< HEAD
class OurSampler(object):
=======
class OAC(object):
>>>>>>> fd85c4bafbe88cdeb3f46d4f9a2088121d1be844

    def __init__(self, env, max_traj_length=1000, qf1=None, qf2=None, device='cuda'):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._current_observation = self.env.reset()
        self.qf1 = qf1
        self.qf2 = qf2
        self.device = device
<<<<<<< HEAD
=======
        self.n_samples = 10
        self.n_ensemble_samples = 10
        self.std = 0.1
>>>>>>> fd85c4bafbe88cdeb3f46d4f9a2088121d1be844

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
            
            next_action = torch.zeros_like(action_tensor)
            max_diff = 0
            for _ in range(self.n_samples):
                new_action = action_tensor + torch.randn(action_tensor.shape, device=action_tensor.device) * self.std
                qf1_vals = self.qf1(observation_tensor, new_action)
                qf2_vals = self.qf2(observation_tensor, new_action)
                action_q_diff = torch.abs(qf1_vals - qf2_vals)
                if action_q_diff > max_diff:
                    max_diff = action_q_diff
                    next_action = new_action

            next_observation, reward, done, _ = self.env.step(next_action.detach().cpu().numpy())

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
        return d

    @property
    def env(self):
        return self._env


class OurSampler(object):
    
    def __init__(self, env, max_traj_length=1000, qf1=None, qf2=None, qf3=None, qf4=None, device='cuda'):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._current_observation = self.env.reset()
        self.qf1 = qf1
        self.qf2 = qf2
        self.qf3 = qf3
        self.qf4 = qf4
        self.device = device
        self.n_samples = 10
        self.n_ensemble_samples = 10
        self.std = 0.1

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
            
            next_action = torch.zeros_like(action_tensor)
            max_diff = 0
            for _ in range(self.n_samples):
                new_action = action_tensor + torch.randn(action_tensor.shape, device=action_tensor.device) * self.std
                qf1_vals = self.qf1(observation_tensor, new_action)
                qf2_vals = self.qf2(observation_tensor, new_action)
                qf3_vals = self.qf3(observation_tensor, new_action)
                qf4_vals = self.qf4(observation_tensor, new_action)
                action_q_diff = 0.5 * torch.abs(qf1_vals + qf2_vals - qf3_vals - qf4_vals)
                if action_q_diff > max_diff:
                    max_diff = action_q_diff
                    next_action = new_action

            next_observation, reward, done, _ = self.env.step(next_action.detach().cpu().numpy())

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
<<<<<<< HEAD
        action_q_diffs = np.array(action_q_diffs)
        indices = np.argsort(action_q_diffs)
        k = len(actions) // 10 # take ~10 %
        top_k_indices = indices[-1 * k :]
        for k in d.keys():
            d[k] = d[k][top_k_indices]
=======
>>>>>>> fd85c4bafbe88cdeb3f46d4f9a2088121d1be844
        return d

    @property
    def env(self):
        return self._env


class EnsembleSampler(object):
<<<<<<< HEAD

=======
    
>>>>>>> fd85c4bafbe88cdeb3f46d4f9a2088121d1be844
    def __init__(self, env, max_traj_length=1000, qf1=None, qf2=None, device='cuda'):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._current_observation = self.env.reset()
        self.qf1 = qf1
        self.qf2 = qf2
        self.device = device
<<<<<<< HEAD
=======
        self.n_samples = 10
        self.n_ensemble_samples = 10
        self.std = 0.1
>>>>>>> fd85c4bafbe88cdeb3f46d4f9a2088121d1be844

    def sample(self, policy, n_steps, deterministic=False, replay_buffer=None):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
<<<<<<< HEAD
        action_q_ensembles = []
=======
        action_q_diffs = []
>>>>>>> fd85c4bafbe88cdeb3f46d4f9a2088121d1be844

        for _ in range(n_steps):

            self._traj_steps += 1
            observation = self._current_observation
            action = policy(
                np.expand_dims(observation, 0), deterministic=deterministic
            )[0, :]

            observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
<<<<<<< HEAD

            action_q_ensemble_vals = []
            for i in range(10):
                qf1_vals = self.qf1(observation_tensor, action_tensor)
                qf2_vals = self.qf2(observation_tensor, action_tensor)
                action_q_ensemble = (qf1_vals.item() + qf2_vals.item()) / 2
                action_q_ensemble_vals.append(action_q_ensemble)
            action_q_ensemble = np.std(action_q_ensemble_vals)

            next_observation, reward, done, _ = self.env.step(action)
=======
            
            next_action = torch.zeros_like(action_tensor)
            max_diff = 0
            for _ in range(self.n_samples):
                new_action = action_tensor + torch.randn(action_tensor.shape, device=action_tensor.device) * self.std
                qf1_vals = self.qf1(observation_tensor, new_action)
                qf2_vals = self.qf2(observation_tensor, new_action)
                all_actions = []
                for t in range(self.n_ensemble_samples):
                    all_actions.append(0.5 * (qf1_vals + qf2_vals))
                    
                action_q_diff = torch.var(torch.stack(all_actions, 0), 0)
                if action_q_diff > max_diff:
                    max_diff = action_q_diff
                    next_action = new_action

            next_observation, reward, done, _ = self.env.step(next_action.detach().cpu().numpy())
>>>>>>> fd85c4bafbe88cdeb3f46d4f9a2088121d1be844

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)
<<<<<<< HEAD
            action_q_ensembles.append(action_q_ensemble.item())
=======
            action_q_diffs.append(action_q_diff.item())
>>>>>>> fd85c4bafbe88cdeb3f46d4f9a2088121d1be844

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
<<<<<<< HEAD
        action_q_ensembles = np.array(action_q_ensembles)
        indices = np.argsort(action_q_ensembles)
        k = len(actions) // 10 # take ~10 %
        top_k_indices = indices[-1 * k :]
        for k in d.keys():
            d[k] = d[k][top_k_indices]
=======
>>>>>>> fd85c4bafbe88cdeb3f46d4f9a2088121d1be844
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
