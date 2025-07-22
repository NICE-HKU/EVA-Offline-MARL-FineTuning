import copy

import torch
import numpy as np
from .utils import sample, sample_with_imitate, padding_obs, padding_ava
from .utils import CPUManager


class RolloutWorker:

    def __init__(self, model, critic_model, frozen_actor_model,buffer, global_obs_dim, local_obs_dim, action_dim,trainer=None):
        self.buffer = buffer
        self.model = model
        self.critic_model = critic_model
        self.global_obs_dim = global_obs_dim
        self.local_obs_dim = local_obs_dim
        self.action_dim = action_dim
       
        self.episode_returns = []
        self.episode_win_rates = []
        self.trainer=trainer
        self.frozen_actor_model = frozen_actor_model
        

    
        self.device = 'cpu'
        if torch.cuda.is_available() and not isinstance(self.model, torch.nn.DataParallel):
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(model).to(self.device)
            self.critic_model = torch.nn.DataParallel(critic_model).to(self.device)
   
    def rollout(self, env, ret, train=True, random_rate=0.):
        self.model.train(False)
        self.critic_model.train(False)

        T_rewards, T_wins, steps, episode_dones = 0., 0., 0, np.zeros(env.n_threads)
        
        
        
        
        obs, share_obs, available_actions = env.real_env.reset()
        obs = padding_obs(obs, self.local_obs_dim)
        share_obs = padding_obs(share_obs, self.global_obs_dim)
        available_actions = padding_ava(available_actions, self.action_dim)

        # x: (n_threads, n_agent, context_lengrh, dim)
        global_states = torch.from_numpy(share_obs).to(self.device).unsqueeze(2)
        local_obss = torch.from_numpy(obs).to(self.device).unsqueeze(2)
        rtgs = np.ones((env.n_threads, env.num_agents, 1, 1)) * ret
        actions = np.zeros((env.n_threads, env.num_agents, 1, 1))
        timesteps = torch.zeros((env.n_threads * env.num_agents, 1, 1), dtype=torch.int64)
        t = 0

        while True:
            sampled_action, v_value = sample(self.model, self.critic_model, state=global_states.view(-1, np.shape(global_states)[2], np.shape(global_states)[3]),
                                             obs=local_obss.view(-1, np.shape(local_obss)[2], np.shape(local_obss)[3]), sample=train,
                                             actions=torch.tensor(actions, dtype=torch.int64).to(self.device).view(-1, np.shape(actions)[2], np.shape(actions)[3]),
                                             rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).view(-1, np.shape(rtgs)[2], np.shape(rtgs)[3]),
                                             timesteps=timesteps.to(self.device),
                                             available_actions=torch.from_numpy(available_actions).view(-1, np.shape(available_actions)[-1]))

            action = sampled_action.view((env.n_threads, env.num_agents, -1)).cpu().numpy()

            cur_global_obs = share_obs
            cur_local_obs = obs
            cur_ava = available_actions

            obs, share_obs, rewards, dones, infos, available_actions = env.real_env.step(action)
            
            
            obs = padding_obs(obs, self.local_obs_dim)
            share_obs = padding_obs(share_obs, self.global_obs_dim)
            available_actions = padding_ava(available_actions, self.action_dim)
            t += 1

            if train:
                v_value = v_value.view((env.n_threads, env.num_agents, -1)).cpu().numpy()
                self.buffer.insert(cur_global_obs, cur_local_obs, action, rewards, dones, cur_ava, v_value)

            for n in range(env.n_threads):
                if not episode_dones[n]:
                    steps += 1
                    T_rewards += np.mean(rewards[n])
                    if np.all(dones[n]):
                        episode_dones[n] = 1
                        if infos[n][0]['won']:
                            T_wins += 1.
                        
                       
            if np.all(episode_dones):
                break

            rtgs = np.concatenate([rtgs, np.expand_dims(rtgs[:, :, -1, :] - rewards, axis=2)], axis=2)
            global_state = torch.from_numpy(share_obs).to(self.device).unsqueeze(2)
            global_states = torch.cat([global_states, global_state], dim=2)
            local_obs = torch.from_numpy(obs).to(self.device).unsqueeze(2)
            local_obss = torch.cat([local_obss, local_obs], dim=2)
            actions = np.concatenate([actions, np.expand_dims(action, axis=2)], axis=2)
            timestep = t * torch.ones((env.n_threads * env.num_agents, 1, 1), dtype=torch.int64)
            timesteps = torch.cat([timesteps, timestep], dim=1)

        aver_return = T_rewards / env.n_threads
        aver_win_rate = T_wins / env.n_threads
        self.model.train(True)
        self.critic_model.train(True)
        return aver_return, aver_win_rate, steps
    def rollout_with_imitate(self, env, ret, train=True, random_rate=0.):
       
        self.model.train(False)
        self.critic_model.train(False)

        T_rewards, T_wins, steps, episode_dones = 0., 0., 0, np.zeros(env.n_threads)

        obs, share_obs, available_actions = env.real_env.reset()
        obs = padding_obs(obs, self.local_obs_dim)
        share_obs = padding_obs(share_obs, self.global_obs_dim)
        available_actions = padding_ava(available_actions, self.action_dim)

        # x: (n_threads, n_agent, context_lengrh, dim)
        global_states = torch.from_numpy(share_obs).to(self.device).unsqueeze(2)
        local_obss = torch.from_numpy(obs).to(self.device).unsqueeze(2)
        rtgs = np.ones((env.n_threads, env.num_agents, 1, 1)) * ret
        actions = np.zeros((env.n_threads, env.num_agents, 1, 1))
        timesteps = torch.zeros((env.n_threads * env.num_agents, 1, 1), dtype=torch.int64)
        t = 0

        while True:
           
            sampled_action, sampled_action_probs, sampled_action_frozen, sampled_action_probs_frozen, v_value = sample_with_imitate(self.model, self.frozen_actor_model, self.critic_model, state=global_states.view(-1, np.shape(global_states)[2], np.shape(global_states)[3]),
                                             obs=local_obss.view(-1, np.shape(local_obss)[2], np.shape(local_obss)[3]), sample=train,
                                             actions=torch.tensor(actions, dtype=torch.int64).to(self.device).view(-1, np.shape(actions)[2], np.shape(actions)[3]),
                                             rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).view(-1, np.shape(rtgs)[2], np.shape(rtgs)[3]),
                                             timesteps=timesteps.to(self.device),
                                             available_actions=torch.from_numpy(available_actions).view(-1, np.shape(available_actions)[-1]))

            # 
            action_dynamic = sampled_action.view((env.n_threads, env.num_agents, -1)).cpu().numpy()
            # Get the frozen model's action
            action_frozen = sampled_action_frozen.view((env.n_threads, env.num_agents, -1)).cpu().numpy()
            # Reshape the probabilities for comparison
            probs_main = sampled_action_probs.view((env.n_threads, env.num_agents, -1))
            probs_frozen = sampled_action_probs_frozen.view((env.n_threads, env.num_agents, -1))
            # Create masks for where frozen action is better
            main_action_probs = torch.gather(probs_main, 2, sampled_action.view((env.n_threads, env.num_agents, -1)))
            frozen_action_probs = torch.gather(probs_frozen, 2,
                                               sampled_action_frozen.view((env.n_threads, env.num_agents, -1)))
            
            better_frozen_mask = (frozen_action_probs > main_action_probs).cpu().numpy()  
            # # Combine actions

            frozen_better_count = np.sum(better_frozen_mask, axis=1)  
            total_agents = better_frozen_mask.shape[1]
            #action_infused = np.where(better_frozen_mask, action_frozen, action_frozen)
            use_frozen_env = frozen_better_count > (total_agents / 2)
            action_infused = np.where(use_frozen_env[:, np.newaxis, :], action_frozen, action_dynamic)
            
            
            cur_global_obs = share_obs
            cur_local_obs = obs
            cur_ava = available_actions

            obs, share_obs, rewards, dones, infos, available_actions = env.real_env.step(action_infused)
            obs = padding_obs(obs, self.local_obs_dim)
            share_obs = padding_obs(share_obs, self.global_obs_dim)
            available_actions = padding_ava(available_actions, self.action_dim)
            t += 1

            if train:
                v_value = v_value.view((env.n_threads, env.num_agents, -1)).cpu().numpy()
                self.buffer.insert(cur_global_obs, cur_local_obs, action_infused, rewards, dones, cur_ava, v_value)

            for n in range(env.n_threads):
                if not episode_dones[n]:
                    steps += 1
                    T_rewards += np.mean(rewards[n])
                    if np.all(dones[n]):
                        episode_dones[n] = 1
                        if infos[n][0]['won']:
                            T_wins += 1.
            if np.all(episode_dones):
                break

            rtgs = np.concatenate([rtgs, np.expand_dims(rtgs[:, :, -1, :] - rewards, axis=2)], axis=2)
            global_state = torch.from_numpy(share_obs).to(self.device).unsqueeze(2)
            global_states = torch.cat([global_states, global_state], dim=2)
            local_obs = torch.from_numpy(obs).to(self.device).unsqueeze(2)
            local_obss = torch.cat([local_obss, local_obs], dim=2)
            actions = np.concatenate([actions, np.expand_dims(action_infused, axis=2)], axis=2)
            timestep = t * torch.ones((env.n_threads * env.num_agents, 1, 1), dtype=torch.int64)
            timesteps = torch.cat([timesteps, timestep], dim=1)

        aver_return = T_rewards / env.n_threads
        aver_win_rate = T_wins / env.n_threads
        self.model.train(True)
        self.critic_model.train(True)
        return aver_return, aver_win_rate, steps