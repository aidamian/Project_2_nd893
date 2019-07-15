
import torch.optim as optim
import torch.nn.functional as F

class Agent():
    """
    Implements DDPG TD3 approach with a few tricks
    """
    def __init__(self, a_size, s_size, dev, 
                 n_env_agents,
                 GAMMA=0.99, 
                 TAU=5e-3, 
                 policy_noise=0.2, 
                 exploration_noise=0.1, 
                 noise_clip=0.5,
                 LR_CRITIC=1e-3, 
                 LR_ACTOR=1e-3, 
                 WEIGHT_DECAY=0, 
                 policy_freq=2, 
                 random_seed=1234,
                 BUFFER_SIZE=int(1e5), 
                 BATCH_SIZE=128, 
                 RANDOM_WARM_UP=1024,
                 name='agent',
                 simplified_critic=False,
                 critic_use_state_bn=True,
                 critic_use_other_bn=True,
                 actor_use_pre_bn=False,
                 actor_use_post_bn=True,
                ):
        self.a_size = a_size
        self.n_env_agents = n_env_agents
        self.name = name
        self.s_size = s_size
        self.MIN_EXPL_NOISE = 0.03
        self.MIN_POLI_NOISE = 0.05
        self.dev = dev
        self.device = dev
        self.GAMMA = GAMMA
        self.TAU = 0.05
        self.RANDOM_WARM_UP = RANDOM_WARM_UP
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.BATCH_SIZE = BATCH_SIZE
        if simplified_critic:
            critic_state_layers = []
        else:
            critic_state_layers = [256]
            
            
            
        self.actor_online = Actor(input_size=self.s_size,
                                  output_size=self.a_size, 
                                  use_pre_bn=actor_use_pre_bn,
                                  use_post_bn=actor_use_post_bn,
                                 ).to(self.dev)
        self.actor_target = Actor(input_size=self.s_size, 
                                  output_size=self.a_size, 
                                  use_pre_bn=actor_use_pre_bn,
                                  use_post_bn=actor_use_post_bn,
                                 ).to(self.dev)
        self.actor_target.load_state_dict(self.actor_online.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_online.parameters(), lr=LR_ACTOR)
        
        self.critic_online_1 = Critic(state_size=self.s_size, 
                                      act_size=self.a_size, 
                                      state_layers=critic_state_layers, 
                                      state_bn=critic_use_state_bn, 
                                      other_bn=critic_use_other_bn).to(self.dev)
        self.critic_target_1 = Critic(state_size=self.s_size, 
                                      act_size=self.a_size, 
                                      state_layers=critic_state_layers, 
                                      state_bn=critic_use_state_bn, 
                                      other_bn=critic_use_other_bn).to(self.dev)
        self.critic_target_1.load_state_dict(self.critic_online_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_online_1.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        self.critic_online_2 = Critic(state_size=self.s_size, 
                                      act_size=self.a_size, 
                                      state_layers=critic_state_layers, 
                                      state_bn=critic_use_state_bn, 
                                      other_bn=critic_use_other_bn).to(self.dev)
        self.critic_target_2 = Critic(state_size=self.s_size, 
                                      act_size=self.a_size, 
                                      state_layers=critic_state_layers, 
                                      state_bn=critic_use_state_bn, 
                                      other_bn=critic_use_other_bn).to(self.dev)
        self.critic_target_2.load_state_dict(self.critic_online_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_online_2.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        self.memory = ReplayBuffer(self.a_size, BUFFER_SIZE, BATCH_SIZE, random_seed, dev=self.dev)
        self.step_counter = 0
        self.steps_to_train_counter = 0
        self.skip_update_timer = 0
        self.train_iters = 0
        self.actor_updates = 0
        self.critic_1_losses = deque(maxlen=100)
        self.critic_2_losses = deque(maxlen=100)
        self.actor_losses = deque(maxlen=100)
        print("Agent '{}' initialized with the following parameters:".format(self.name))
        print("   Env agents:   {}".format(self.n_env_agents))
        print("   Explor noise: {:>6.4f}".format(self.exploration_noise))
        print("   Policy noise: {:>6.4f}".format(self.policy_noise))
        print("   Warm-up size: {:>6}".format(self.RANDOM_WARM_UP))
        print("Actor DAG:\n{}\nCritic DAG:\n{}".format(self.actor_online, self.critic_online_1))
        
        return
    

    def step(self, states, actions, rewards, next_states, dones, train_every_steps):
        """Save experience in replay memory. train if required
        The approach consists in letting the policy run "free" for `train_every_steps` steps and then train another 
        `train_every_steps` at each step.
        """
        # Save experience / reward
        self.step_counter += 1
        for _a in range(self.n_env_agents):
            self.memory.add(states[_a], actions[_a], rewards[_a], next_states[_a], dones[_a])
        
        if not self.is_warming_up():
            if self.steps_to_train_counter > 0:
                self.train(nr_iters=1)
                self.steps_to_train_counter -= 1
                self.skip_update_timer = 0
            else:
                self.skip_update_timer += 1

            if self.skip_update_timer >= train_every_steps:
                self.steps_to_train_counter = train_every_steps # // 2 # only half training
                self.skip_update_timer = 0            
        return
    
    def reduce_explore_noise(self, down_scale=0.9):
        self.exploration_noise = max(self.MIN_EXPL_NOISE, self.exploration_noise * down_scale)
        print("\nNew explor noise: {:.4f}".format(self.exploration_noise))
        return
    
    
    def reduce_policy_noise(self, down_scale=0.9):
        self.policy_noise = max(self.MIN_POLI_NOISE, self.policy_noise * down_scale)
        print("\nNew policy noise: {:.4f}".format(self.policy_noise))
        return
        
        
    def reduce_noise(self, down_scale):
        self.reduce_explore_noise(down_scale)
        self.reduce_policy_noise(down_scale)
        return
    
    def clear_policy_noise(self):
        if self.policy_noise != 0:
            self.policy_noise = 0
            print("\nPolicy noise stopped!")
        return
    
    def clear_explore_noise(self):
        if self.exploration_noise != 0:
            self.exploration_noise = 0
            print("\nExploration noise stopped")
        return


    def train(self, nr_iters):
        """ use random sample from buffer to learn """
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.RANDOM_WARM_UP:
            for _ in range(nr_iters):
                experiences = self.memory.sample()
                self._train(experiences, self.GAMMA)    
        return
    
    def is_warming_up(self):
        return len(self.memory) < self.RANDOM_WARM_UP
    
    
    def act(self, states, add_noise=False):
        """Returns actions for given state as per current policy."""
        if len(states.shape) == 1:
            states = states.reshape(1,-1)
        t_states = torch.from_numpy(states).float().to(self.device)
        self.actor_online.eval()
        with torch.no_grad():
            np_actions = self.actor_online(t_states).cpu().data.numpy()
        self.actor_online.train()
        if add_noise:
            # we are obviously in training so now check if the "act" was called before warmpup
            assert not self.is_warming_up()
            noise = np.random.normal(loc=0, scale=self.exploration_noise, size=np_actions.shape)
            np_actions += noise
        return np.clip(np_actions, -1, 1)
    
    
    def _train(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences: tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        self.train_iters += 1
        if self.train_iters == 1:
            print("\nFirst training iter at step {} (memory={})".format(self.step_counter, len(self.memory)))
        states, actions, rewards, next_states, dones = experiences
        actions_ = actions.cpu().numpy()
    
        actions_next = self.actor_target(next_states)
        noise = torch.FloatTensor(actions_).data.normal_(0, self.policy_noise).to(self.device)
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        actions_next += noise
        actions_next = torch.clamp(actions_next, -1, 1)
        
        Q_targets_next_1 = self.critic_target_1(next_states, actions_next)
        Q_targets_next_2 = self.critic_target_2(next_states, actions_next)
        
        Q_targets_next = torch.min(Q_targets_next_1, Q_targets_next_2)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)).detach()

        # Compute critic loss 1
        Q_expected_1 = self.critic_online_1(states, actions)
        critic_1_loss = F.mse_loss(Q_expected_1, Q_targets)
        # Minimize the loss for critic 1
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_online_1.parameters(), 1)
        self.critic_1_optimizer.step()
        self.np_loss_1 = critic_1_loss.detach().cpu().item()

        
        # Compute critic loss 2
        Q_expected_2 = self.critic_online_2(states, actions)
        critic_2_loss = F.mse_loss(Q_expected_2, Q_targets)
        # Minimize the loss for critic 2
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_online_2.parameters(), 1)
        self.critic_2_optimizer.step()
        self.np_loss_2 = critic_2_loss.detach().cpu().item()
        
        self.critic_1_losses.append(self.np_loss_1)
        self.critic_2_losses.append(self.np_loss_2)
        
        if (self.train_iters % self.policy_freq) == 0:
            actions_pred = self.actor_online(states)
            actor_loss = -self.critic_online_1(states, actions_pred).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.np_loss_actor = actor_loss.detach().cpu().item()
            
            self.soft_update_actor()
            self.soft_update_critics()
            
            self.actor_losses.append(self.np_loss_actor)
            self.actor_updates += 1
            
        return

    def save(self, label):
        fn = '{}_actor_it_{:010}_{}.policy'.format(self.name, self.train_iters, label)
        torch.save(self.actor_online.state_dict(), fn)
        return

    
    def soft_update_actor(self):
        self._soft_update(self.actor_online, self.actor_target, self.TAU)
        return

        
    def soft_update_critics(self):
        self._soft_update(self.critic_online_1, self.critic_target_1, self.TAU)
        self._soft_update(self.critic_online_2, self.critic_target_2, self.TAU)
        return
        
        
    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)    
        return
    
    def debug_weights(self):
        layers_stats(self.actor_online)
        layers_stats(self.critic_online_1)
        return
