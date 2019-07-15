import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline  
import pandas as pd

np.set_printoptions(precision=4)
np.set_printoptions(linewidth=130)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)     

def training_loop(env, agent, n_episodes, policy_noise_reduction, explor_noise_reduction,
                  stop_policy_noise=0, stop_explor_noise=0,
                  max_t=10000, train_every_steps=10,
                  DEBUG=1,
                 ):
    
    print("Starting training for {} episodes...".format(n_episodes))
    print("  explor_noise_reduction={}".format(explor_noise_reduction))
    print("  policy_noise_reduction={}".format(policy_noise_reduction))
    print("  stop_policy_noise={}".format(stop_policy_noise))
    print("  stop_explor_noise={}".format(stop_explor_noise))
    solved_episode = 0
    scores_deque = deque(maxlen=100)
    steps_deque = deque(maxlen=100)
    scores_avg = []
    scores = []
    ep_times = []
    for i_episode in range(1, n_episodes+1):
        t_start = time.time()
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        np_score = np.zeros(num_agents)        
        for t in range(max_t):
            if agent.is_warming_up():
                actions = sample_action()
            else:
                actions = agent.act(states, add_noise=True)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            agent.step(states, actions, rewards, next_states, dones, train_every_steps=train_every_steps)
            states = next_states
            np_score += rewards
            if np.any(dones):
                break           
        if train_every_steps==0:
            agent.train(nr_iters=t//10)
        episode_max = np_score.max()
        score = np_score.mean()
        scores_deque.append(score)
        scores.append(score)
        scores_avg.append(np.mean(scores_deque))
        steps_deque.append(t)
        t_end = time.time()
        ep_time = t_end - t_start
        ep_times.append(ep_time)
        _cl1 = np.mean(agent.critic_1_losses)
        _cl2 = np.mean(agent.critic_2_losses)
        _al = np.mean(agent.actor_losses)
        max_score = np.max(scores_deque)
        print('\rEpisode {:>4}  Score/M100/Avg: {:>4.1f}/{:>4.1f}/{:>4.1f}  Steps: {:>4}  [μcL1/μcL2: {:>8.1e}/{:>8.1e} μaL: {:>8.1e}]  t:{:>4.1f}s    '.format(
            i_episode, score, max_score, np.mean(scores_deque), t, _cl1,_cl2, _al, ep_time), end="", flush=True)
        if (np.mean(scores_deque) > 30) and (solved_episode == 0):
            print("\nEnvironment solved at episode {}!".format(i_episode))
            agent.save('ep_{}_solved'.format(i_episode))
            solved_episode = i_episode
        if i_episode % 50 == 0:
            mean_ep = np.mean(ep_times)
            elapsed = i_episode * mean_ep
            total = (n_episodes + 1) * mean_ep
            left_time_hrs = (total - elapsed) / 3600            
            print('\rEpisode {:>4}  Score/M100/Avg: {:>4.1f}/{:>4.1f}/{:>4.1f}  AvStp: {:>4.0f}  [μcL1/μcL2: {:>8.1e}/{:>8.1e} μaL: {:>8.1e}]  t-left:{:>4.1f} h    '.format(
                i_episode, score, max_score, np.mean(scores_deque), np.mean(steps_deque), _cl1,_cl2, _al, left_time_hrs))
            if DEBUG >= 1:
                print("  Loaded steps: {:>10} (Replay memory: {})".format(agent.step_counter, len(agent.memory)))
                print("  Critic/Actor updates:  {:>10} / {:>10}".format(agent.train_iters, agent.actor_updates))
            if DEBUG >= 2:
                agent.debug_weights()
            if explor_noise_reduction:
                agent.reduce_explore_noise(0.8)
            if policy_noise_reduction:
                agent.reduce_policy_noise(0.8)
        if stop_policy_noise>0 and i_episode >= stop_policy_noise:
            agent.clear_policy_noise()
        if stop_explor_noise>0 and i_episode >= stop_explor_noise:
            agent.clear_explore_noise()
                
    #agent.save('ep_{}'.format(i_episode))
    return scores, scores_avg, solved_episode

from workspace_utils import active_session
 
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEBUG = 1

# ab1  : Actor with batch-norm only on state input
# ab2  : Actor with batch-norm only on pre-activations
# ab3  : Actor with all batch-norms
# ann  : Actor without any norming
# csbn : Critic with batch-normed featurized state
# cvnn : Critic with no norm on featureized state input
# cdbn : Critic with direct state (no featurization layer) and full batch-norm
# cdnn : Critic simplest version (similar to TD3 paper) - concate then fc no norm
# conn : Critic post-state has no normalization
# cobn : Critic post-state has batch-norm
if num_agents == 1:
    act_opt = ['ab1', 'ab2', 'ab3', 'ann']
    cri_opt = ['csbn_conn', 'csbn_cobn', 'csnn_conn','cdbn','cdnn']
else:
    act_opt = ['ab2', 'ann']
    cri_opt = ['csbn_conn' , 'cdnn' ]
    
noi_opt = ["f_noi" ] #, "stp_pol_noise", "stp_exp_noise", "stp_all"]#['policy_noise_reduction', 'explore_noise_reduction']
iters = [x+"_"+y+"_"+z for x in act_opt for y in cri_opt for z in noi_opt]
print("The search grid contains {} iterations:".format(len(iters)))
for _ii in iters:
    print("  {}".format(_ii))
results = {
    "MODEL" : [],
    "EP_SOL" : [],
    "BEST_AVG" : [],
    "AVG1": [],
    "AVG2": [],
    "AVG3": [],
    "AVG4": [],
}

stop_noise_after = 290
all_iters = []
iter_base = 'm_' if num_agents > 1 else 's_'
iter_base = iter_base + str(dev.type) + '_'
with active_session():
    for ii, iteration_name in enumerate(iters):
        reset_seed()
        iteration_name = iter_base + iteration_name
        print("\n\nStarting grid search iteration {}/{}:'{}'".format(ii+1, len(iters), iteration_name))
        policy_noise_red = ('policy' in iteration_name) and  ("reduction" in iteration_name)
        explor_noise_red = ('explor' in iteration_name) and  ("reduction" in iteration_name)
        stop_policy_noise = stop_noise_after if (("stop" in iteration_name) and ("policy" in iteration_name)) else 0
        stop_explor_noise = stop_noise_after if (("stop" in iteration_name) and ("explor" in iteration_name)) else 0
        simple_critic = ('cdbn' in iteration_name) or ('cdnn' in iteration_name)
        use_bn_actor_pre = ('ab1' in iteration_name) or ('ab3' in iteration_name)
        use_bn_actor_post = 'ab2' in iteration_name
        use_bn_critic_state = 'csbn' in iteration_name
        use_bn_critic_other = ('cobn' in iteration_name) or ('cdbn' in iteration_name)
        train_every_steps=10
        
        if num_agents > 1:
            _sampl = [ 30, 70,120,145]
            n_ep = 150
        else:
            _sampl = [ 50,150,250,300]
            n_ep = 350
            
        agent = Agent(a_size=_action_size, s_size=_state_size, 
                      dev=dev, 
                      n_env_agents=num_agents,
                      simplified_critic=simple_critic,
                      critic_use_state_bn=use_bn_critic_state,
                      critic_use_other_bn=use_bn_critic_other,
                      actor_use_pre_bn=use_bn_actor_pre,
                      actor_use_post_bn=use_bn_actor_post,
                      name=iteration_name, 
                     )
        _res = training_loop(env=env, agent=agent, n_episodes=n_ep, 
                             train_every_steps=train_every_steps,
                             policy_noise_reduction=policy_noise_red,
                             explor_noise_reduction=explor_noise_red,
                             stop_policy_noise=stop_policy_noise,
                             stop_explor_noise=stop_explor_noise,
                             DEBUG=DEBUG,
                            )
        scores, scores_avg, i_solved = _res
        results['MODEL'].append(iteration_name)
        results['BEST_AVG'].append(np.max(scores_avg))
        results['EP_SOL'].append(i_solved)
        all_iters.append((iteration_name, scores_avg))
        n_sampl = len(_sampl)
        for _i,_pos in enumerate(_sampl):
            results['AVG'+str(_i+1)].append(scores_avg[_pos] if len(scores_avg) > _pos else scores_avg[-(n_sampl-_i)])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores)+1), scores,"-b", label='score')
        plt.plot(np.arange(1, len(scores)+1), scores_avg,"-r", label='average')
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.legend()
        plt.title("Continuous Control - " + iteration_name)
        plt.axhline(y=30, linestyle='--', color='green')
        plt.savefig(iteration_name+'.png')
        plt.show()

        df_scores = pd.DataFrame({"EPISODE":np.arange(1, len(scores)+1), "SCORE": scores, "AVG_SCORE100":scores_avg})
        df_scores.to_csv("scores_{}.csv".format(iteration_name))
        df_res = pd.DataFrame(results).sort_values('BEST_AVG')
        print(df_res)
        df_res.to_csv('results.csv')

    
    clrs_opt = ['b','g','r','c','m','y']
    styles_opt = ['-','--',':']
    styles = [x+y for x in styles_opt for y in clrs_opt]
    plt.figure(figsize=(15,10))
    for ii, done_iter in enumerate(all_iters):
        iter_name = done_iter[0]
        avg_scores = done_iter[1]
        plt.plot(np.arange(1, len(avg_scores)+1), avg_scores, styles[ii], label=iter_name)
    plt.legend()
    plt.title("Results comparison")
    plt.ylabel('Avg. Score')
    plt.xlabel('Episode #')
    plt.axhline(y=30, linestyle='--', color='black')
    plt.savefig('comparison.png')
    plt.show()
        

