from unityagents import UnityEnvironment
import agents
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def train(self, n_episodes=10):
        all_agent_scores = []
        for i in range(n_episodes):

            _, states, _ = env.reset()
            scores = np.zeros(self.env.num_agents)
            while True:
                # select an action (for each agent)
                actions = [self.agent.act(state, add_noise=True) for state in states]

                # Act
                rewards, next_states, dones = self.env.step(actions)
                for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                    self.agent.step(state, action, reward, next_state, done)

                # Update
                scores += rewards
                states = next_states
                
                # Exit if any of the agents finish 
                if np.any(dones):
                    # TODO: Is this the best behavior? 
                    # This termination criteria may cause us to never learn the late-game states,
                    # but it should be fine in this case since the problem does not evolve over time
                    break
            
            all_agent_scores.append(scores)
            print(scores)
            print('Total score (averaged over agents) episode {}: {}'.format(i, np.mean(scores)))
            # print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
            # torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            # torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            # if i_episode % print_every == 0:
            #     print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        return np.array(all_agent_scores)

class EnvUnityMLAgents:
    def __init__(self, file_name):
        self.env = UnityEnvironment(file_name=file_name)
        self.brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(env_info.agents)
        brain = self.env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size
        states = env_info.vector_observations
        self.state_size = states.shape[1]

        print('Number of agents:', self.num_agents)
        print('Size of each action:', self.action_size)
        print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], self.state_size))
        print('The state for the first agent looks like:', states[0])

    def reset(self):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        rewards = env_info.rewards
        next_states = env_info.vector_observations
        dones = env_info.local_done
        return rewards, next_states, dones

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        return rewards, next_states, dones

    def close(self):
        self.env.close()


if __name__ == "__main__":
    # Setup
    env = EnvUnityMLAgents("./Reacher_Linux_20_Agents_NoVis/Reacher.x86_64")
    agent = agents.DDPGAgent(env.state_size, env.action_size, random_seed=0)
    trainer = Trainer(agent, env)

    # Train
    NUM_EPISODES = 10
    all_scores = trainer.train(n_episodes=NUM_EPISODES)    
    env.close()

    print(all_scores)

    plt.figure(figsize=(20, 10))
    for agent_idx in range(all_scores.shape[1]):
        plt.plot(all_scores[:, agent_idx])
    plt.fill_between(x=range(NUM_EPISODES), y1=all_scores.min(axis=1), y2=all_scores.max(axis=1), alpha=0.2)
    plt.plot(all_scores.mean(axis=1), color='black', linewidth=2)
    plt.ylabel("score")
    plt.xlabel("episode")
    plt.savefig("scores.png")