from unityagents import UnityEnvironment
import agents
import matplotlib.pyplot as plt
import numpy as np
import time

class Trainer:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.last_time = time.time()

    def train(self, max_episodes=200):
        all_agent_scores = []
        for i in range(max_episodes):

            _, states, _ = env.reset()
            scores = np.zeros(self.env.num_agents)
            timestep = 0
            while True:
                # select an action (for each agent)
                actions = self.agent.act(states, add_noise=True) # Network expects inputs in batches so feed all at once

                # Act
                rewards, next_states, dones = self.env.step(actions)
                for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                    self.agent.step(state, action, reward, next_state, done, timestep)

                # Update
                scores += rewards
                states = next_states
                
                # Exit if any of the agents finish 
                if np.any(dones):
                    # TODO: Is this the best behavior? 
                    # This termination criteria may cause us to never learn the late-game states,
                    # but it should be fine in this case since the problem does not evolve over time
                    break
                timestep += 1
            
            all_agent_scores.append(scores)
            t = time.time()
            mvg_avg = np.mean(all_agent_scores[-100:])

            print('Episode {} ({:.2f}s) -- Min: {:.2f} -- Max: {:.2f} -- Mean: {:.2f} -- Moving Average: {:.2f}'
                .format(i, t - self.last_time, np.min(scores), np.max(scores), np.mean(scores), mvg_avg))
            self.last_time = t
            if mvg_avg > 30 and len(all_agent_scores) >= 100:
                break

        # Save model
        self.agent.save()
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

def moving_averages(values, window=100):
    return [np.mean(values[:i+1][-window:]) for i, _ in enumerate(values)]

if __name__ == "__main__":
    MAX_EPISODES = 200
    SOLVE_SCORE = 30

    # Setup
    env = EnvUnityMLAgents("./Reacher_Linux_20_Agents_NoVis/Reacher.x86_64")
    # env = EnvUnityMLAgents("./Reacher_Linux_1_Agent_NoVis/Reacher.x86_64")
    agent = agents.DDPGAgent(env.state_size, env.action_size, random_seed=0)
    trainer = Trainer(agent, env)

    # Train
    all_scores = trainer.train(max_episodes=MAX_EPISODES)
    np.save('all_scores.npy', all_scores)
    env.close()

    # Plot results
    plt.figure(figsize=(20, 10))
    for agent_idx in range(all_scores.shape[1]):
        plt.plot(all_scores[:, agent_idx])
    plt.fill_between(x=range(len(all_scores)), y1=all_scores.min(axis=1), y2=all_scores.max(axis=1), alpha=0.2)
    mvg_avgs = moving_averages(np.mean(all_scores, axis=1))
    print(mvg_avgs)
    plt.axhline(y = SOLVE_SCORE, color="red")
    plt.plot(mvg_avgs, color='black', linewidth=2)
    plt.ylabel("score")
    plt.xlabel("episode")
    plt.savefig("scores.png")