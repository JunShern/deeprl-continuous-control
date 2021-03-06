from lib import env, agents
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a trained agent.')
    parser.add_argument('--env_path', type=str, default="./envs/Reacher_Linux_20_Agents/Reacher.x86_64", help='path to Unity ML Agents environnment file')
    parser.add_argument('--checkpoints_path', type=str, default="./checkpoints", help='path containing checkpoint_actor.pth and checkpoint_critic.pth models')
    args = parser.parse_args()

    # Setup
    env = env.EnvUnityMLAgents(args.env_path, train_mode=False)
    agent = agents.DDPGAgent(env.state_size, env.action_size, random_seed=0)
    agent.load(path=args.checkpoints_path)

    _, states, _ = env.reset()

    scores = np.zeros(env.num_agents)
    while True:
        actions = agent.act(states, add_noise=False)
        rewards, states, dones = env.step(actions)
        scores += rewards
        if np.any(dones):
            break

    env.close()
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
