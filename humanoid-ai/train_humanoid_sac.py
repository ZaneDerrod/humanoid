# Soft Actor-Critic (SAC) for Humanoid-v4 — CleanRL style
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro

from utils import make_env, layer_init, make_writer, SCALAR_KEYS


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "humanoid-v4"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    resume_path: Optional[str] = None
    """Path to a .pt checkpoint file to resume training from"""

    # Algorithm specific arguments
    env_id: str = "Humanoid-v4"
    """the id of the environment"""
    total_timesteps: int = 8_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 1_000_000
    """the replay buffer size"""
    batch_size: int = 256
    """the batch size for training"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target network update rate (Polyak)"""
    learning_starts: int = 5_000
    """timestep to start learning"""
    policy_update_frequency: int = 1
    """frequency of actor and alpha updates"""
    target_network_frequency: int = 1
    """frequency of target network updates"""


LOG_STD_MIN = -5
LOG_STD_MAX = 2


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_high):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", torch.tensor((action_high + 1.0) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_high - 1.0) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # enforce action bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, buffer_size, device):
        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=torch.tensor(self.obs[idxs], device=self.device),
            actions=torch.tensor(self.actions[idxs], device=self.device),
            rewards=torch.tensor(self.rewards[idxs], device=self.device),
            next_obs=torch.tensor(self.next_obs[idxs], device=self.device),
            dones=torch.tensor(self.dones[idxs], device=self.device),
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.resume_path is not None:
        run_name += "_resumed"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = make_writer(run_name, args)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(1)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    obs_dim = np.array(envs.single_observation_space.shape).prod()
    action_dim = np.prod(envs.single_action_space.shape)
    action_high = envs.single_action_space.high

    actor = Actor(obs_dim, action_dim, action_high).to(device)
    qf1 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    if args.resume_path is not None:
        actor.load_state_dict(torch.load(args.resume_path, map_location=device))
        print(f"Resumed actor from checkpoint: {args.resume_path}")

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)

    # automatic entropy tuning
    target_entropy = -action_dim
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    alpha_optimizer = optim.Adam([log_alpha], lr=args.learning_rate)

    rb = ReplayBuffer(obs_dim, action_dim, args.buffer_size, device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)

    try:
        for global_step in range(1, args.total_timesteps + 1):
            # ALGO LOGIC: action logic
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample()])
            else:
                with torch.no_grad():
                    actions, _, _ = actor.get_action(next_obs)
                    actions = actions.cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_np, rewards, terminations, truncations, infos = envs.step(actions)
            dones = np.logical_or(terminations, truncations)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # store transition
            real_next_obs = next_obs_np.copy()
            rb.add(
                next_obs.cpu().numpy().flatten(),
                actions.flatten(),
                rewards.flatten()[0],
                real_next_obs.flatten(),
                dones.flatten()[0],
            )

            next_obs = torch.Tensor(next_obs_np).to(device)

            # ALGO LOGIC: training
            if global_step >= args.learning_starts:
                data = rb.sample(args.batch_size)

                # --- critic update ---
                with torch.no_grad():
                    next_actions, next_log_pi, _ = actor.get_action(data["next_obs"])
                    qf1_next = qf1_target(data["next_obs"], next_actions)
                    qf2_next = qf2_target(data["next_obs"], next_actions)
                    min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
                    target_q = data["rewards"].unsqueeze(-1) + (1 - data["dones"].unsqueeze(-1)) * args.gamma * min_qf_next

                qf1_val = qf1(data["obs"], data["actions"])
                qf2_val = qf2(data["obs"], data["actions"])
                qf1_loss = F.mse_loss(qf1_val, target_q)
                qf2_loss = F.mse_loss(qf2_val, target_q)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # --- actor and alpha update ---
                if global_step % args.policy_update_frequency == 0:
                    pi_actions, log_pi, _ = actor.get_action(data["obs"])
                    qf1_pi = qf1(data["obs"], pi_actions)
                    qf2_pi = qf2(data["obs"], pi_actions)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = (alpha * log_pi - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # alpha update
                    with torch.no_grad():
                        _, new_log_pi, _ = actor.get_action(data["obs"])
                    alpha_loss = (-log_alpha * (new_log_pi + target_entropy).detach()).mean()

                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                    alpha = log_alpha.exp().item()

                # --- target network update ---
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                # --- logging ---
                if global_step % 1000 == 0:
                    sps = int(global_step / (time.time() - start_time))
                    print("SPS:", sps)
                    writer.add_scalar("charts/learning_rate", args.learning_rate, global_step)
                    writer.add_scalar("charts/SPS", sps, global_step)
                    writer.add_scalar("losses/value_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/policy_loss", actor_loss.item() if global_step % args.policy_update_frequency == 0 else 0.0, global_step)
                    writer.add_scalar("losses/entropy", -log_pi.mean().item() if global_step % args.policy_update_frequency == 0 else 0.0, global_step)
                    writer.add_scalar("losses/approx_kl", 0.0, global_step)
                    writer.add_scalar("losses/clipfrac", 0.0, global_step)
                    writer.add_scalar("losses/explained_variance", 0.0, global_step)

                # --- checkpoint ---
                if global_step % 50_000 == 0:
                    os.makedirs(f"runs/{run_name}", exist_ok=True)
                    torch.save(actor.state_dict(), f"runs/{run_name}/agent_actor_{global_step}.pt")
                    print(f"Checkpoint saved at step {global_step}")

    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current weights...")
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        torch.save(actor.state_dict(), f"runs/{run_name}/agent_interrupted_{global_step}.pt")
        print(f"Model saved to runs/{run_name}/agent_interrupted_{global_step}.pt")

    # Save final model
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    torch.save(actor.state_dict(), f"runs/{run_name}/agent_final.pt")
    print(f"Final model saved to runs/{run_name}/agent_final.pt")

    envs.close()
    writer.close()
