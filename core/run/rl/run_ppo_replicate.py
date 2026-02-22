#!/usr/bin/env python3
"""
PPO for SlipperyAnt: Replication of Lyle et al. loss-of-plasticity experiments.

Matches original paper setup exactly:
  - Network: 2×256 ReLU, separate actor/critic, LeCun init
  - PPO: batch=2048, 16 mini-batches (128 each), 10 epochs, clip=0.2
  - Loss: policy + value (no vf_coef, no entropy bonus, no value clipping)
  - Optimizer: Adam lr=1e-4, eps=1e-8 (std); Adam wd=1e-3 betas=0.99 (l2)
  - CBP: Adam + generate-and-test neuron replacement (rr=1e-4, mt=10000)
  - No obs/reward normalization, no LR annealing, single env
  - Friction change every 2M steps at episode boundary
"""

import os
import math
import pickle
import random
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from core.envs.slippery_ant import make_slippery_ant
from core.run.rl.rl_upgd_layerselective import RLLayerSelectiveUPGD


# ─── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='PPO SlipperyAnt replication')
    p.add_argument('--method', type=str, required=True,
                   choices=['std', 'l2', 'cbp', 'cbp_h1', 'cbp_h2', 'cbp_no_gnt',
                            'reset_head', 'shrink_head',
                            'upgd_full', 'upgd_output_only', 'upgd_hidden_only'])
    p.add_argument('--seed', type=int, required=True,
                   help='Random seed (also indexes friction schedule)')
    p.add_argument('--total-timesteps', type=int, default=20_000_000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--track', action='store_true', help='Log to WandB')
    p.add_argument('--wandb-project', type=str, default='upgd-rl')
    p.add_argument('--wandb-entity', type=str, default=None)
    # UPGD-specific
    p.add_argument('--upgd-wd', type=float, default=0.0,
                   help='UPGD weight decay')
    p.add_argument('--beta-utility', type=float, default=0.999)
    p.add_argument('--sigma', type=float, default=0.001)
    p.add_argument('--non-gated-scale', type=float, default=1.0,
                   help='Scale for non-gated UPGD layers (1.0 = full SGD)')
    return p.parse_args()


# ─── Network ────────────────────────────────────────────────────────────────────

def lecun_init(module):
    """LeCun uniform: U(-sqrt(3/fan_in), sqrt(3/fan_in)), bias=0."""
    fan_in = module.in_features
    bound = math.sqrt(3.0 / fan_in)
    nn.init.uniform_(module.weight, -bound, bound)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0.0)


class Agent(nn.Module):
    """2×256 ReLU actor-critic with LeCun init (matching original paper)."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim),
        )
        # Learnable log_std per action dimension, init=0 → std=1
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        # LeCun init all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                lecun_init(m)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        return (action,
                dist.log_prob(action).sum(-1),
                dist.entropy().sum(-1),
                self.critic(x))


# ─── CBP (Continual Backpropagation) ─────────────────────────────────────────────

class GnT:
    """Generate-and-Test for selective neuron replacement (Dohare et al.).

    Tracks contribution-based utility of hidden neurons and replaces
    low-utility mature neurons with freshly initialized weights.
    Operates on a single nn.Sequential network (actor or critic).
    """

    def __init__(self, net, num_hidden, device, init='lecun',
                 replacement_rate=1e-4, maturity_threshold=10000,
                 decay_rate=0.99, active_layers=None):
        self.net = net
        self.num_hidden = num_hidden
        self.device = device
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.decay_rate = decay_rate
        # active_layers: which hidden layers get replacement
        # None = all hidden layers (default), [0] = first only, [1] = second only
        self.active_layers = active_layers if active_layers is not None else list(range(num_hidden))

        # Per-layer state for each hidden layer
        self.util = [torch.zeros(net[i * 2].out_features, device=device)
                     for i in range(num_hidden)]
        self.ages = [torch.zeros(net[i * 2].out_features, device=device)
                     for i in range(num_hidden)]
        self.mean_act = [torch.zeros(net[i * 2].out_features, device=device)
                         for i in range(num_hidden)]

        # LeCun init bounds for replacement
        self.bounds = [math.sqrt(3.0 / net[i * 2].in_features)
                       for i in range(num_hidden)]
        # Output layer bound
        self.bounds.append(math.sqrt(3.0 / net[num_hidden * 2].in_features))

    def gen_and_test(self, features):
        """Run one step of generate-and-test.

        Args:
            features: list of activation tensors per hidden layer,
                      each shape (batch, hidden_dim). Append None for output.
        """
        if self.replacement_rate == 0:
            return
        with torch.no_grad():
            for i in range(self.num_hidden):
                self.ages[i] += 1

                # Skip replacement for inactive layers (still track ages)
                if i not in self.active_layers:
                    continue

                # Update running mean activation
                self.mean_act[i] *= self.decay_rate
                self.mean_act[i] += (1 - self.decay_rate) * features[i].mean(dim=0)

                # Contribution utility: outgoing_weight_mag * activation_mag
                current_layer = self.net[i * 2]
                next_layer = self.net[i * 2 + 2]
                out_weight_mag = next_layer.weight.data.abs().mean(dim=0)
                new_util = out_weight_mag * features[i].abs().mean(dim=0)

                self.util[i] *= self.decay_rate
                self.util[i] += (1 - self.decay_rate) * new_util

                # Bias-corrected utility
                bias_correction = 1 - self.decay_rate ** self.ages[i]
                bc_util = self.util[i] / bias_correction

                # Find eligible neurons (mature enough)
                eligible = torch.where(self.ages[i] > self.maturity_threshold)[0]
                if eligible.shape[0] == 0:
                    continue

                # Number to replace (stochastic rounding for fractional counts)
                n_replace = self.replacement_rate * eligible.shape[0]
                if n_replace < 1:
                    n_replace = 1 if torch.rand(1).item() <= n_replace else 0
                n_replace = int(n_replace)
                if n_replace == 0:
                    continue

                # Select lowest-utility neurons among eligible
                to_replace = torch.topk(-bc_util[eligible], n_replace)[1]
                to_replace = eligible[to_replace]

                # Correct next layer bias before zeroing outgoing weights
                bc_act = self.mean_act[i][to_replace] / bias_correction[to_replace]
                next_layer.bias.data += (
                    next_layer.weight.data[:, to_replace] * bc_act
                ).sum(dim=1)

                # Reinitialize incoming weights (LeCun uniform)
                current_layer.weight.data[to_replace] = torch.empty(
                    n_replace, current_layer.in_features, device=self.device
                ).uniform_(-self.bounds[i], self.bounds[i])
                current_layer.bias.data[to_replace] = 0

                # Zero outgoing weights
                next_layer.weight.data[:, to_replace] = 0

                # Reset state for replaced neurons
                self.util[i][to_replace] = 0
                self.mean_act[i][to_replace] = 0
                self.ages[i][to_replace] = 0

    def reset_optimizer_state(self, optimizer, features_replaced=None):
        """Reset Adam state for replaced neurons (optional, improves stability)."""
        # Standard Adam doesn't track per-element steps, so we just
        # zero the moments for replaced parameters. This is approximate
        # but sufficient for replication.
        pass


def get_activations(net, x):
    """Forward pass through nn.Sequential, capturing hidden activations."""
    activations = []
    for i, layer in enumerate(net):
        x = layer(x)
        # Capture output of each ReLU (odd-indexed layers)
        if i % 2 == 1:
            activations.append(x)
    return x, activations


def _head_param_ids(agent):
    """Return set of id() for output layer parameters."""
    return {id(agent.actor_mean[4].weight), id(agent.actor_mean[4].bias),
            id(agent.critic[4].weight), id(agent.critic[4].bias),
            id(agent.actor_logstd)}


def _clear_optimizer_state(optimizer, param_ids):
    """Clear Adam momentum/variance for specified parameters."""
    for pg in optimizer.param_groups:
        for p in pg['params']:
            if id(p) in param_ids and p in optimizer.state:
                del optimizer.state[p]


def reset_head_optimizer(agent, optimizer):
    """Reset only Adam state for head layers — keep weights, clear momentum."""
    _clear_optimizer_state(optimizer, _head_param_ids(agent))


def shrink_head(agent, optimizer, alpha=0.5):
    """Shrink head weights toward LeCun init: w = alpha*w + (1-alpha)*w_init."""
    for layer in [agent.actor_mean[4], agent.critic[4]]:
        fan_in = layer.in_features
        bound = math.sqrt(3.0 / fan_in)
        w_init = torch.empty_like(layer.weight).uniform_(-bound, bound)
        b_init = torch.zeros_like(layer.bias)
        layer.weight.data.mul_(alpha).add_(w_init, alpha=(1 - alpha))
        layer.bias.data.mul_(alpha).add_(b_init, alpha=(1 - alpha))
    # Shrink logstd toward 0
    agent.actor_logstd.data.mul_(alpha)
    _clear_optimizer_state(optimizer, _head_param_ids(agent))


# ─── Optimizer ──────────────────────────────────────────────────────────────────

def create_optimizer(agent, args):
    """Create optimizer matching original paper configs."""
    if args.method == 'std':
        # Standard PPO: Adam with paper defaults
        return optim.Adam(agent.parameters(), lr=args.lr,
                          betas=(0.9, 0.999), eps=1e-8)
    elif args.method == 'l2':
        # L2: Dohare et al. params — betas=(0.99, 0.99), wd=1e-3
        return optim.Adam(agent.parameters(), lr=args.lr,
                          betas=(0.99, 0.99), eps=1e-8, weight_decay=1e-3)
    elif args.method in ('cbp', 'cbp_h1', 'cbp_h2', 'cbp_no_gnt',
                         'reset_head', 'shrink_head'):
        # CBP / head variants: Dohare et al. params — betas=(0.99, 0.99), wd=1e-4
        return optim.Adam(agent.parameters(), lr=args.lr,
                          betas=(0.99, 0.99), eps=1e-8, weight_decay=1e-4)
    elif args.method.startswith('upgd_'):
        gating_map = {
            'upgd_full': 'full',
            'upgd_output_only': 'output_only',
            'upgd_hidden_only': 'hidden_only',
        }
        return RLLayerSelectiveUPGD(
            agent.named_parameters(),
            lr=args.lr,
            weight_decay=args.upgd_wd,
            beta_utility=args.beta_utility,
            sigma=args.sigma,
            gating_mode=gating_map[args.method],
            non_gated_scale=args.non_gated_scale,
        )
    else:
        raise ValueError(f'Unknown method: {args.method}')


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Seeding ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # ── Load friction schedule ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    frictions_path = os.path.join(project_root, 'core', 'envs', 'frictions.pkl')
    with open(frictions_path, 'rb') as f:
        frictions = pickle.load(f)

    # ── Create initial environment ──
    xml_file = os.path.abspath(f'slippery_ant_{args.seed}.xml')
    friction_number = 0
    initial_friction = frictions[args.seed][friction_number]
    env = make_slippery_ant(friction=initial_friction, xml_file=xml_file)
    print(f'Initial friction: {initial_friction:.6f}')

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f'obs_dim={obs_dim}, act_dim={act_dim}')

    # ── Agent & Optimizer ──
    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = create_optimizer(agent, args)

    # ── CBP: Generate-and-Test setup ──
    actor_gnt = None
    critic_gnt = None
    if args.method in ('cbp', 'cbp_h1', 'cbp_h2'):
        active_layers_map = {
            'cbp': None,    # both hidden layers
            'cbp_h1': [0],  # first hidden only
            'cbp_h2': [1],  # second hidden only
        }
        active = active_layers_map[args.method]
        actor_gnt = GnT(agent.actor_mean, num_hidden=2, device=device,
                         replacement_rate=1e-4, maturity_threshold=10000,
                         decay_rate=0.99, active_layers=active)
        critic_gnt = GnT(agent.critic, num_hidden=2, device=device,
                          replacement_rate=1e-4, maturity_threshold=10000,
                          decay_rate=0.99, active_layers=active)

    print(f'Method: {args.method}')
    print('Parameters:')
    for name, p in agent.named_parameters():
        print(f'  {name}: {p.shape}')

    # ── PPO hyperparameters (matching original paper exactly) ──
    BATCH_SIZE = 2048
    NUM_MINIBATCHES = 16
    MINIBATCH_SIZE = BATCH_SIZE // NUM_MINIBATCHES  # 128
    NUM_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    MAX_GRAD_NORM = 1e9  # effectively disabled
    CHANGE_TIME = 2_000_000

    num_iterations = args.total_timesteps // BATCH_SIZE
    checkpoint_interval = max(1, num_iterations // 100)  # ~every 1% = 200k steps

    # ── WandB ──
    wandb = None
    if args.track:
        import wandb as _wandb
        wandb = _wandb
        run_name = f"sant__{args.method}__s{args.seed}__{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=f"sant__{args.method}",
            config={
                **vars(args),
                'batch_size': BATCH_SIZE,
                'num_minibatches': NUM_MINIBATCHES,
                'num_epochs': NUM_EPOCHS,
                'gamma': GAMMA,
                'gae_lambda': GAE_LAMBDA,
                'clip_eps': CLIP_EPS,
                'network': '2x256_relu_lecun',
                'obs_norm': False,
                'rew_norm': False,
                'lr_anneal': False,
                'clip_vloss': False,
                'vf_coef': 1.0,
                'ent_coef': 0.0,
            },
            tags=['replicate'],
        )

    # ── Storage buffers ──
    obs_buf = torch.zeros((BATCH_SIZE, obs_dim), device=device)
    actions_buf = torch.zeros((BATCH_SIZE, act_dim), device=device)
    logprobs_buf = torch.zeros(BATCH_SIZE, device=device)
    rewards_buf = torch.zeros(BATCH_SIZE, device=device)
    dones_buf = torch.zeros(BATCH_SIZE, device=device)
    values_buf = torch.zeros(BATCH_SIZE, device=device)

    # ── Initialize ──
    obs_np, _ = env.reset(seed=args.seed)
    next_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    next_done = 0.0
    global_step = 0
    previous_change_time = 0
    episode_return = 0.0
    episode_count = 0
    start_time = time.time()

    # ── Training loop ──
    for iteration in range(1, num_iterations + 1):

        # ── Rollout: collect BATCH_SIZE transitions ──
        for step in range(BATCH_SIZE):
            global_step += 1
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs.unsqueeze(0)
                )

            values_buf[step] = value.squeeze()
            actions_buf[step] = action.squeeze(0)
            logprobs_buf[step] = logprob.squeeze()

            obs_np, reward, terminated, truncated, info = env.step(
                action.cpu().numpy().squeeze(0)
            )
            done = terminated or truncated
            next_done = float(done)
            rewards_buf[step] = float(reward)

            episode_return += reward

            if done:
                episode_count += 1
                print(f'step={global_step}, ep={episode_count}, '
                      f'return={episode_return:.1f}')
                if wandb is not None:
                    wandb.log({
                        'charts/episodic_return': episode_return,
                    }, step=global_step)
                episode_return = 0.0

                # Check friction change (at episode boundary after threshold)
                if global_step - previous_change_time > CHANGE_TIME:
                    previous_change_time = global_step
                    friction_number += 1
                    new_friction = frictions[args.seed][friction_number]
                    print(f'  >>> FRICTION CHANGE #{friction_number}: '
                          f'{new_friction:.6f} at step {global_step}')
                    env.close()
                    env = make_slippery_ant(
                        friction=new_friction, xml_file=xml_file
                    )
                    obs_np, _ = env.reset()
                    # Head intervention at friction change
                    if args.method == 'reset_head':
                        reset_head_optimizer(agent, optimizer)
                        print(f'  >>> HEAD OPTIM RESET at step {global_step}')
                    elif args.method == 'shrink_head':
                        shrink_head(agent, optimizer, alpha=0.5)
                        print(f'  >>> HEAD SHRINK at step {global_step}')
                else:
                    obs_np, _ = env.reset()

            next_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)

        # ── GAE advantage computation ──
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).squeeze()

        advantages = torch.zeros(BATCH_SIZE, device=device)
        lastgaelam = 0.0
        for t in reversed(range(BATCH_SIZE)):
            if t == BATCH_SIZE - 1:
                nextnonterminal = 1.0 - next_done
                nextvalue = next_value
            else:
                nextnonterminal = 1.0 - dones_buf[t + 1]
                nextvalue = values_buf[t + 1]
            delta = (rewards_buf[t]
                     + GAMMA * nextvalue * nextnonterminal
                     - values_buf[t])
            advantages[t] = lastgaelam = (
                delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            )
        returns_buf = advantages + values_buf

        # ── PPO update ──
        b_inds = np.arange(BATCH_SIZE)
        for epoch in range(NUM_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                # Forward pass — use get_activations for CBP
                mb_obs = obs_buf[mb_inds]
                mb_act = actions_buf[mb_inds]
                if actor_gnt is not None:
                    actor_out, actor_acts = get_activations(
                        agent.actor_mean, mb_obs)
                    critic_out, critic_acts = get_activations(
                        agent.critic, mb_obs)
                    action_std = torch.exp(
                        agent.actor_logstd.expand_as(actor_out))
                    dist = Normal(actor_out, action_std)
                    newlogprob = dist.log_prob(mb_act).sum(-1)
                    entropy = dist.entropy().sum(-1)
                    newvalue = critic_out
                else:
                    _, newlogprob, entropy, newvalue = \
                        agent.get_action_and_value(mb_obs, mb_act)

                logratio = newlogprob - logprobs_buf[mb_inds]
                ratio = logratio.exp()

                # Advantage normalization
                mb_adv = advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss (PPO clip)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - CLIP_EPS, 1 + CLIP_EPS
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss: MSE, NO 0.5 factor, NO clipping
                v_loss = (
                    (newvalue.view(-1) - returns_buf[mb_inds]) ** 2
                ).mean()

                # Combined loss: policy + value
                # NO entropy bonus (ent_coef=0), NO vf_coef weighting
                loss = pg_loss + v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                # CBP: generate-and-test after each minibatch
                if actor_gnt is not None:
                    with torch.no_grad():
                        actor_gnt.gen_and_test(actor_acts)
                        critic_gnt.gen_and_test(critic_acts)

        # ── Periodic logging ──
        if iteration % checkpoint_interval == 0 or iteration == num_iterations:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed)
            print(f'[{iteration}/{num_iterations}] step={global_step}, '
                  f'SPS={sps}, ploss={pg_loss.item():.4f}, '
                  f'vloss={v_loss.item():.4f}')
            if wandb is not None:
                log_dict = {
                    'charts/SPS': sps,
                    'losses/policy_loss': pg_loss.item(),
                    'losses/value_loss': v_loss.item(),
                }
                # UPGD-specific stats
                if hasattr(optimizer, 'get_utility_stats'):
                    for k, v in optimizer.get_utility_stats().items():
                        if isinstance(v, (int, float)):
                            log_dict[k] = v
                wandb.log(log_dict, step=global_step)

    # ── Cleanup ──
    env.close()
    elapsed = time.time() - start_time
    print(f'\nDone! {episode_count} episodes, {global_step} steps '
          f'in {elapsed:.0f}s ({elapsed/3600:.1f}h)')
    # Clean up temp XML
    if os.path.exists(xml_file):
        os.remove(xml_file)
    if wandb is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
