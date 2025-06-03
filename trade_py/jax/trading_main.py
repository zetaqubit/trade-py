import functools
import jax
import os

from datetime import datetime
from jax import numpy as jnp

import brax

import flax
from brax import envs
from brax.io import model
from brax.io import json
from brax.io import html
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac

from trade_py.jax import trading_env

def get_prices(filepath):
    import pandas as pd
    df = pd.read_csv(filepath)
    return jnp.array(df.close.to_numpy())

# prices = jnp.arange(1, 2000)
# prices = get_prices('/home/z/data/zetaqubit/stock/alpaca/AAPL-5y-1Hour-2025-05-31.csv')
prices = get_prices('/home/z/data/zetaqubit/stock/alpaca/GOOG-5y-1Hour-2025-05-31.csv')

env = trading_env.SingleAssetTradingEnv(prices, max_episode_len=1000, window_size=20)

# train_fn = functools.partial(ppo.train, num_timesteps=2_000_000, num_evals=20, reward_scaling=1000, episode_length=1000, normalize_observations=False, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=64, batch_size=128, seed=1)
train_fn = functools.partial(ppo.train, num_timesteps=20_000_000, num_evals=200, reward_scaling=1000, episode_length=1000, normalize_observations=False, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-5, entropy_cost=1e-2, num_envs=2048, batch_size=1024, seed=1)

xdata, ydata = [], []
times = [datetime.now()]

rl_rewards = []
bh_rewards = []

def progress(num_steps, metrics):
    m = metrics
    times.append(datetime.now())
    xdata.append(num_steps)
    rl_rewards.append(jnp.exp(metrics['eval/episode_reward']))
    bh_rewards.append(jnp.exp(m['eval/episode_buy_hold_reward']))
    print(xdata[-1], 'reward:', rl_rewards[-1],
          'buy_hold_reward', bh_rewards[-1],
          'alloc', m['eval/episode_allocation'],
          '/', m['eval/episode_max_alloc'])


# with jax.disable_jit():
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f'avg rl reward (last 50 evals)', jnp.mean(jnp.array(rl_rewards[-50:])))
print(f'avg bh reward (last 50 evals)', jnp.mean(jnp.array(bh_rewards[-50:])))

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')
