import functools
import jax
import os

from contextlib import nullcontext
from datetime import datetime, timedelta
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

def load_df(filepath):
    import pandas as pd
    return pd.read_csv(filepath)

def get_prices(df, start=None, end=None):
    COL_DATE = 'timestamp'
    if start:
      df = df.query(f'{COL_DATE} >= "{start}"')
    if end:
      df = df.query(f'{COL_DATE} < "{end}"')
    print(df.head(1))
    print(df.tail(1))
    return jnp.array(df.close.to_numpy())

def get_train_eval_envs(csv, train_start_date, train_days=28, eval_days=7, max_episode_len=1000, window_size=20):
    df = load_df(csv)
    train_start_date = datetime.strptime(train_start_date, '%Y-%m-%d')
    train_end_date = train_start_date + timedelta(days=train_days + 1)  # exclusive
    eval_start_date = train_end_date
    eval_end_date = eval_start_date + timedelta(days=eval_days + 1)  # exclusive

    train_prices = get_prices(df, train_start_date, train_end_date)
    eval_prices = get_prices(df, eval_start_date, eval_end_date)

    train_env = trading_env.SingleAssetTradingEnv(train_prices, max_episode_len=max_episode_len, window_size=window_size)
    # eval_env = trading_env.SingleAssetTradingEnv(train_prices, max_episode_len=max_episode_len, window_size=window_size, is_eval=True)
    eval_env = trading_env.SingleAssetTradingEnv(eval_prices, max_episode_len=max_episode_len, window_size=window_size, is_eval=True)
    return train_env, eval_env


def train_and_eval(train_start_date, train_days, eval_days, max_episode_len, debug=False):
    # df = load_df('/home/z/data/zetaqubit/stock/alpaca/AAPL-5y-1Hour-2025-05-31.csv')
    # df = load_df('/home/z/data/zetaqubit/stock/alpaca/GOOG-5y-1Hour-2025-05-31.csv')
    # prices = jnp.arange(1, 2000)
    # prices = get_prices(df, start='2025-01-01')

    train_env, eval_env = get_train_eval_envs(
        '/home/z/data/zetaqubit/stock/alpaca/GOOG-5y-1Hour-2025-05-31.csv',
        train_start_date=train_start_date,
        train_days=train_days,
        eval_days=eval_days,
        max_episode_len=100,
    )

    if debug:
        train_fn = functools.partial(ppo.train, num_timesteps=2_000_000, num_evals=20, reward_scaling=1000, episode_length=max_episode_len, normalize_observations=False, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-5, entropy_cost=1e-2, num_envs=64, batch_size=128, seed=1)
    else:
        train_fn = functools.partial(ppo.train, num_timesteps=2_000_000, num_evals=100, reward_scaling=1000, episode_length=max_episode_len, normalize_observations=False, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-5, entropy_cost=1e-2, num_envs=2048, batch_size=1024, seed=1)

    xdata, ydata = [], []
    times = [datetime.now()]

    rl_rewards = []
    bh_rewards = []

    def cagr(return_frac, days):
        annualized_return = (1 + return_frac) ** (365 / days) - 1
        return annualized_return

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


    ctx = jax.disable_jit() if debug else nullcontext()
    with ctx:
        make_inference_fn, params, _ = train_fn(environment=train_env, eval_env=eval_env, progress_fn=progress)

    if len(times) > 3:
        print(f'avg rl reward (last 50 evals)', jnp.mean(jnp.array(rl_rewards[-50:])), '(first 50 evals)', jnp.mean(jnp.array(rl_rewards[:50])))
        print(f'avg bh reward (last 50 evals)', jnp.mean(jnp.array(bh_rewards[-50:])), '(first 50 evals)', jnp.mean(jnp.array(bh_rewards[:50])))
        print(f'CAGR rl (last 50 evals)', cagr(jnp.mean(jnp.array(rl_rewards[-50:])) - 1, days=eval_days), '(first 50 evals)', cagr(jnp.mean(jnp.array(rl_rewards[:50])) - 1, eval_days))

        print(f'time to jit: {times[1] - times[0]}')
        print(f'time to train: {times[-1] - times[1]}')


if __name__ == '__main__':
    train_and_eval(train_start_date='2024-09-01', train_days = 28, eval_days = 28, max_episode_len = 100)

