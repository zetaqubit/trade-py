"""Trading environment implemented in Jax."""

from brax import envs
from brax.envs import base as env
from flax import struct
import jax
import jax.numpy as jnp
from jax import lax


@struct.dataclass
class State:
    start_step: jax.Array
    step: jax.Array


class SingleAssetTradingEnv(env.Env):
    def __init__(self, prices: jnp.ndarray, max_episode_len: int = 1000, window_size: int = 5,
                 is_eval: bool = False):
        """
        prices: [T] array of prices
        window_size: number of previous prices in observation
        """
        assert prices.ndim == 1
        self.prices = prices
        self.window_size = window_size
        if is_eval:
            self.max_episode_len = len(prices) - window_size - 1
            self.max_start = 1
        else:
            self.max_episode_len = max_episode_len
            self.max_start = len(prices) - max_episode_len - window_size - 1  # leave space for window and step


    @property
    def observation_size(self):
        return 3 + self.window_size  # portfolio, buy_hold_portfolio, allocation, window

    @property
    def action_size(self):
        return 1

    @property
    def backend(self):
        return 'positional'


    def reset(self, rng: jnp.ndarray) -> env.State:
        start_step = jax.random.randint(rng, shape=(), minval=0, maxval=self.max_start)
        portfolio_value = jnp.array(1.0)
        buy_hold_portfolio_value = jnp.array(1.0)
        allocation = jnp.array(0.0)  # all cash at start

        state = State(start_step=start_step, step=start_step)
        obs = self._get_obs(start_step, allocation, portfolio_value, buy_hold_portfolio_value)
        metrics = {
            'allocation': 0.0,
            'max_alloc': 0.0,
            'buy_hold_reward': 0.0,
        }
        return env.State(pipeline_state=state, obs=obs, reward=jnp.array(0.0), done=jnp.array(0.0),
                         metrics=metrics)

    def step(self, state: env.State, action: jnp.ndarray) -> env.State:
        internal_state = state.pipeline_state
        step = internal_state.step.astype(jnp.int32)
        portfolio_value, buy_hold_portfolio_value = state.obs[1], state.obs[2]
        prev_allocation = state.obs[-1]

        # Clip action to [0, 1]
        allocation = jnp.clip(action[0], 0.0, 1.0)
        cash_allocation = 1.0 - allocation

        # Get prices at t and t+1
        price_t = self.prices[step + self.window_size - 1]
        price_tp1 = self.prices[step + self.window_size]
        asset_return = price_tp1 / price_t

        # Compute new portfolio value
        weighted_return = prev_allocation * asset_return + (1.0 - prev_allocation) * 1.0
        new_portfolio_value = portfolio_value * weighted_return

        def get_reward(new_v, prev_v):
            # return new_v - prev_v
            return jnp.log(new_v / prev_v)

        new_buy_hold_portfolio_value = buy_hold_portfolio_value * asset_return
        buy_hold_reward = get_reward(new_buy_hold_portfolio_value, buy_hold_portfolio_value)

        reward = get_reward(new_portfolio_value, portfolio_value)

        # Advance step
        step = step + 1
        done = jnp.where(step - internal_state.start_step >= self.max_episode_len, 1.0, 0.0)
        obs = self._get_obs(step, allocation, new_portfolio_value, new_buy_hold_portfolio_value)

        metrics = state.metrics | {
            'allocation': allocation,
            'max_alloc': 1.0,
            'buy_hold_reward': buy_hold_reward,
        }

        return state.replace(
            pipeline_state=internal_state.replace(step=step),
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    def _get_obs(self, step, allocation, portfolio_value, buy_hold_portfolio_value):
        # Dynamic slice of shape (window_size,)
        price_window = lax.dynamic_slice(self.prices, (step,), (self.window_size,))
        return jnp.concatenate([
            jnp.array([portfolio_value]),
            jnp.array([buy_hold_portfolio_value]),
            price_window,
            jnp.array([allocation]),
        ])

envs.register_environment('single_asset', SingleAssetTradingEnv)

