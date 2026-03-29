import gym
import numpy as np
from gym import spaces
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class IPMSMMotor:
    def __init__(self, params):
        self.R_s = params['R_s']  # stator resistance
        self.L_d = params['L_d']  # d-axis inductance
        self.L_q = params['L_q']  # q-axis inductance
        self.Psi_f = params['Psi_f']  # PM flux linkage
        self.J = params['J']  # inertia
        self.B = params['B']  # damping
        self.p = params['p']  # pole pairs

    def dynamics(self, t, x, v_d, v_q, T_L):
        i_d, i_q, omega, theta = x
        di_d_dt = (1 / self.L_d) * (v_d - self.R_s * i_d + omega * self.L_q * i_q)
        di_q_dt = (1 / self.L_q) * (v_q - self.R_s * i_q - omega * self.L_d * i_d - omega * self.Psi_f)
        domega_dt = (self.p / self.J) * (self.Psi_f * i_q + (self.L_d - self.L_q) * i_d * i_q - T_L - self.B * omega)
        dtheta_dt = omega
        return [di_d_dt, di_q_dt, domega_dt, dtheta_dt]

    def step(self, x, v_d, v_q, T_L, dt):
        sol = solve_ivp(self.dynamics, [0, dt], x, args=(v_d, v_q, T_L), method='RK45')
        return sol.y[:, -1]

class IPMSMEnv(gym.Env):
    def __init__(self):
        super(IPMSMEnv, self).__init__()
        # Motor parameters (typical values)
        params = {
            'R_s': 0.5,  # Ohm
            'L_d': 0.004,  # H
            'L_q': 0.008,  # H
            'Psi_f': 0.1,  # Wb
            'J': 0.01,  # kg*m^2
            'B': 0.001,  # N*m*s
            'p': 3  # pole pairs
        }
        self.motor = IPMSMMotor(params)
        self.dt = 1e-4  # 10 kHz
        self.t_max = 1.0  # 1 second episode
        self.time = 0
        self.omega_ref = 100 * np.pi  # 100 rad/s reference speed
        self.T_L = 0  # load torque (can be varied)

        # Normalization factors
        self.I_max = 20  # A
        self.omega_max = 200 * np.pi  # rad/s
        self.V_max = 400 / np.sqrt(3)  # V

        # State space: [i_d, i_q, omega, theta] normalized
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Action space: [v_d, v_q] normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.state = None
        self.reset()

    def reset(self):
        # Initial state: small random currents, zero speed, random angle
        i_d = np.random.uniform(-0.1, 0.1)
        i_q = np.random.uniform(-0.1, 0.1)
        omega = 0
        theta = np.random.uniform(0, 2*np.pi)
        self.state = np.array([i_d, i_q, omega, theta])
        self.time = 0
        return self._get_obs()

    def _get_obs(self):
        i_d, i_q, omega, theta = self.state
        return np.array([
            i_d / self.I_max,
            i_q / self.I_max,
            omega / self.omega_max,
            theta / (2 * np.pi)
        ], dtype=np.float32)

    def step(self, action):
        v_d_norm, v_q_norm = action
        v_d = v_d_norm * self.V_max
        v_q = v_q_norm * self.V_max

        # Simulate one step
        self.state = self.motor.step(self.state, v_d, v_q, self.T_L, self.dt)
        self.time += self.dt

        obs = self._get_obs()
        reward = self._calculate_reward()
        done = self._is_done()

        return obs, reward, done, {}

    def _calculate_reward(self):
        i_d, i_q, omega, theta = self.state
        omega_error = (self.omega_ref - omega) ** 2
        current_penalty = (i_d ** 2 + i_q ** 2) * 0.1
        voltage_penalty = abs(i_d) * 0.05  # approximate
        reward = -10 * omega_error - current_penalty - voltage_penalty
        return reward

    def _is_done(self):
        i_d, i_q, omega, theta = self.state
        if abs(omega) > self.omega_max or abs(i_d) > self.I_max or abs(i_q) > self.I_max:
            return True
        if self.time >= self.t_max:
            return True
        return False

    def render(self, mode='human'):
        # Simple plot (for debugging)
        pass

    def close(self):
        pass