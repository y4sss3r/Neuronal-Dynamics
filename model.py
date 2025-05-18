import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class LIF_Neuron:
    def __init__(self, current_function=None, v_init=-56, v_rest=-65.0, R=10.0, threshold=-55.0, tau=10.0, refract_time=5.0, dt=0.1):
        self.v_rest = v_rest
        self.v = v_init
        self.tau = tau
        self.R = R
        self.dt = dt
        self.threshold = threshold
        self.refract_time = refract_time
        
        if current_function is None:
            self.current_function= lambda x:0
        else:
            self.current_function=current_function

        self.refract_counter = 0
        self.spikes = []

    def get_input_current(self, t):
        return self.current_function(t)

    def reset(self):
        self.v = self.v_rest
        self.refract_counter = 0
        self.spikes.clear()

    def step(self, t):
        if self.refract_counter > 0:
            self.refract_counter -= 1
            return self.v_rest

        I_t = self.get_input_current(t)
        dv = (-(self.v - self.v_rest) + self.R * I_t) * (self.dt / self.tau)
        self.v += dv

        if self.v >= self.threshold:
            self.v = self.v_rest
            self.refract_counter = int(self.refract_time / self.dt)
            self.spikes.append(t)

        return self.v

    def simulate(self, T):
        time_points = np.arange(0, T, self.dt)
        voltages = [self.step(t) for t in time_points]
        return time_points, voltages, self.spikes
