import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import LIF_Neuron

def static_plot(T_total:int, lif:LIF_Neuron):
    time, voltages, spikes = lif.simulate(T_total)
    plt.figure(figsize=(12, 6))
    plt.plot(time, voltages, label="Membrane potential (V)")
    plt.scatter(spikes, [lif.threshold] * len(spikes), color='red', marker='x', label="Spikes")
    plt.axhline(y=lif.threshold, color='gray', linestyle='--', label="Threshold")
    plt.title("LIF Neuron Simulation")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def dynamic_plot(T_total:int, lif:LIF_Neuron):
    time, voltages, spikes = lif.simulate(T_total)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, T_total)
    ax.set_ylim(min(voltages) - 5, max(voltages) + 5)
    line, = ax.plot([], [], lw=2)
    threshold_line = ax.axhline(y=lif.threshold, color='gray', linestyle='--')
    scatter = ax.scatter([], [], color='red', marker='x')

    def init():
        line.set_data([], [])
        scatter.set_offsets(np.empty((0, 2)))
        return line, scatter


    def update(frame):
        x = time[:frame]
        y = voltages[:frame]
        line.set_data(x, y)
        spike_pts = np.array([[t, lif.threshold] for t in spikes if t <= time[frame - 1]])
        if len(spike_pts) > 0:
            scatter.set_offsets(spike_pts)
        return line, scatter

    ani = animation.FuncAnimation(fig, update, frames=len(time), init_func=init,
                                blit=True, interval=5, repeat=True)

    plt.title("LIF Neuron Membrane Potential Animation")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def no_current():
    def my_current(x):
        return 0
    
    lif = LIF_Neuron(lambda x:my_current(x))
    T_total = 100
    dynamic_plot(T_total, lif)
 
def impulse_current_no_spike():
    def my_current(x):
        if x<25:return 0
        else: return 0.8
    
    lif = LIF_Neuron(lambda x:my_current(x), -65)
    T_total = 100
    dynamic_plot(T_total, lif)

def impulse_current_with_spike():
    def my_current(x):
        if x<25:return 0
        else: return 1.5
    
    lif = LIF_Neuron(lambda x:my_current(x), -65)
    T_total = 100
    dynamic_plot(T_total, lif)
 
def variable_current():
    def my_current(x):
        return np.sin(x)**2 + 1.5
    
    lif = LIF_Neuron(lambda x:my_current(x), -65)
    T_total = 100
    dynamic_plot(T_total, lif)
    
#no_current()
#impulse_current_no_spike()
#impulse_current_with_spike()
#variable_current()