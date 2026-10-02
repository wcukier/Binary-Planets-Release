import numpy as np
from scipy.stats import describe
# pyplot is unused here -- the only references (plot_corner's savefig/close) are
# commented out.  Kept out of the import chain so batch jobs do not load matplotlib.
from os import sys

def init_log(n_log, n_particles):
    elements = np.zeros((n_log, n_particles-1, 6)) * np.nan
    distances = np.zeros(n_log) * np.nan
    # state vectors (x, y, z, vx, vy, vz, t) for each non-star particle
    states = np.zeros((n_log, n_particles-1, 7)) * np.nan

    return elements, distances, states, 0

def get_log_len(log):
    return log[0].shape[0]


def log_state_vectors(sim, states, t_step):
    """Record the position and velocity state vectors of every non-star
    (i.e. injected/secondary) particle for the current time step."""
    particles = sim.particles
    for i in range(len(particles))[1:]:
        p = particles[i]
        states[t_step, i-1] = [p.x, p.y, p.z, p.vx, p.vy, p.vz, sim.t]
    return states


def _bounds_halt(o, planet):
    if (o.a < 0) or (o.a > 5):
        return f"code=a_out_of_bounds;planet={planet};a={o.a}\t"
    if (o.e < 0) or (o.e > 1):
        return f"code=e_out_of_bounds;planet={planet};e={o.e}\t"
    return ""


def com_orbit(sim):
    """Orbit of the injected pair's (particles 1, 2) centre of mass about the star."""
    return sim.com(first=1, last=3).orbit(primary=sim.particles[0], G=sim.G)


def log_elements(sim, log, mode, a_com0=None):
    """a_com0: the pair's initial centre-of-mass semi-major axis; required for mode 2."""
    # sim.status()
    particles = sim.particles
    elements, distances, states, t_step = log
    n_log, n_particles, _ = elements.shape
    n_particles=len(particles)

    halt = ""
    for i in range(n_particles)[1:]:
        p = particles[i]
        if i == 0:
            primary = particles[2]
        # elif i==1:
        #     primary = particles[1]
        else:
            primary = particles[0]
            
        o = p.orbit(primary=primary)
        elements[t_step, i-1] = [o.a, o.e, o.inc, o.Omega, o.omega, sim.t]
        
        if p.m > 1e-15 and not (mode == 2 and i in (1, 2)):
            halt += _bounds_halt(o, i)

    # LLM-GENERATED — UNVERIFIED (tier: simulation): the pair's bounds and unbound tests
    # use its centre of mass.  A component's own heliocentric orbit includes its motion
    # about its partner, so the pair (particles 1, 2) is tested as one body.
    if mode == 2:
        halt += _bounds_halt(com_orbit(sim), "1,2")

    states = log_state_vectors(sim, states, t_step)

    d = particles[1] ** particles[2]
    distances[t_step] = d
    if mode == 2:
        if d > a_com0 / 2:
            halt += f"code=binary_unbound;planet=1,2;d={d}\t"


    t_step += 1
    # halt = o.a < 0
    return [elements, distances, states, t_step], halt

def save_log(log, file="output"):
    elements, distances, states, _ = log
    np.save(file+"/elements.npy", elements)
    np.save(file+"/distances.npy", distances)
    np.save(file+"/states.npy", states)

def calc_moments(log, file=None):
    elements, _, _, _= log
    n_log, n_particles, n_elements = elements.shape
    summary_stats = np.zeros((n_particles, n_elements-1, 4))
    for i in range(n_particles):
        for j in range(n_elements-1):
            summary_stats[i,j,:] = [i for i in describe(elements[:,i,j])][-4:]
    if file:
        np.save(file, summary_stats)
    return summary_stats

# def plot_corner(log, file):
#     elements, _, _ = log
#     corner.corner(np.hstack((elements[:,0,:5], elements[:,1,:5])), 
#                   range=[.999]*10, 
#                   labels=["a1", "e1", "inc1", "Omega1", "omega1", 
#                           "a2", "e2", "inc2", "Omega2", "omega2"])
#     plt.savefig(file)
#     plt.close()
    
def get_derivatives(log):
    elements, _, _, _= log
    decile = int(round(elements.shape[0] / 10))
    first_i = np.mean((elements[:decile-2, :, :5] - elements[2:decile, :, :5]) 
                      / np.mean(elements[:decile-2, :, 5] - elements[2:decile, :, 5]), 
                      axis=0)
    first_f = np.mean((elements[-decile:-2, :, :5] - elements[-decile+2:, :, :5]) 
                      / np.mean(elements[-decile:-2, :, 5] - elements[-decile+2:, :, 5]), 
                      axis=0)
    
    second_i = np.mean((elements[:decile-2, :, :5] -2*elements[1:decile-1, :, :5]
                        +elements[2:decile, :, :5]) 
                      / np.mean(elements[:decile-2, :, 5] - elements[2:decile, :, 5])**2/4, 
                      axis=0)
    
    second_f = np.mean((elements[-decile:-2, :, :5] - 2*elements[-decile+1:-1, :, :5]
                       + elements[-decile+2:, :, :5]) 
                    / np.mean(elements[-decile:-2, :, 5] - elements[-decile+2:, :, 5])**2/4, 
                    axis=0)
    
    return first_i, first_f, second_i, second_f