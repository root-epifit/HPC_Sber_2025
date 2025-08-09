import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from tqdm import tqdm

from memory_profiler import profile

c = 1.5
cc = 2.1
dd = 1.
gamma = cc - c*1j

d = 0.1
N = 12871
x_max = 128.0*np.pi

@profile
def generate_domains():
    if N%2 == 0:
        print("please make N an odd number")
        return 0, 0
    domain = np.linspace(-x_max, x_max, N)
    dx = (domain[1]-domain[0])
    f_domain = np.fft.fftfreq(domain.shape[0], dx)
    x = [domain]
    omega = [f_domain]
    print("dx:", dx)
    print("k_max: ", f_domain[len(f_domain)//2])
    print("k_min: ", f_domain[len(f_domain)//2+1])
    return x, omega

@profile
def initial_conditions(shape):
    A = np.ones(shape, dtype=np.complex128)
    A.imag = 0.
    return A

@profile
def add_perturb_1D(A, a, k, func, real, x):
    res = copy(A)
    if real:
        res.real += a*func(k*x)
    else:
        res.imag += a*func(k*x)
    return res

@profile
def add_noize(A):
    return A + 0.1*np.random.rand(*A.shape)

@profile
def non_linear_step(A, m, q, Om, i, dt):
    temp = np.empty(A.shape, dtype = float)
    potential = np.zeros(A.shape, dtype = float)
    for j in range(len(m)):
        potential += m[j]*np.cos(q[j]*numpy_x[j])
        
    potential *= np.cos(Om*i*dt)
    temp = np.exp(((1. - np.abs(A)**2)*gamma + 4.j*potential) * dt)
    return A*temp

@profile
def computation_loop(A, i, m, q, Om, multiplier, dt):
    fft_func = np.fft.fft if len(m)==1 else np.fft.fft2
    ifft_func = np.fft.ifft if len(m)==1 else np.fft.ifft2
    A = fft_func(A)
    A = A*multiplier
    A = ifft_func(A)
    A = non_linear_step(A, m, q, Om, i, dt)
    A = fft_func(A)
    A = A*multiplier
    return ifft_func(A)

@profile
def iterations(A, num, m, q, Om, a = 0, k = 0, func = 0, real = 0):
    dt = 0.1
    period = int(2*np.pi/Om/dt)
    dt = dt if (Om == 0) else 2*np.pi/Om/period

    n_frames = num//period
    img = 0

    multiplier = np.empty(numpy_omega[0].shape, dtype = np.complex128)
    temp = np.zeros(A.shape, dtype = np.complex128)
    for i in range(len(m)):
        temp +=numpy_omega[i]**2
    multiplier = np.exp(-(d+dd*1.j)*(temp)*dt/2)

    print("dt:", dt)
    print("period:", period, "n_frames:", n_frames)
    elems = tqdm(range(num))

    l_array = []
    total_solution = np.zeros((n_frames, A.shape[0]))
    
    for i in elems:
        if (i%period==period-1):
            total_solution[i//period] = np.copy(np.abs(A))

        A = computation_loop(A, i, m, q, Om, multiplier, dt)

    return A, total_solution, l_array

@profile
def plot_1D_heatmap(heatmap_1D):
    plt.figure(figsize=(6,5))
    plt.imshow(np.abs(heatmap_1D[:,1000:1000+heatmap_1D.shape[1]//16]),
               aspect='auto', origin="lower", cmap='binary_r', vmin=0.85, vmax=1.1)
    cb = plt.colorbar()
    cb.set_label(label='|A|', fontsize=12)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(10)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("t", fontsize=12)
    plt.tight_layout()
    plt.savefig("res.png")
    return

if __name__ == "__main__":
    ############################## 1 ##############################
    x, omega = generate_domains()
    numpy_x = np.copy(x)
    numpy_omega = np.copy(omega)
    A_init = initial_conditions(x[0].shape)
    A_noize = add_noize(A_init)
    A_device = np.copy(A_noize)
    ############################## 2 ##############################
    A_device, heatmap_1D, _ = iterations(A_device, num = 5*250,
                                         m = [0.0], q = [0.0], Om = 12.16)
    ############################## 3 ##############################
    plot_1D_heatmap(heatmap_1D)
    ###############################################################
    
