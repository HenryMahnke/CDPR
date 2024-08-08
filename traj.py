import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory(t, x_des, x_des_dot, a_des, title):
    plt.figure(figsize=(12, 8))
    
    # Plot desired positions
    plt.subplot(3, 1, 1)
    plt.plot(t, x_des[:, 1], label='x')
    plt.plot(t, x_des[:, 2], label='y')
    plt.plot(t, x_des[:, 3], label='z')
    # plt.plot(t, x_des[:, 3], label='w')
    plt.title(f'{title} - Desired Position')
    plt.legend()
    
    # Plot desired velocities
    plt.subplot(3, 1, 2)
    # plt.plot(t, x_des_dot[:, 1], label='dx/dt')
    # plt.plot(t, x_des_dot[:, 2], label='dy/dt')
    plt.plot(t, x_des_dot[:, 3], label='dz/dt')
    # plt.plot(t, x_des_dot[:, 3], label='dw/dt')
    plt.title(f'{title} - Desired Velocity')
    plt.legend()
    
    # Plot desired accelerations
    plt.subplot(3, 1, 3)
    # plt.plot(t, a_des[:, 0], label='a_x')
    # plt.plot(t, a_des[:, 1], label='a_y')
    plt.plot(t, a_des[:, 2], label='a_z')
    plt.title(f'{title} - Desired Acceleration')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def setuptraj(case_to_run, const):
    delt = 0.001
    T = const['T']
    t = np.arange(0, T + delt, delt)

    x_des = np.zeros((len(t), 4))
    x_des_dot = np.zeros((len(t), 4))
    a_des = np.zeros((len(t), 3))
    a_des_dot = np.zeros((len(t), 3))

    if case_to_run == 1:
        T1 = const['T1']
        t10 = np.arange(0, T1 + delt, delt)
        x10 = np.column_stack((t10,np.zeros((len(t10),2)), const['upAmp'] * t10 / T1 + const['zInit']))
        print(x10)
        z1 = const['zInit'] + const['upAmp']
        print(z1)
        x_des[:len(t10), :] = x10
        x_des[len(t10):, 3] = z1
        print(x_des)
        
        x10_dot = np.column_stack((t10, np.zeros((len(t10),2)), const['upAmp']/T1 * np.ones((len(t10),1))))
        x_des_dot[:len(t10), :] = x10_dot
        x_des_dot[len(t10):, 3] = 0
        
        plot_trajectory(t, x_des, x_des_dot, a_des, 'Up and Plateau')

    elif case_to_run == 2:
        T1 = const['T1']
        t2_temp = np.arange(T1, T + delt, delt)
        t2 = np.arange(0, len(t2_temp) * delt, delt)
        
        zHarmonic = const['zSinAmp'] * np.sin(2 * np.pi * const['zSinFre'] * t2)
        zHarmonic_dot = 2 * np.pi * const['zSinFre'] * const['zSinAmp'] * np.cos(2 * np.pi * const['zSinFre'] * t2)
        
        if len(t2) > len(x_des[int(T1/delt):, 3]):
            t2 = t2[:len(x_des[int(T1/delt):, 3])]
            zHarmonic = zHarmonic[:len(x_des[int(T1/delt):, 3])]
            zHarmonic_dot = zHarmonic_dot[:len(x_des[int(T1/delt):, 3])]
        
        x_des[int(T1/delt):, 3] += zHarmonic
        x_des_dot[int(T1/delt):, 3] = zHarmonic_dot
        
        plot_trajectory(t, x_des, x_des_dot, a_des, 'Vertical Sine')
    
    elif case_to_run == 3:
        T1 = const['T1']
        t2_temp = np.arange(T1, T + delt, delt)
        t2 = np.arange(0, len(t2_temp) * delt, delt)
        
        A = const['angAmp'] / 180 * np.pi
        lam = const['lam']
        f = const['frq']
        
        start_idx = int(T1 / delt)
        end_idx = start_idx + len(t2)
        
        if end_idx > len(a_des):
            end_idx = len(a_des)
            t2 = t2[:(end_idx - start_idx)]
            a_des = a_des[:end_idx, :]
            a_des_dot = a_des_dot[:end_idx, :]
            x_des = x_des[:end_idx, :]
            x_des_dot = x_des_dot[:end_idx, :]
        
        a_des[start_idx:end_idx, 1] = A * np.sin(2 * np.pi * f * t2)
        a_des[start_idx:end:end, 2] = A * np.cos(2 * np.pi * f * t2) * (1 - np.exp(-lam * t2))
        
        a_des_dot[start_idx:end_idx, 1] = 2 * np.pi * f * A * np.cos(2 * np.pi * f * t2)
        a_des_dot[start_idx:end_idx, 2] = A * lam * np.exp(-lam * t2) * np.cos(2 * np.pi * f * t2) + 2 * A * f * np.pi * np.sin(2 * np.pi * f * t2) * (np.exp(-lam * t2) - 1)
        
        x_des[start_idx:end_idx, 0] = const['upAmp'] * np.sin(2 * np.pi * f * t2)
        x_des[start_idx:end_idx, 1] = const['zInit'] + A * np.cos(2 * np.pi * f * t2)
        
        x_des_dot[start_idx:end_idx, 0] = 2 * np.pi * f * const['upAmp'] * np.cos(2 * np.pi * f * t2)
        x_des_dot[start_idx:end:end, 1] = -2 * np.pi * f * A * np.sin(2 * np.pi * f * t2)
        
        plot_trajectory(t, x_des, x_des_dot, a_des, 'Top Motion')
    
    elif case_to_run == 4:
        T1 = const['T1']
        t2_temp = np.arange(T1, T + delt, delt)
        t2 = np.arange(0, len(t2_temp) * delt, delt)
        
        A = const['cirR']
        f = const['frq']
        lam = const['lam']
        
        x_des[int(T1/delt):, 1] = A * np.sin(2 * np.pi * f * t2)
        x_des[int(T1/delt):, 3] = A * np.cos(2 * np.pi * f * t2) * (1 - np.exp(-lam * t2)) + const['upAmp'] + const['zInit']
        
        x_des_dot[int(T1/delt):, 1] = 2 * np.pi * f * A * np.cos(2 * np.pi * f * t2)
        x_des_dot[int(T1/delt):, 3] = A * lam * np.exp(-lam * t2) * np.cos(2 * np.pi * f * t2) + 2 * A * f * np.pi * np.sin(2 * np.pi * f * t2) * (np.exp(-lam * t2) - 1)
        
        plot_trajectory(t, x_des, x_des_dot, a_des, 'XZ Circle')
    
    elif case_to_run == 5:
        T1 = const['T1']
        t2_temp = np.arange(T1, T + delt, delt)
        t2 = np.arange(0, len(t2_temp) * delt, delt)
        
        A = const['cirR']
        f = const['cirfrq']
        zInit = const['zInit']
        
        x_des[int(T1/delt):, 0] = A * np.sin(2 * np.pi * f * t2)
        x_des[int(T1/delt):, 1] = A * np.cos(2 * np.pi * f * t2) + zInit
        
        x_des_dot[int(T1/delt):, 0] = 2 * np.pi * f * A * np.cos(2 * np.pi * f * t2)
        x_des_dot[int(T1/delt):, 1] = -2 * np.pi * f * A * np.sin(2 * np.pi * f * t2)
        
        plot_trajectory(t, x_des, x_des_dot, a_des, 'Circle in XY Plane')

    else:
        raise ValueError('Unknown case number')
    
    return x_des, x_des_dot, a_des, a_des_dot

# Constants for the case
const = {
    'T': 60,
    'T1': 10,
    'upAmp': 0.5,
    'zInit': 0.4,
    'zSinAmp': 0.5,
    'zSinFre': 0.5,
    'angAmp': 10,
    'lam': 1,
    'frq': 0.1,
    'cirR': 1,
    'cirfrq': 0.1
}

# Example usage
case_to_run = 1  # Set this to the desired case number
setuptraj(case_to_run, const)
