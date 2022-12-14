"Ising model simulation of a two dimensional lattice of spins"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal
from numpy import exp, sinh, log, sqrt
from matplotlib import colors
from scipy.optimize import curve_fit
import time

"""
Create initial state of randomly orientated spins (+1 or -1)
"""
initialiseRandom = lambda N, T: 2 * np.random.randint(2, size=(N, N)) - 1

"""
Create initial state of up orientated spins (+1)
"""
initialiseOnes = lambda N, T: np.ones((N, N), dtype=int)

"""
Depending on temperature create initial state of up or randomly orientated spins 
"""
initialiseByT = lambda N, T: initialiseRandom(N, T) if T >= 2.28 else initialiseOnes(N, T)


def spin_flip_energy(N, J, mu, H, state, i, j):
    """
        Compute energy to flip each state (using periodic boundary conditions)

    :param N: number sites, giving a lattice of N^2 spins
    :param J: exchange energy
    :param mu: magnetic permeability
    :param H: magnetic field strength
    :param state: configuration of the lattice of spins
    :param i: row
    :param j: column
    :return: energy required to flip the spin
    """
    # set boundary conditions

    left_state, right_state = state[i][(j - 1) % N], state[i][(j + 1) % N]
    top_state, bottom_state = state[(i - 1) % N][j], state[(i + 1) % N][j]
    sum_Sj = left_state + right_state + top_state + bottom_state

    return 2 * J * state[i][j] * sum_Sj + mu * H * state[i][j]


def boltzmann_factor(delta_E, T, H, boltz4J, boltz8J):
    """
        Computes the boltzmann factor

    :param delta_E: energy required to flip the spin
    :param T: temperature
    :param H: magnetic field strength
    :param boltz4J: pre-set possible values of the boltzmann factor for optimisation
    :param boltz8J: pre-set possible values of the boltzmann factor for optimisation
    :return: the boltzmann factor for the given temperature and temperature and energy
    """
    boltzmann = 0
    if H == 0:
        if delta_E == 0:
            boltzmann = 1.0
        elif delta_E == 4:
            boltzmann = boltz4J
        elif delta_E == 8:
            boltzmann = boltz8J
    else:
        boltzmann = exp(-delta_E / T)
    return boltzmann


def state_change(N, state, T, J, mu, H):
    """
        Evolves the lattice in 'time' with one Monte Carlo sweep

    :param N: number sites, giving a lattice of N^2 spins
    :param state: configuration of the lattice of spins
    :param T: temperature
    :param J: exchange energy
    :param mu: magnetic permeability
    :param H: magnetic field strength
    :return: updated state after one Monte Carlo sweep
    """

    # possible values of boltzmann factor: boltz4 corresponds to boltzmann factor for an
    # energy change of 4J, only possible positive energy changes are 0J, 4J and 8J.
    boltz4J = exp(- 4 / T)
    boltz8J = exp(- 8 / T)

    # create arrays of random numbers to randomly select sites as we step through the lattice.
    iRnd = np.random.randint(N, size=N ** 2)
    jRnd = np.random.randint(N, size=N ** 2)
    pRnd = np.random.uniform(0, 1, size=N ** 2)

    for k in range(N ** 2):
        i, j = iRnd[k], jRnd[k]
        delta_E = spin_flip_energy(N, J, mu, H, state, i, j)
        p = pRnd[k]
        boltzmann = boltzmann_factor(delta_E, T, H, boltz4J, boltz8J)
        if delta_E < 0:
            state[i][j] *= -1
        elif boltzmann >= p:
            state[i][j] *= -1
    return state


def energy(N, state, J, mu, H):
    """
        Computes the total energy of the lattice

    :param N: number sites, giving a lattice of N^2 spins
    :param state: configuration of the lattice of spins
    :param J: exchange energy
    :param mu: magnetic permeability
    :param H: magnetic field strength
    :return: total energy of the lattice
    """
    energy = 0
    for i in np.arange(0, N):
        for j in np.arange(0, N):
            left_state, right_state = state[i][(j - 1) % N], state[i][(j + 1) % N]
            top_state, bottom_state = state[(i - 1) % N][j], state[(i + 1) % N][j]
            sum_Sj = left_state + right_state + top_state + bottom_state
            E_site = -J * state[i][j] * sum_Sj - mu * H * state[i][j]
            energy += E_site
    return energy / 4  # factor of 1/4 accounts for over-counting counting of interaction energies


def find_magnetisation(initialise, T, N, num_sweeps, J, mu, H, equilibrate):
    """
        Equilibrates the lattice then computes the average magnetisation per site

    :param initialise: initialise the lattice
    :param T: temperature
    :param N: number sites, giving a lattice of N^2 spins
    :param num_sweeps: number of sweeps to average over
    :param J: exchange energy
    :param mu: magnetic permeability
    :param H: magnetic field strength
    :param equilibrate: number of sweeps to equilibrate the lattice
    :return: magnitude of the average magnetisation per site, an array of the samples
             of the absolute magnetisations
    """
    mag_vector = np.zeros(num_sweeps)
    state = initialise(N, T)

    for eq in range(equilibrate):
        state_change(N, state, T, J, mu, H)

    for sweep in range(num_sweeps):
        state = state_change(N, state, T, J, mu, H)
        mag_vector[sweep] = abs(np.mean(state))

    magnetisation = np.mean(mag_vector)

    return magnetisation, mag_vector


def autocorrelation(initialise, T, N, taulength, num_sweeps, J, mu, H, equilibrate):
    """
        Computes the autocorrelation and lag time

    :param initialise: initialise the lattice
    :param T: temperature
    :param N: number sites, giving a lattice of N^2 spins
    :param taulength: time scale to show magnetisation autocorrelation over
    :param num_sweeps: number of sweeps to average over
    :param J: exchange energy
    :param mu: magnetic permeability
    :param H: magnetic field strength
    :param equilibrate: number of sweeps to equilibrate the system
    :return: magnetisation autocorrelation vector, time lag - the number of sweeps for autocorrelation to
             decay to 1/e of the initial value
    """

    mean_mag = find_magnetisation(initialise, T, N, num_sweeps, J, mu, H, equilibrate)[0]
    mag_vector = np.array(find_magnetisation(initialise, T, N, num_sweeps, J, mu, H, equilibrate)[1])
    tau_range = np.array(range(taulength))
    A = np.zeros(taulength)
    tau_efold = []
    Mtau = mag_vector[taulength:]
    Mtau_prime = Mtau - mean_mag

    A[0] = abs(np.mean(np.square(Mtau_prime)))

    for tau in range(1, taulength):
        # M is magnetisation snapshot array truncated by tau elements, Mtau is the snapshot array without the
        # first tau elements, and the primes correspond to taking the fluctuations of these about the mean
        Mprime = mag_vector[taulength - tau_range[tau]: - tau_range[tau]] - mean_mag

        A[tau] = abs(np.mean(np.multiply(Mprime, Mtau_prime)))
        tau_efold.append(A[tau])

    # Now normalise to get autocorrelation
    a = A / A[0]

    # etau (the time lag) indicates the time required for fluctuations of the lattice about the mean to
    # become negligible so gives an indication of the number of sweeps we would need to average over to get
    # accurate readings
    etau = curve_fit(lambda x, b: exp(- x / b), tau_range, a)[0]

    return a, etau


def hysteresis_eq(N, T, J, mu, equilibrate, num_sweeps, Hrange):
    """
        Provides the magnetisation of the system each time the state is equilibrated at a different H

    :param N: number sites, giving a lattice of N^2 spins
    :param T: temperature
    :param J: exchange energy
    :param mu: magnetic permeability
    :param equilibrate: number of sweeps to equilibrate the system
    :param num_sweeps: number of sweeps to average over
    :param Hrange: dictates whether H goes from negative to positive, or positive to negative
    """
    state = initialiseRandom(N, T)
    mag = np.zeros(len(Hrange))
    for i in range(len(Hrange)):
        H = Hrange[i]
        for eq in range(equilibrate):
            state_change(N, state, T, J, mu, H)

        m = np.zeros(num_sweeps)
        for sweep in range(num_sweeps):
            state_change(N, state, T, J, mu, H)
            m[sweep] = np.mean(state)
        mag[i] = np.mean(m)

    return mag


def hysteresis(J, mu, num_sweeps=80, N=16, T=1.6, equilibrate=100):
    """
        Shows how the magnetisation changes as the field is slowly vaired (i.e. allowing the system to
        equilibrate each time the field is changed)

    :param J: exchange energy
    :param mu: magnetic permeability
    :param num_sweeps: number of sweeps to average over
    :param N: number sites, giving a lattice of N^2 spins
    :param T: temperature
    :param equilibrate: number of sweeps to equilibrate the system
    """
    H_increase = np.arange(-2, 2, 0.1)

    # Reverse the array such that the field decreases from a positive value to a negative one
    H_decrease = H_increase[::-1]

    mag = hysteresis_eq(N, T, J, mu, equilibrate, num_sweeps, H_increase)
    plt.plot(H_increase, mag, 'r', label='Increasing H')

    mag = hysteresis_eq(N, T, J, mu, equilibrate, num_sweeps, H_decrease)
    plt.plot(H_decrease, mag, 'b', label='Decreasing H')

    plt.legend(loc='best')
    plt.xlabel('External field strength [H]', fontsize='12')
    plt.ylabel('Magnetisation per site [' + r'$\mu$]', fontsize='12')
    plt.tick_params(direction='in', which='major')
    plt.title('Hysteresis loop')


def external_field_mag(J, mu, N=20):
    """
        Shows how the absolute magnetisation per site varies with external field strength

    :param J: exchange energy
    :param mu: magnetic permeability
    :param N: number sites, giving a lattice of N^2 spins
    """
    H_field = np.append(np.arange(0, 0.1, 0.01), np.arange(0.1, 1, 0.1))
    T_range = np.array([1.8, 2.3, 3])
    H_length = len(H_field)
    T_length = len(T_range)

    for i in range(T_length):
        print(str(i) + '/' + str(len(T_range)))
        mag = np.zeros(H_length)
        temp = T_range[i]
        if temp > 2.2 and temp < 2.6:
            num_sweeps = 4000
            equilibrate = N ** 2
        else:
            num_sweeps = 400
            equilibrate = int(0.25 * N ** 2)

        for j in range(H_length):
            H = H_field[j]
            mag[j] = find_magnetisation(initialiseByT, temp, N, num_sweeps, J, mu, H, equilibrate)[0]
        H_filter = scipy.signal.savgol_filter(mag, 9, 2)

        plt.plot(H_field, H_filter, label='T = ' + str(temp))
    plt.legend(loc='best')
    plt.xlabel('External field strength [H]', fontsize='12')
    plt.ylabel('Magnetisation per site [' + r'$\mu$]', fontsize='12')
    plt.tick_params(direction='in', which='major')
    plt.title('Magnetisation vs. External field strength')


def finite_scaling():
    """
        Shows how the critical temperature varies with lattice size, and provides and estimate of the true Tc
        (i.e. in the limit of an infinitely large lattice).

    """
    # Critical temperatures
    N = np.array([16, 20, 24, 36, 48, 64])
    T_mean = np.array([2.314, 2.304, 2.288, 2.284, 2.280, 2.275])
    T_err = np.array([0.026, 0.018, 0.016, 0.016, 0.014, 0.015])

    popt, var = curve_fit(lambda x, Tc_inf, a, nu: Tc_inf + a * x ** (-1 / nu), N, T_mean, sigma=T_err)
    Tc_inf, a, nu = popt
    Tc_err, a_err, nu_err = np.sqrt(np.diag(var))

    N_range = np.arange(1, 85, 0.1)
    T_fit = Tc_inf + a * np.power(N_range, (-1 / nu))

    plt.errorbar(N, T_mean, yerr=T_err, fmt='o', color='r', markersize=4, capsize=5, ecolor='r')
    plt.plot(N_range, T_fit, 'b', label='Finite-scaling curve fit')
    plt.axhline(y=2.269, color='k', linestyle='--', label='T = 2.269, Onsagers exact result')
    plt.legend(loc='best')
    plt.ylim(2.25, 2.5)
    plt.xlim(0, 85)
    plt.xlabel('Number of sites (NxN lattice)', fontsize='12')
    plt.ylabel('Tc [J/k' + r'$\ _B$]', fontsize='12')
    plt.tick_params(direction='in', which='major')
    plt.title('Critical temperature vs. Lattice size')
    print('Tc = ' + str(Tc_inf) + ' +/- ' + str(Tc_err))
    print('nu = ' + str(nu) + ' +/- ' + str(nu_err))


def autocorrelation3(J, mu, H, taulength=50):
    """
        Shows how the time lag varies for different temperatures and lattice sizes together

    :param J: exchange energy
    :param mu: magnetic permeability
    :param H: magnetic field strength
    :param taulength: time scale to show magnetisation autocorrelation over
    """
    T_range = np.append(np.append(np.arange(2.0, 2.1, 0.1), np.arange(2.15, 2.4, 0.05)), np.arange(2.4, 3, 0.1))
    lag_time = np.zeros(len(T_range))
    N_range = np.array([16, 24])

    for N in N_range:
        for i in range(len(T_range)):
            temp = T_range[i]
            if temp > 2.1 and temp < 2.5:
                num_sweeps = 50000
                equilibrate = 4 * N ** 2
            else:
                num_sweeps = 10000
                equilibrate = int(N ** 2)
            print(str(i) + '/' + str(len(T_range)))
            lag_time[i] = autocorrelation(initialiseByT, temp, N, taulength, num_sweeps, J, mu, H, equilibrate)[1]
        plt.plot(T_range, lag_time, 'o', linestyle='-', label='N=' + str(N))
    plt.xlabel('Temperature [J / k' + r'$\ _B$]', fontsize='12')
    plt.ylabel('Time lag (equilibration time)', fontsize='12')
    plt.legend(loc='best')
    plt.tick_params(direction='in', which='major')
    plt.title('Autocorrelation time lag vs. Temperature')


def autocorrelation2(J, mu, H, num_sweeps=2000, T=2, equilibrate=600, taulength=40):
    """
        Shows exponential decay of autocorrelation for a set temperature and different N's

    :param J: exchange energy
    :param mu: magnetic permeability
    :param H: magnetic field strength
    :param num_sweeps: number of sweeps to average over
    :param T: temperature
    :param equilibrate: number of sweeps to equilibrate the system
    :param taulength: time scale to show magnetisation autocorrelation over
    """
    tau_range = np.array(range(taulength))
    N_range = np.array([16, 24])

    for N in N_range:
        a = autocorrelation(initialiseByT, T, N, taulength, num_sweeps, J, mu, H, equilibrate)[0]
        plt.plot(tau_range, a, label='N=' + str(N))

    plt.legend(loc='best')
    plt.xlabel('Time (tau)', fontsize='12')
    plt.ylabel('Autocorrelation', fontsize='12')
    plt.tick_params(direction='in', which='major')


def autocorrelation1(J, mu, H, num_sweeps=2000, N=16, equilibrate=1000, taulength=50):
    """
        Shows exponential decay of autocorrelation for a set N and different temperatures

    :param J: exchange energy
    :param mu: magnetic permeability
    :param H: magnetic field strength
    :param num_sweeps: number of sweeps to average over
    :param N: number sites, giving a lattice of N^2 spins
    :param equilibrate: number of sweeps to equilibrate the system
    :param taulength: time scale to show magnetisation autocorrelation over
    """
    tau_range = np.array(range(taulength))
    T_range = np.array([2, 2.3, 2.7])

    for i in range(len(T_range)):
        print(str(i) + '/' + str(len(T_range)))
        temp = T_range[i]
        a = autocorrelation(initialiseByT, temp, N, taulength, num_sweeps, J, mu, H, equilibrate)[0]
        plt.plot(tau_range, a, label='T=' + str(temp))

    plt.legend(loc='best')
    plt.xlabel('Time (tau)', fontsize='12')
    plt.ylabel('Autocorrelation', fontsize='12')
    plt.tick_params(direction='in', which='major')
    plt.title('Autocorrelation vs. Time')


def susceptibility_plot(J, mu, H, N=20):
    """
        Shows how magnetic susceptibility varies with temperature and provides an estimate of Tc

    :param J: exchange energy
    :param mu: magnetic permeability
    :param H: magnetic field strength
    :param N: number sites, giving a lattice of N^2 spins
    """
    T_range = np.append(np.append(np.arange(1.3, 2.1, 0.1), np.arange(2.1, 2.5, 0.05)), np.arange(2.5, 3.4, 0.1))
    chi = np.zeros(len(T_range))

    for i in range(len(T_range)):
        temp = T_range[i]
        print(str(i) + '/' + str(len(T_range)))

        if temp > 2.1 and temp < 2.6:
            num_sweeps = 4000
            equilibrate = N ** 2
        else:
            num_sweeps = 700
            equilibrate = int(0.25 * N ** 2)

        magnetisation, mag_vector = find_magnetisation(initialiseByT, temp, N, num_sweeps, J, mu, H, equilibrate)
        chi[i] = (np.std(mag_vector) ** 2) / temp

        if chi[i] == max(chi):
            imax = i

    Tc_estimate_SC = T_range[imax]
    print('Curie temperature estimate = ', Tc_estimate_SC)

    plt.xlabel('Temperature [J / k' + r'$\ _B$]', fontsize='12')
    plt.ylabel('Susceptibility ['r'$\mu^2$' + '/ J]', fontsize='12')
    plt.title('Susceptibility vs. Temperature')
    plt.tick_params(direction='in', which='major')
    plt.plot(T_range, chi, 'o')


def heat_capacity(J, mu, H, N=30):
    """
        Shows how heat capacity varies with tmeperature and provides and estimate of Tc

    :param J: exchange energy
    :param mu: magnetic permeability
    :param H: magnetic field strength
    :param N: number sites, giving a lattice of N^2 spins
    """
    T_range = np.append(np.append(np.arange(1.3, 2.1, 0.1), np.arange(2.1, 2.6, 0.02)), np.arange(2.6, 3.4, 0.1))
    spec_heat = np.zeros(len(T_range))

    for i in range(len(T_range)):
        print(str(i) + '/' + str(len(T_range)))
        temp = T_range[i]
        state = initialiseByT(N, temp)

        if temp > 2.1 and temp < 2.6:
            num_sweeps = 4000
            equilibrate = N ** 2
        else:
            num_sweeps = 600
            equilibrate = int(0.25 * N ** 2)

        energy_samples = np.zeros(num_sweeps)
        for eq in range(equilibrate):
            state_change(N, state, temp, J, mu, H)

        for sweep in range(num_sweeps):
            state_change(N, state, temp, J, mu, H)
            energy_samples[sweep] = energy(N, state, J, mu, H)

        spec_heat[i] = ((np.std(energy_samples)) / temp) ** 2

    C_filter = scipy.signal.savgol_filter(spec_heat, 9, 2)

    # Find T value corresponding to the maximum in the filtered heat capacity curve
    Tc = T_range[C_filter.argmax()]

    print('Tc for N = ' + str(N) + ' is', Tc)
    plt.plot(T_range, C_filter, 'b')
    plt.plot(T_range, spec_heat, 'o')
    plt.xlabel('Temperature [J / k' + r'$\ _B$]', fontsize='12')
    plt.ylabel('Heat Capacity [k' + r'$\ _B$]', fontsize='12')
    plt.title('Heat Capacity vs. Temperature for N = ' + str(N))
    plt.tick_params(direction='in', which='major')


def energy_plot(J, mu, H, N=36):
    """
        Shows how energy per site varies with temperature

    :param J: exchange energy
    :param mu: magnetic permeability
    :param H: magnetic field strength
    """
    T_range = np.append(np.append(np.arange(1.2, 2.1, 0.1), np.arange(2.1, 2.6, 0.05)), np.arange(2.6, 4, 0.1))

    Energy0 = []

    for i in range(len(T_range)):
        temp = T_range[i]
        state = initialiseByT(N, temp)
        E1 = 0
        print(str(i) + '/' + str(len(T_range)))

        if temp > 2.2 and temp < 2.6:
            num_sweeps = 3200
            equilibrate = N ** 2
        else:
            num_sweeps = 800
            equilibrate = int(0.025 * N ** 2)

        scale = 1 / (N * N * num_sweeps)

        for eq in range(equilibrate):
            state_change(N, state, temp, J, mu, H)

        for sweep in range(num_sweeps):
            state_change(N, state, temp, J, mu, H)
            Ene = energy(N, state, J, mu, H)
            E1 = E1 + Ene
        Energy0.append(E1 * scale)

    plt.plot(T_range, Energy0, 'o', linestyle='-', label='N=' + str(N))
    plt.tick_params(direction='in', which='major')
    plt.xlabel('Temperature [J / k' + r'$\ _B$]', fontsize='12')
    plt.ylabel('Energy [J]', fontsize='12')
    plt.title('Energy per site vs. Temperature')
    plt.legend(loc='best')


def mag_plot(J, mu, H, N=30):
    """
        Shows how magnetisation per site varies with temperature, and provides an estimate of Tc and beta

    :param J: exchange energy
    :param mu: magnetic permeability
    :param H: magnetic field strength
    :param N: number sites, giving a lattice of N^2 spins
    """
    T_range = np.append(np.append(np.arange(0.5, 2, 0.1), np.arange(2, 2.7, 0.05)), np.arange(2.8, 3.1, 0.1))

    mag = np.zeros(len(T_range))
    mag_err = np.zeros(len(T_range))
    log_err = np.zeros(len(T_range))
    mag_array = []
    for i in range(len(T_range)):
        temp = T_range[i]
        print(str(i) + '/' + str(len(T_range)))

        if temp == 2.25:
            num_sweeps = 10000
            equilibrate = N ** 2
        elif temp > 1.9 and temp < 2.7:
            num_sweeps = 5000
            equilibrate = N ** 2
        else:
            num_sweeps = 400
            equilibrate = int(0.25 * N ** 2)

        mag[i] = find_magnetisation(initialiseByT, temp, N, num_sweeps, J, mu, H, equilibrate)[0]
        mag_vector = find_magnetisation(initialiseByT, temp, N, num_sweeps, J, mu, H, equilibrate)[1]
        mag_err[i] = np.std(mag_vector)
        log_err[i] = mag_err[i] / mag[i]

        if i > 0:
            mag_diff = mag[i] - mag[i - 1]
            mag_array.append(mag_diff)
            steepest_slope = min(mag_array)
            if mag_diff == steepest_slope:
                imax = i

    T_onsager = 2.27
    T_range2 = np.append(np.arange(0.5, 2.26, 0.01), np.arange(2.26, 2.27, 0.001))
    M_analytic = (1 - (sinh(log(1 + sqrt(2))) * T_onsager / T_range2) ** -4) ** (1 / 8)

    coeffs, var = curve_fit(lambda x, beta, const: const + beta * x, log(2.86 - T_range[15:-13]),
                            log(mag[15:-13]), sigma=log_err[15:-13])
    beta = coeffs[0]
    beta_err = np.sqrt(np.diag(var))[0]

    print('Beta = ' + str(beta) + ' +/- ' + str(beta_err))
    Tc_estimate = T_range[imax]
    print('Critical temperature for N = ' + str(N) + ' from steepest slope is', Tc_estimate)

    plt.xlabel('Temperature [J / k' + r'$\ _B$]', fontsize='12')
    plt.ylabel('Magnetisation per site [' + r'$\mu$]', fontsize='12')
    plt.tick_params(direction='in', which='major')
    plt.plot(T_range, mag, 'o', linestyle='-', label='Simulation for N=' + str(N))
    plt.errorbar(T_range[15:-13], mag[15:-13], yerr=mag_err[15:-13], fmt='o', color='b', markersize=4, capsize=5,
                 ecolor='b')
    plt.plot(T_range2, M_analytic, color='black', linestyle='--', label='Onsagers analytic solution')
    plt.xlim(0.5, 3)
    plt.title('Magnetisation per site vs. Temperature')
    plt.legend(loc='best')


def mag_equilibrium(J, mu, H, num_sweeps=300, T=1.2, equilibrate=0):
    """
        Shows how the magnetisation varies as the system is brought to equilibrium

    :param J: exchange energy
    :param mu: magnetic permeability
    :param H: magnetic field strength
    :param num_sweeps: number of sweeps to average over
    :param T: temperature
    :param equilibrate: number of sweeps to equilibrate the system
    """
    N_range = np.array([20, 40])

    for N in N_range:
        mag_vector = find_magnetisation(initialiseRandom, T, N, num_sweeps, J, mu, H, equilibrate)[1]
        plt.plot(range(num_sweeps), mag_vector, label='N = ' + str(N))

    plt.xlabel('Time [number of sweeps]', fontsize='12')
    plt.ylabel('Magnetisation per site [' + r'$\mu$]', fontsize='12')
    plt.tick_params(direction='in', which='major')
    plt.title('Magnetisation per site vs. number of sweeps, T = ' + str(T))
    plt.legend(loc='best')


def plot_spins(J, mu, H, N=25, T=0.1, equilibrate=200):
    """
        Shows how the spins evolve

    :param J: exchange energy
    :param mu: magnetic permeability
    :param H: magnetic field strength
    :param N: number sites, giving a lattice of N^2 spins
    :param T: temperature
    :param equilibrate: number of sweeps to equilibrate the system
    """
    state = initialiseRandom(N, T)
    cmap = matplotlib.cm.seismic
    bounds = [-1, 0, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.subplot(231)
    plt.imshow(state, cmap=cmap, norm=norm)

    for sweep in range(equilibrate):
        state = state_change(N, state, T, J, mu, H)

        if sweep == int(0.05 * equilibrate):
            plt.subplot(232)
            plt.imshow(state, cmap=cmap, norm=norm)
        if sweep == int(0.1 * equilibrate):
            plt.subplot(233)
            plt.imshow(state, cmap=cmap, norm=norm)
        if sweep == int(0.2 * equilibrate):
            plt.subplot(234)
            plt.imshow(state, cmap=cmap, norm=norm)
        if sweep == int(0.5 * equilibrate):
            plt.subplot(235)
            plt.imshow(state, cmap=cmap, norm=norm)
        if sweep == int(equilibrate - 1):
            plt.subplot(236)
            plt.imshow(state, cmap=cmap, norm=norm)


def simulate(simulation, J=1, H=0, mu=1):
    start = time.time()
    if simulation == "spin_plot":
        plot_spins(J, mu, H)

    if simulation == "mag_equilibrium":
        mag_equilibrium(J, mu, H)

    if simulation == "mag_plot":
        mag_plot(J, mu, H)

    if simulation == "energy_plot":
        energy_plot(J, mu, H)

    if simulation == "heat_capacity":
        heat_capacity(J, mu, H)

    if simulation == "susceptibility_plot":
        susceptibility_plot(J, mu, H)

    if simulation == "autocorrelation1":
        autocorrelation1(J, mu, H)

    if simulation == "autocorrelation2":
        autocorrelation2(J, mu, H)

    if simulation == "autocorrelation3":
        autocorrelation3(J, mu, H)

    if simulation == "finite_scaling":
        finite_scaling()

    if simulation == "external_field_mag":
        external_field_mag(J, mu)

    if simulation == "hysteresis":
        hysteresis(J, mu)

    end = time.time()
    print("Program run time =", end - start)

    plt.show()


simulate("spin_plot")
# simulate("mag_equilibrium")
# simulate("mag_plot")
# simulate("energy_plot")
# simulate("heat_capacity")
# simulate("susceptibility_plot")
# simulate("autocorrelation1")
# simulate("autocorrelation2")
# simulate("autocorrelation3")
# simulate("finite_scaling")
# simulate("external_field_mag")
# simulate("hysteresis")
