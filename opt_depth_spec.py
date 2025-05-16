import matplotlib.pyplot as plt
import numpy as np

S = 1.610E-23 # line strength in 1/cm
L = 40.3 #length in cm
# L = 1.6843 # cm
P = 1 # atm
r = 2.43 # cm
V = float(np.pi * r**2 * L) # cm^3 
print(f'volume = {V}')
k_erg = 1.380469E-16 # erg/K
k = k_erg * (9.869E-7) # atm/K
T = 293 # kelvin
n = P*V / (k * T)
chi = 1 #tells us the percentage of molecules in the cell that are CO

gamma = 0.068 #broadening coefficient in 1/(cm * atm)
nu = np.linspace(-5, 5,1000)
phi = [] # lineshape function
tau = [] # optical depth

for i in range(len(nu)):
    phi_val = float((1/np.pi) * ((gamma*P)/((gamma * P)**2 + nu[i]**2)))
    phi.append(phi_val)
    tau_val = chi * n * L * S * phi_val
    tau.append(tau_val)

plt.figure()
plt.plot(nu, tau)
plt.ylabel('optical depth (1/cm)')
plt.xlabel('nu (1/cm)')
plt.show()