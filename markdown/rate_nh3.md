---
theme: unicorn
---

## Calculating reaction rate from adsorption energies
#### Setup
```python
import pandas as pd

# parameters
T = 700  # K
ptot = 100.0  # bar
kJtoeV = 1/98.415
kB = 8.617e-5  # eV/K
RT = 8.314*1.0e-3*T*kJtoeV  # J/K -> kJ/K * K -> eV

# load reaction energy into Pandas dataframe
df_reac  = pd.read_json(reac_json)
```

---

## Loop over reaction energies (within loop)
#### Gibbs energy
```python
unique_id = df_reac.iloc[id].name
deltaE  = df_reac.iloc[id].reaction_energy
deltaE  = np.array(deltaE)

deltaH  = deltaE + deltaZPE + deltaTherm

deltaG  = deltaH - T*deltaS
deltaG += RTlnP

K = np.exp(-deltaG/(kB*T))

# index of rate-determining step
rds = 0

# activation energy
# Bronsted-Evans-Polanyi --- universal a and b for stepped surface (Norskov 2002)
alpha = 0.87
beta  = 1.34
tmp = alpha*deltaE + beta
Ea  = tmp[rds]
```

---

## Rate constant 
```python
A  = 1.2e6 / np.sqrt(T)  # Dahl J.Catal., converted from bar^-1 to Pa^-1
k  = A*np.exp(-Ea/RT)

# coverage
tmp = 1 + np.sqrt(K[1]*p[gas["H2"]]) \
        + p[gas["NH3"]]/(np.sqrt(K[1]*p[gas["H2"]])*K[4]*K[5]) \
        + p[gas["NH3"]]/(K[1]*p[gas["H2"]]*K[3]*K[4]*K[5]) \
        + p[gas["NH3"]]/(K[1]**(3/2)*p[gas["H2"]]**(3/2)*K[2]*K[3]*K[4]*K[5]) \
        + p[gas["NH3"]]/K[5]

theta[ads["vac"]] = 1/tmp
theta[ads["H"]]   = np.sqrt(K[1]*p[gas["H2"]])*theta[ads["vac"]]
theta[ads["NH3"]] = (p[gas["NH3"]]/K[5])*theta[ads["vac"]]
theta[ads["NH2"]] = (p[gas["NH3"]]/(np.sqrt(K[1]*p[gas["H2"]])*K[4]*K[5]))*theta[ads["vac"]]
theta[ads["NH"]]  = (p[gas["NH3"]]/(K[1]*p[gas["H2"]]*K[3]*K[4]*K[5]))*theta[ads["vac"]]
theta[ads["N"]]   = (p[gas["NH3"]]/(K[1]**(3/2)*p[gas["H2"]]**(3/2)*K[2]*K[3]*K[4]*K[5]))*theta[ads["vac"]]

Keq     = K[0]*(K[1]**3)*K[2]**2*K[3]**2*K[4]**2*K[5]**2
gamma   = (1/Keq)*(p[gas["NH3"]]**2/(p[gas["N2"]]*p[gas["H2"]]**3))
rate    = k*p[gas["N2"]]*theta[ads["vac"]]**2*(1-gamma)
```