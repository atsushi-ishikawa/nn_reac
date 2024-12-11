---
theme: unicorn
---

## Calculating reaction energy
#### Using Atomic Simulation Environment (ASE) library
```python
from ase import Atoms
from ase.build import add_adsorbate
from ase.calculators.vasp import Vasp

# define adsorbate
adsorbate = Atoms(symbols="N2", positions=[[0,0,0], [0,0,1.2]])

# define calculator
calc = Vasp(prec="normal", encut=400, xc="pbe")

# calculate reaction energy
deltaE = np.array([])
for side in ["reactant", "product"]:
    add_adsorbate(atoms, adsorbate, height=1.5)
    energy = atoms.get_potential_energy()
    energies[side] = energy

dE = energies["product"] - energies["reactant"]
deltaE = np.append(deltaE, dE)
print("reaction energy = %8.4f" % dE)

# save to external json file (e.g. "reaction_energy.json")
```