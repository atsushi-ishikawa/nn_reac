import numpy as np
from ase import Atoms
from ase.visualize import view
from ase.build import surface
from ase.io import read, write

def sort_atoms_by(atoms, xyz="x"):
    # keep information for original Atoms
    tags = atoms.get_tags()
    pbc  = atoms.get_pbc()
    cell = atoms.get_cell()
    dtype = [("idx", int), (xyz, float)]

    newatoms = Atoms()
    symbols = list(set(atoms.get_chemical_symbols()))
    for symbol in symbols:
        subatoms = Atoms(list(filter(lambda x: x.symbol == symbol, atoms)))
        atomlist = np.array([], dtype=dtype)
        for idx, atom in enumerate(subatoms):
            if xyz == "x":
                tmp = np.array([(idx, atom.x)], dtype=dtype)
            elif xyz == "y":
                tmp = np.array([(idx, atom.y)], dtype=dtype)
            else:
                tmp = np.array([(idx, atom.z)], dtype=dtype)

            atomlist = np.append(atomlist, tmp)

        atomlist = np.sort(atomlist, order=xyz)

        for i in atomlist:
            idx = i[0]
            newatoms.append(subatoms[idx])

    # restore
    newatoms.set_tags(tags)
    newatoms.set_pbc(pbc)
    newatoms.set_cell(cell)

    return newatoms

def set_tags_by_z(atoms):
    import pandas as pd

    pbc  = atoms.get_pbc()
    cell = atoms.get_cell()

    newatoms = Atoms()
    symbols = list(set(atoms.get_chemical_symbols()))
    symbols = sorted(symbols)

    for symbol in symbols:
        subatoms = Atoms(list(filter(lambda x: x.symbol == symbol, atoms)))
        pos  = subatoms.positions
        zpos = np.round(pos[:, 2], decimals=1)
        bins = list(set(zpos))
        bins = np.sort(bins)
        bins = np.array(bins) + 1.0e-2
        bins = np.insert(bins, 0, 0)

        labels = []
        for i in range(len(bins)-1):
            labels.append(i)

        tags = pd.cut(zpos, bins=bins, labels=labels).to_list()

        subatoms.set_tags(tags)
        newatoms += subatoms

    # restore
    newatoms.set_pbc(pbc)
    newatoms.set_cell(cell)

    return newatoms

def remove_layer(atoms=None, symbol=None, higher=1):
    import pandas as pd
    from ase.constraints import FixAtoms

    pbc  = atoms.get_pbc()
    cell = atoms.get_cell()

    atoms_copy = atoms.copy()

    # sort
    atoms_copy = sort_atoms_by(atoms_copy, xyz="z")

    # set tags
    atoms_copy = set_tags_by_z(atoms_copy)

    newatoms = Atoms()

    tags = atoms_copy.get_tags()
    maxtag = max(list(tags))

    for i, atom in enumerate(atoms_copy):
        if atom.tag >= maxtag - higher + 1 and atom.symbol == symbol:
            # remove this atom
            pass
        else:
            newatoms += atom

    newatoms.set_pbc(pbc)
    newatoms.set_cell(cell)

    return newatoms

bulk = read("RuO2.cif")
vacuum = 10.0

surf = surface(bulk, indices=[1,1,0], layers=4, vacuum=vacuum, periodic=True)
surf = surf*[1, 2, 1]
surf.translate([0, 0, -vacuum+0.1])
surf.wrap(eps=1.0e-2)

surf = remove_layer(atoms=surf, symbol="O", higher=1)
write("POSCAR", surf)

