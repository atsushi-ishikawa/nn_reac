import os
import numpy as np
import pandas as pd
import argparse
import json
from tools import load_ase_json
from ase.db import connect

parser = argparse.ArgumentParser()
parser.add_argument("--surf_json", default="surf.json", help="json for surfaces")
parser.add_argument("--reac_json", default="reaction_energy.json", help="json for reaction energies")
parser.add_argument("--target", default="score", help="target key name")

args = parser.parse_args()
surf_json = args.surf_json
reac_json = args.reac_json
target = args.target

print("deleting unfinished result from {}".format(surf_json))

df_surf = load_ase_json(surf_json)

if os.path.exists(reac_json):
    df_reac = pd.read_json(reac_json)
else:
    df_reac = pd.DataFrame(columns=["unique_id"])

df_surf = df_surf.set_index("unique_id")
df_reac = df_reac.set_index("unique_id")
df = pd.concat([df_surf, df_reac], axis=1)

# delete null
null_list = df[df[target].isnull()].index.values

# delete errournous value
thre = -3.0
bad_list = df[df[target] < thre].index.values

del_list = np.concatenate([null_list, bad_list])

db = connect(surf_json)
for i in del_list:
    id = db.get(unique_id=i).id
    db.delete([id])
