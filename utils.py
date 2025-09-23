import json
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt

import extractJSON


def display_json_structure_help(data):
    print("📁 data[\"physicals\"]")
    try:
        variables = data["physicals"]["Variables"]
        print("   ├── Available variables:")
        for var in variables:
            print(f"   │   🔹 {var}")
        print("   └── To access values:")
        print("       • data['physicals']['Physical'][i]")
        print("       • data['physicals']['Reference'][i]")
        print("       • data['physicals']['No dimension'][i]")
        print("       ↪️  where i is the index of the variable in 'Variables'")
    except:
        print("   ⚠️ Failed to access 'physicals'")

    print("\n📁 data[\"adimensional_numbers\"]")
    try:
        names = data["adimensional_numbers"]["Number"]
        print("   ├── Available dimensionless numbers:")
        for n in set(names):
            print(f"   │   🔹 {n}")
        keys = [k for k in data["adimensional_numbers"].keys() if k != "Number"]
        print("   └── To access values:")
        for key in keys:
            print(f"       • data['adimensional_numbers']['{key}'][i]")
        print("       ↪️  where i is the index of the number in 'Number'")
    except:
        print("   ⚠️ Failed to access 'adimensional_numbers'")

    print("\n📁 data[\"LB_WBS_parameters\"]")
    try:
        LB_data = data["LB_WBS_parameters"]
        if isinstance(LB_data, dict):
            models = list(LB_data.keys())
        else:
            models = LB_data["LB-WBS model"]
        print("   ├── Available LB models:")
        for m in set(models):
            print(f"   │   🔹 {m}")
        print("   └── To access parameters for a model:")
        print("       • data['LB_WBS_parameters']['O(eps3)']['s_v']")
        print("       • data['LB_WBS_parameters']['O(eps1)']['alpha']")
        print("       • data['LB_WBS_parameters']['Selected']['s_j']")
    except:
        print("   ⚠️ Failed to access 'LB_WBS_parameters'")

# Charger un fichier JSON
# with open("milieu_test1.json", "r") as f:
#     data = json.load(f)
#
# # Appeler ta fonction
# display_json_structure_help(data)

# def relaxation_matric_according_porous(porous, )

def show_porous_media_and_LB_paramters(porous, lb_parameters, gamma_value):

    colors = [
        "lightblue", "lightgreen", "orange", "plum", "khaki", "lightsalmon",
        "mediumaquamarine", "thistle", "powderblue", "peachpuff", "palegreen",
        "wheat", "skyblue", "navajowhite", "mistyrose"
    ]

    number_of_layers = len(lb_parameters)
    cmap = ListedColormap(colors[:number_of_layers])
    bounds = np.arange(-0.5, number_of_layers + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    # Affichage de l'image
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(porous, cmap=cmap, norm=norm)  # transpose for visual convention if needed

    # Colorbar sans ticks numériques
    cbar = plt.colorbar(im, ax=ax, boundaries=bounds)
    cbar.set_ticks([])
    # Ajout des annotations
    for i, media, gamma in zip(range(len(lb_parameters)+1),lb_parameters, gamma_value):
        selected = media["LB_parameters"].get("Selected", {})
        sv = selected.get("s_v", "-")
        alpha = selected.get("alpha", "-")
        sj = selected.get("s_j", "-")
        name_file = media["fichier"]

        label = fr"File : {name_file}" + "\n" + fr"$s_v = {sv}$" + "\n" + fr"$\alpha = {alpha}$" + "\n" + fr"$s_j = {sj}$" + "\n" + fr"$\gamma={gamma_to_tex(gamma)}$"
        cbar.ax.text(1.2, i, label, va='center', fontsize=8, ha='left',
                     transform=cbar.ax.transData, linespacing=1.5)

    plt.tight_layout()
    plt.show()

def gamma_to_tex(gamma):
    if gamma==0:
        gm='0'
    elif gamma==1:
        gm='1'
    elif gamma==2:
        gm="1-\epsilon"
    return gm


def equilibrium(k,cx, cy , u,v, rho, cs, w,type='linear'):
    if type =='linear':
        feq = w[k] * (rho + 1 / cs ** 2 * (u * cx[k]) + 1 /cs ** 2 * (v * cy[k]))
    elif type=='non linear':
        u2 = u ** 2 + v ** 2
        cu = cx[k] *u + cy[k] * v
        feq = w[k]  * (rho + (cu / cs**2) +( (0.5 * (cu ** 2) /cs**2) - (0.5 * u2 /cs**2)))
    return feq


import numpy as np
from skimage.transform import resize


def resize_array_to_newshape(array, new_size=500):
    """
    Redimensionne un array 2D de n'importe quelle taille vers (500, 500)
    tout en conservant l'information géométrique.

    Parameters:
        array (np.ndarray): Array 2D d'entrée (ex: 200x200)

    Returns:
        np.ndarray: Array 500x500 redimensionné
    """
    if array.ndim != 2:
        raise ValueError("L'array doit être 2D.")

    resized = resize(array, (new_size, new_size), order=1, preserve_range=True, anti_aliasing=True)
    return resized
