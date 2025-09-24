import json
import matplotlib.pyplot as plt
import pandas as pd
import PorousMedia
from matplotlib.colors import ListedColormap, BoundaryNorm

def all_data(path_json):
    with open(path_json, 'r') as f:
        data = json.load(f)

    # === Nombres adimensionnels ===
    adim_raw = data["adimensional_numbers"]
    if "With u" in adim_raw and "With u_Darcy" in adim_raw:
        adim_dict = {
            name: {
                "With u": val_u,
                "With u_Darcy": val_darcy
            }
            for name, val_u, val_darcy in zip(adim_raw["Number"], adim_raw["With u"], adim_raw["With u_Darcy"])
        }
    else:
        col_name = [key for key in adim_raw.keys() if key != "Number"][0]
        adim_dict = {
            name: value
            for name, value in zip(adim_raw["Number"], adim_raw[col_name])
        }

    # === Paramètres LB-WBS ===
    LB_raw = data["LB_WBS_parameters"]
    LB_dict = {
        model: {
            "s_v": sv,
            "alpha": alpha,
            "s_j": sj
        }
        for model, sv, alpha, sj in zip(LB_raw["LB-WBS model"], LB_raw["s_v"], LB_raw["alpha"], LB_raw["s_j"])
    }

    # === Paramètres physiques, références et adimensionnés ===
    phys_raw = data["physicals"]
    variables = phys_raw["Variables"]

    phys_dict = {var: val for var, val in zip(variables, phys_raw["Physical"])}
    ref_dict = {var: val for var, val in zip(variables, phys_raw["Reference"])}
    adi_dict = {var: val for var, val in zip(variables, phys_raw["No dimension"])}

    return adim_dict, LB_dict, phys_dict, ref_dict, adi_dict
def multi_model(paths_json, view_table=False):
    """Extract all data in the list of JSON path
            pathsJSON : liste of paths [exemple1.json, exemple2.json]
            option view-table : if True, allows to visualize into the terminal the value of the data """
    resultats = []

    for path in paths_json:
        try:
            adim, LB, phys, ref, adi = all_data(path)

            # ✅ Si le fichier contient 'impermeable', on force les valeurs
            if 'impermeable' in path.lower():
                LB.setdefault("Selected", {})
                LB["Selected"]["alpha"] = 0
                LB["Selected"]["s_v"] = 2
                LB["Selected"]["s_j"] = 2

            fichier_data = {
                "fichier": path,
                "adimensionnels": adim,
                "LB_parameters": LB,
                "physical": phys,
                "reference": ref,
                "adimensionne": adi
            }
            resultats.append(fichier_data)

            if view_table:
                print(f"\n📁 Fichier : {path}")

                print("📐 Nombres adimensionnels :")
                print(pd.DataFrame(adim).T)

                print("\n⚙️ Paramètres LB-WBS :")
                print(pd.DataFrame(LB).T)

                print("\n🔬 Paramètres physiques :")
                print(pd.DataFrame(phys.items(), columns=["Variable", "Valeur"]))

                # print("\n📏 Paramètres de référence :")
                # print(pd.DataFrame(ref.items(), columns=["Variable", "Valeur"]))

                print("\n📐 Paramètres adimensionnés :")
                print(pd.DataFrame(adi.items(), columns=["Variable", "Valeur"]))

                print("\n" + "=" * 70)

        except Exception as e:
            print(f"❌ Erreur lors du traitement de {path} : {e}")

    return resultats







