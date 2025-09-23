import json

def generate_impermeable_json(output_path='impermeable.json'):
    # Définition des champs
    num_vars = 14
    num_adim = 5

    json_data = {
        "physicals": {
            "Variables": [
                "rho", "u", "k", "mu", "L", "dx", "dt", "delta_p", "rho_out",
                "p_out", "p_in", "rho_in", "u_Darcy", "phi"
            ],
            "Physical": ["None"] * num_vars,
            "Reference": ["None"] * num_vars,
            "No dimension": ["None"] * num_vars
        },
        "adimensional_numbers": {
            "Number": ["Re", "Re_k", "Kn", "Da", "Ma"],
            "With u": ["None"] * num_adim,
            "With u_Darcy": ["None"] * num_adim
        },
        "LB_WBS_parameters": {
            "LB-WBS model": ["O(eps3)", "O(eps2)", "O(eps1)", "Selected"],
            "s_v": ["None", "None", "None", "2"],
            "alpha": ["None", "None", "None", "0"],
            "s_j": ["None", "None", "None", "2"]
        }
    }

    # Écriture du fichier JSON
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"✅ Fichier '{output_path}' généré avec succès.")

# Exemple d'appel
if __name__ == "__main__":
    generate_impermeable_json()
