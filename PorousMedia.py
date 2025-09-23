import numpy as np
import matplotlib.pyplot as plt
import porespy as ps
# Dimensions de la matrice
from skimage.transform import resize


def generate_layers(height=200, width=200, num_layers=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    array = np.zeros((height, width), dtype=int)

    if num_layers is None:
        num_layers = np.random.randint(2, 7)  # entre 2 et 6 couches

    for layer in range(0, num_layers ):
        base_start = np.random.randint(0, width // 2)
        base_end = base_start + np.random.randint(width // 10, width // 4)
        amp_sin = np.random.uniform(3, 10)
        amp_cos = np.random.uniform(3, 10)
        freq_sin = np.random.uniform(5, 20)
        freq_cos = np.random.uniform(5, 20)

        for i in range(height):
            start = int(base_start + amp_sin * np.sin(i / freq_sin))
            end = int(base_end + amp_cos * np.cos(i / freq_cos))
            start = max(0, min(width - 1, start))
            end = max(start + 1, min(width, end))
            array[i, start:end] = layer

    return array
def generate_circular(height=200, width=200, num_disks=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    array = np.zeros((height, width), dtype=int)

    if num_disks is None:
        num_disks = np.random.randint(3, 8)  # entre 3 et 7 disques

    for label in range(0, num_disks ):
        # Centre du disque
        center_x = np.random.randint(width//8, width//8*7)
        center_y = np.random.randint(width//8, width//8*7)

        # Rayon du disque
        radius = np.random.randint(6, min(height, width)//8)

        # Grille de coordonnées
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

        # Masque circulaire
        mask = dist_from_center <= radius

        # Remplissage avec la valeur du disque (label)
        array[mask] = label

    return array
def generate_vein_network_to_bottom_binary(size=512, n_veins=40, thickness=6, seed=None):
    image = np.ones((size, size))  # Fond blanc (1)
    if seed is not None:
        np.random.seed(seed)
    for _ in range(n_veins):
        x, y = np.random.randint(0, size), 0
        angle = np.pi / 2 + (np.random.rand() - 0.5)

        while y < size - thickness:
            dx = int(np.cos(angle) * 2)
            dy = int(np.sin(angle) * 2)
            x += dx + np.random.randint(-1, 2)
            y += dy + np.random.randint(0, 2)
            if not (0 <= x < size and 0 <= y < size):
                break
            # Dessine un disque autour de (x, y) avec valeur 0 (noir)
            for i in range(-thickness, thickness + 1):
                for j in range(-thickness, thickness + 1):
                    if 0 <= x+i < size and 0 <= y+j < size:
                        if i**2 + j**2 <= thickness**2:
                            image[y+j, x+i] = 0

            if np.random.rand() < 0.05:
                angle += (np.random.rand() - 0.5)
        # image[0:3*size//4,0:size//4]=1
        # image[0:3*size//4,3*size//4:]=1
    return image

def generate_square(size=100):
    square = np.zeros(shape=(size,size))
    square[3*size//8:5*size//8,3*size//8:5*size//8]=1
    return square
def generate_centered_circle(height=200, width=200):
    array = np.zeros((height, width), dtype=int)

    # Centre du cercle
    center_x = width // 2
    center_y = height // 2

    # Rayon = taille minimale / 4
    radius = min(height, width) // 6

    # Grille de coordonnées
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

    # Masque circulaire
    mask = dist_from_center <= radius

    # Remplissage avec la valeur 1
    array[mask] = 1

    return array


    return image
def generate_blobs(NX, NY, number_of_medium=4, porosity_list=[0.1, 0.24, 0.17, 0.4], blobiness_list=[1, 1, 2, 1], superposed_layers=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if superposed_layers== False:
        # Calcul de la hauteur de chaque couche
        layer_height = NY // number_of_medium
        im_total = np.zeros((NX, NY), dtype=int)  # Initialisation de l'image globale
        for i, (phi, blobiness) in enumerate(zip(porosity_list, blobiness_list)):
            # Génération de la couche binaire
            im = ps.generators.blobs(shape=[NX, layer_height], porosity=phi, blobiness=blobiness)
            im = im.astype(int) * (i + 1)  # Marquer cette couche avec une valeur différente

            # Calcul des indices de début et fin de cette couche dans l'image finale
            y_start = i * layer_height
            y_end = y_start + layer_height

            if y_end > NY:
                y_end = NY
                im = im[:, :NY - y_start]  # Ajuster la hauteur si nécessaire

            im_total[:, y_start:y_end] = im
        return im_total.T
    else :
        im_total = np.zeros((NX, NY), dtype=int)  # L'image finale commence vide

        for i, (phi, blobiness) in enumerate(zip(porosity_list, blobiness_list)):
            # Générer une nouvelle couche de blobs (binaire)
            im_layer = ps.generators.blobs(shape=[NX, NY], porosity=phi, blobiness=blobiness)
            im_layer = im_layer.astype(bool)

            # Ajouter cette couche au-dessus des précédentes :
            # Remplir les zones où la nouvelle couche est True
            im_total[im_layer] = i + 1  # i+1 car 0 = fond

        return im_total




def creer_masque_cercles(NX, cercles_par_ligne, facteur_rayon=0.2):
    """
    Crée un masque 2D avec des 1 à l'intérieur des cercles et des 0 à l'extérieur.

    Paramètres :
    - NX : nombre de points de discrétisation (NX x NX)
    - cercles_par_ligne : nombre de cercles par ligne et colonne
    - facteur_rayon : taille relative du rayon par rapport à l'espacement (valeur typique 0.4)

    Retour :
    - mask : array 2D de taille (NX, NX) avec 1 dans les cercles et 0 ailleurs
    """
    # Domaine
    Lx, Ly = 1.0, 1.0

    # Maillage
    x = np.linspace(0, Lx, NX)
    y = np.linspace(0, Ly, NX)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Espacement et rayon
    espace_x = Lx / cercles_par_ligne
    espace_y = Ly / cercles_par_ligne
    rayon = min(espace_x, espace_y) * facteur_rayon

    # Centres des cercles
    centres_x = np.linspace(espace_x / 2, Lx - espace_x / 2, cercles_par_ligne)
    centres_y = np.linspace(espace_y / 2, Ly - espace_y / 2, cercles_par_ligne)

    # Masque initialisé à zéro
    mask = np.zeros((NX, NX), dtype=int)

    # Remplissage du masque
    for cx in centres_x:
        for cy in centres_y:
            distance = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            mask[distance <= rayon] = 1

    return mask



