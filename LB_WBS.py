import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
# from vtk.util import numpy_support

NPOP = 9  # number of velocities
w = np.array([1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36,4/9])  # weights
cx = np.array([1, 0, -1, 0, 1, -1, -1, 1,0])  # velocities, x components
cy = np.array([0, 1, 0, -1, 1, 1, -1, -1,0])  # velocities, y components


def opposite(k):
    if k == 0:
        return 2
    elif k == 1:
        return 3
    elif k == 2:
        return 0
    elif k == 3:
        return 1
    elif k == 4:
        return 6
    elif k == 5:
        return 7
    elif k == 6:
        return 4
    elif k == 7:
        return 5
    elif k == 8:
        return 8

class PressureDependanceModel():
    def __init__(self, THETA_FIELD,GAMMA_FIELD, XI, NX,NY, RHO_IN, RHO_OUT, CS):
        self.rho = None
        self.u = None
        self.v = None

        self.f = None
        self.feq = None
        self.fprop = None
        self.fprecoll = None

        self.theta = THETA_FIELD

        self.lattice_sound_speed = CS

        self.NX =NX
        self.NY = NY


        self.rho_in = RHO_IN
        self.rho_out =RHO_OUT

        self.xi = XI
        self.gamma = GAMMA_FIELD

        self.right_bc = 'No slip'
        self.left_bc = 'No slip'

        self.streaming_model = 'WBS'


        self.maximum_of_iteration = 20000

    def initilisation(self, RHO0, UX0, UY0):
        self.feq = np.zeros((self.NX, self.NY, NPOP))
        self.rho = np.zeros(shape=(self.NX, self.NY))+ RHO0
        for k in range(NPOP):
            self.feq[:, :, k] = w[k] * self.rho  # assuming density equal one and zero velocity initial state
        self.f = self.feq.copy()
        # self.theta_cst = THETA
        self.fprop = self.feq.copy()
        self.u = np.zeros_like(self.rho)+UX0
        self.v = np.zeros_like(self.rho)+UY0




    def macro(self):
        # density
        self.rho = np.sum(self.fprop, axis=2)

        # momentum components
        self.u = np.sum(self.fprop[:, :, [0, 4, 7]], axis=2) - np.sum(self.fprop[:, :, [2, 5, 6]], axis=2)
        self.v = np.sum(self.fprop[:, :, [1, 4, 5]], axis=2) - np.sum(self.fprop[:, :, [3, 6, 7]], axis=2)

    def equilibrium(self):
        u2 = self.u ** 2 + self.v ** 2
        for k in range(NPOP):
            cu = cx[k] * self.u + cy[k] * self.v
            self.feq[:, :, k] = w[k]  * (self.rho + (cu / self.lattice_sound_speed**2) +self.gamma*( (0.5 * (cu ** 2) / self.lattice_sound_speed**2) - (0.5 * u2 / self.lattice_sound_speed**2)))

    def pre_collision(self):
        self.fprecoll = self.fprop.copy()

    def collision(self):
        #print(self.f[mask_porous].shape)
        self.f = self.fprop - np.einsum('nmij,nmi->nmi', self.xi, self.fprop - self.feq)

    def pressure_topAndBottom_BoundaryCondition(self):
        a = -2
        b = 1
        for k in range(NPOP):
            self.f[0, :, k] = w[k] * ( self.rho_in + 1 / self.lattice_sound_speed ** 2 * (cx[k] * self.u[a, :]) + 1 / self.lattice_sound_speed **2* (cy[k] * self.v[a, :])) + (self.f[a, :, k] - self.feq[a, :, k])
            self.f[-1, :, k] = w[k] * (self.rho_out + 1 / self.lattice_sound_speed ** 2 * (cx[k] * self.u[b, :]) + 1 / self.lattice_sound_speed **2 *(cy[k] * self.v[b, :])) + (self.f[b, :, k] - self.feq[b, :, k])
    def streaming(self):
        if self.streaming_model == 'WBS':
            for k in range(NPOP):
                self.fprop[:, :, k] = np.roll(self.f[:, :, k], (cx[k], cy[k]), axis=(0, 1)) + np.roll(self.theta,(cx[k], cy[k]),axis=(0, 1)) * (np.roll(self.fprecoll[:, :, opposite(k)], (cx[k], cy[k]),axis=(0, 1)) - np.roll(self.f[:, :, k], (cx[k], cy[k]), axis=(0, 1)))
        elif self.streaming_model == 'ZM':
            for k in range(NPOP):
                self.fprop[:, :, k] = np.roll(self.f[:, :, k], (cx[k], cy[k]), axis=(0, 1)) + np.roll(self.theta,(cx[k], cy[k]),axis=(0, 1)) * (np.roll(self.f[:, :, opposite(k)], (cx[k], cy[k]),axis=(0, 1)) - np.roll(self.f[:, :, k], (cx[k], cy[k]), axis=(0, 1)))
        else :
            print("Choose a scheme : type_streaming= 'WBS' or 'YH' ")
    def right_BoundaryCondition(self):
        if self.right_bc == 'No slip':
            self.fprop[:, -1, 3] = self.f[:, -1, 1]
            self.fprop[:, -1, 6] = self.f[:, -1, 4]
            self.fprop[:, -1, 7] = self.f[:, -1, 5]
        elif self.right_bc== "Free slip":
            self.fprop[:, -1, 3] = self.f[:, -1, 1]
            self.fprop[:, -1, 6] = self.f[:, -1, 5]
            self.fprop[:, -1, 7] = self.f[:, -1, 4]

        else: print("Affect a boundary condition on the right boundary: model.right_bc = 'No slip' or 'Free slip'")

    def left_BoundaryCondition(self):
        # Bottom wall (rest)
        if self.left_bc =='No slip':
            self.fprop[:, 0, 1] = self.f[:, 0, 3]
            self.fprop[:, 0, 4] = self.f[:, 0, 6]
            self.fprop[:, 0, 5] = self.f[:, 0, 7]
        elif self.left_bc =='Free slip':
            # Bottom wall (rest)
            self.fprop[:, 0, 1] = self.f[:, 0, 3]
            self.fprop[:, 0, 4] = self.f[:, 0, 7]
            self.fprop[:, 0, 5] = self.f[:, 0, 6]
        else: print("Affect a boundary condition on the left boundary: model.left_bc = 'No slip' or 'Free slip'")

    def run(self):
        for i in tqdm.tqdm(range(self.maximum_of_iteration)):
            self.equilibrium()
            self.pre_collision()
            self.collision()
            self.pressure_topAndBottom_BoundaryCondition()
            self.streaming()
            self.right_BoundaryCondition()
            self.left_BoundaryCondition()
            self.macro()

            # Vérification NaN
            if np.isnan(self.u).any() or np.isnan(self.v).any():
                print(f"NaN détecté à l'étape {i} — arrêt de la simulation.")

        print(f'End of simulation after {self.maximum_of_iteration} iterations of the {self.streaming_model} scheme')


def relaxation_time_matrix2(sv, sj, nx):
    XI_all = np.zeros(shape=(nx, nx,9,9))
    S_V = sv
    S_RHO = S_V
    S_E = 1.1
    S_EPSILON = 1.2
    S_J = sj

    S_Q = 1.1

    RELAXATION_TIME_MATRIX = np.diag([S_RHO, S_E, S_EPSILON, S_J, S_Q, S_J, S_Q, S_V, S_V])
    MOMENTUM_MATRIX = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                                [4, -2, -2, -2, -2, 1, 1, 1, 1],
                                [0, 1, 0, -1, 0, 1, -1, -1, 1],
                                [0, -2, 0, 2, 0, 1, -1, -1, 1],
                                [0, 0, 1, 0, -1, 1, 1, -1, -1],
                                [0, 0, -2, 0, 2, 1, 1, -1, -1],
                                [0, 1, -1, 1, -1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, -1, 1, -1]
                                ])
    MOMENTUM_MATRIX = np.roll(MOMENTUM_MATRIX, -1, axis=1)
    # Construction de XI_layer
    XI = np.linalg.inv(MOMENTUM_MATRIX) @ RELAXATION_TIME_MATRIX @ MOMENTUM_MATRIX
    XI_all[:,:]= XI
    return XI_all
def relaxation_time_matrix(NX,NY,DX,number_of_layers, multi_porous, porous_media, gamma_value):
    THETA_FIELD = np.zeros(shape=(NX, NY))
    XI = np.zeros(shape=(NX, NY, 9, 9))
    GAMMA = np.zeros(shape=(NX,NY))
    for i, media, gm in zip(range(number_of_layers), multi_porous, gamma_value):
        S_V = float(media['LB_parameters'].get('Selected')['s_v'])
        S_RHO = S_V
        S_E = 1.1
        S_EPSILON = 1.2
        s_j_value = media['LB_parameters'].get('Selected', {}).get('s_j', '-')
        try:
            S_J = float(s_j_value)
        except ValueError:
            S_J = S_V  # fallback
        S_Q = 1.1

        RELAXATION_TIME_MATRIX = np.diag([S_RHO, S_E, S_EPSILON, S_J, S_Q, S_J, S_Q, S_V, S_V])
        MOMENTUM_MATRIX = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                                    [4, -2, -2, -2, -2, 1, 1, 1, 1],
                                    [0, 1, 0, -1, 0, 1, -1, -1, 1],
                                    [0, -2, 0, 2, 0, 1, -1, -1, 1],
                                    [0, 0, 1, 0, -1, 1, 1, -1, -1],
                                    [0, 0, -2, 0, 2, 1, 1, -1, -1],
                                    [0, 1, -1, 1, -1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, -1, 1, -1]
                                    ])
        MOMENTUM_MATRIX = np.roll(MOMENTUM_MATRIX, -1, axis=1)
        # Construction de XI_layer
        XI_layer = np.linalg.inv(MOMENTUM_MATRIX) @ RELAXATION_TIME_MATRIX @ MOMENTUM_MATRIX

        # Appliquer la même matrice à tous les points du milieu courant
        mask = (porous_media == i)
        indices = np.argwhere(mask)  # Liste des (i, j)

        for ix, iy in indices:
            XI[ix, iy, :, :] = XI_layer

        # Ajout dans le champ theta
        alpha = float(media['LB_parameters'].get('Selected')['alpha'])
        kn= media["adimensionnels"].get('Kn')["With u"]
        if kn == 'None':
            kn = 1#float(media["adimensionnels"].get('Kn')["With u"])
            theta = kn**alpha
        else :
            kn = float(kn)
            theta=kn**alpha
        if alpha > 3:
            THETA_FIELD[mask] = 0
        else:
            THETA_FIELD[mask] = theta**alpha

        if gm==1:
            GAMMA[mask]=1
        elif gm==2:
            GAMMA[mask]=1-theta

    return XI, THETA_FIELD, GAMMA

