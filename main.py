import matplotlib.pyplot as plt
import numpy as np
import LB_WBS as LB
import extractJSON
import PorousMedia
import utils


#An exemple to compute LB-WBS scheme in heterogeneous permeable media
NX=200
NY=200


# Adi
UX0=0
UY0=0

root_file = 'media_json/'
path_json = ['medium_test4.json',
             'medium_test3.json',
             'medium_test6.json',
             'medium_test6.json']

path_json = [root_file + file_json for file_json in path_json]
number_of_layers = len(path_json)


# Step 1 : generation of porous matrix
porous_media = PorousMedia.generate_layers(height=NX, width=NX, num_layers=number_of_layers, seed=33).T

range_value = np.arange(len(path_json))
multi_porous = extractJSON.multi_model(path_json, view_table=False) #JSON data extraction of path_json

value_gamma= [0]*number_of_layers #non linear term

utils.show_porous_media_and_LB_paramters(porous_media, multi_porous, gamma_value=value_gamma)

######## Pre-procesing step
## Parameters for LB-WBS model based on pressure dependance flow
RHO_OUT=float(multi_porous[0]['adimensionne']['rho_out'])
RHO0= RHO_OUT
RHO_IN=float(multi_porous[0]['adimensionne']['rho_in'])
LENGTH = float(multi_porous[0]['adimensionne']['L'])
DX=float(multi_porous[0]['adimensionne']['dx'])
DT = float(multi_porous[0]['adimensionne']['dt'])
CS= DX/DT/np.sqrt(3)


# Define the relxation time matrix according the porous media (sv,sj) and défine the theta_field.
XI, THETA_FIELD, GAMMA_FIELD= LB.relaxation_time_matrix(NX=NX,
                                                NY=NY,
                                                DX=DX,
                                                number_of_layers = number_of_layers,
                                                multi_porous=multi_porous,
                                                porous_media=porous_media,
                                                gamma_value=value_gamma)


# LB_WBS model
model = LB.PressureDependanceModel(THETA_FIELD=THETA_FIELD,
                                   GAMMA_FIELD=GAMMA_FIELD,
                                     XI= XI,
                                     NX=NX,
                                     NY=NY,
                                     RHO_IN=RHO_IN,
                                     RHO_OUT=RHO_OUT,
                                     CS=CS,
                                   )

# End pre-processing step

ite_max=2000 #maximum of iteration

model.initilisation(RHO0=RHO0, UX0=0, UY0=0) #initialization of the model

model.right_bc = 'No slip'
model.left_bc = 'No slip'

model.streaming_model ='WBS'
model.maximum_of_iteration = 2000

model.run()

magnitude = np.sqrt(model.u**2+model.v**2)
fig, ax = plt.subplots(1,1, figsize=(6,5))

im = ax.imshow(magnitude, extent=[0, LENGTH, 0, LENGTH],
               cmap='jet', aspect='equal')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Magnitude")
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.show()
