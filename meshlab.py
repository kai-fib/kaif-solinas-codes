#https://pymeshlab.readthedocs.io/en/latest/filter_list.html

import pymeshlab
ms = pymeshlab.MeshSet()
path ='C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/'
ms.load_new_mesh(path + 'texturedMesh.obj')
ms.generate_convex_hull()
ms.save_current_mesh(path + 'vishal.ply')
ms.create_noisy_isosurface(resolution=128)
