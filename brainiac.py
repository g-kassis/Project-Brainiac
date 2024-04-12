import numpy as np
import pyvista as pv
from pyvista import themes
import h5py
import scipy.ndimage as sci
from scipy.spatial.transform import Rotation as R
from random import randrange

# pv.set_plot_theme(themes.DarkTheme())
sagittal_angle = randrange(20) #x
horizontal_angle = randrange(20) #z
coronal_angle = randrange(20) #y


grid = pv.ImageData()

#visualization using pyvistas ImageData for volume rendering
def volumeRendering(plotter, brain_scan, x_pos, y_pos,z_pos, opac, cmap,n_evo):

    if not np.all(brain_scan == 0): #to take away top and bottom rectangles (not plotting all zero)

        plotter.subplot(0,0)
        plotter.add_text("Unshiftted", font_size=30)

        grid.dimensions = brain_scan.shape[:3] 
        grid.spacing = (1, 1, 1)
        grid.origin = (x_pos, y_pos, z_pos)
        grid.point_data['normal'] = brain_scan.flatten(order='F')  #flatten 

        pls = evolution(plotter, cmap, n_evo) #min 2 (no evo)

        #rotation
        prev_sag, prev_hor, prev_cor = 0,0,0
        for pl in range(len(pls)):
            plotter.subplot(0,pl)

            #no rotation (og scan)
            if pl == 0:
                pass
        
            #full rotation (final)
            elif pl == len(pls)-1:
                rotate_brain(pls[pl], sagittal_angle, horizontal_angle, coronal_angle)

            #evolutions
            else:
                rotate_brain(pls[pl], prev_sag/n_evo, prev_hor/n_evo, prev_cor/n_evo)
                prev_sag, prev_hor, prev_cor = prev_sag/n_evo, prev_hor/n_evo, prev_cor/n_evo

        plotter.subplot(0,0)


        #plotter.add_volume(grid, scalars='normal', shade=False, opacity=opac ,cmap=cmap)

def evolution(plotter, cmap, n_evo):

    pls = []

    for evo in range(n_evo):

        #No Brain Shift
        if evo == 0:
            plotter.subplot(0,0)
            plotter.add_text("Unshiftted", font_size=30)
            contour = grid.contour()
            plMain = plotter.add_mesh(contour, style='wireframe', cmap=cmap, show_edges=True)
            pls.append(plMain)

        #Brain Shift (Fiinal)
        elif evo == n_evo-1:
            plotter.subplot(0,evo)
            plotter.add_text("Brain Shift", font_size=30)
            contour = grid.contour()
            pl = plotter.add_mesh(contour, style='wireframe', cmap=cmap, show_edges=True)
            pls.append(pl)

        else:
            plotter.subplot(0,evo)
            plotter.add_text("Evo" + str(evo), font_size=30)
            contour = grid.contour()
            pl = plotter.add_mesh(contour, style='wireframe', cmap=cmap, show_edges=True)
            pls.append(pl)

    return pls


    # #mesh layer #1 
        

    #     #mesh layer #2 
    #     plotter.subplot(0,1)
    #     plotter.add_text("Evo", font_size=30)
    #     contour = grid.contour()
    #     pl2 = plotter.add_mesh(contour, style='wireframe', cmap=cmap, show_edges=True)

    #     #main layer 
    #     plotter.subplot(0,0)
    #     plotter.add_text("Unshiftted", font_size=30)
    #     contour = grid.contour()
    #     plMain = plotter.add_mesh(contour, style='wireframe', cmap=cmap, show_edges=True)

    # return pl, pl2


def rotate_brain(pl, sagittal_angle, horizontal_angle, coronal_angle):

    #rot for eeach axis
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(np.radians(sagittal_angle)), np.sin(np.radians(sagittal_angle)), 0],
        [0, -np.sin(np.radians(sagittal_angle)), np.cos(np.radians(sagittal_angle)), 0],
        [0, 0, 0, 1],
    ])

    Ry = np.array([
        [np.cos(np.radians(coronal_angle)), 0, -np.sin(np.radians(coronal_angle)), 0],
        [0, 1, 0, 0],
        [np.sin(np.radians(coronal_angle)), 0, np.cos(np.radians(coronal_angle)), 0],
        [0, 0, 0, 1],
    ])

    Rz = np.array([
        [np.cos(np.radians(horizontal_angle)), -np.sin(np.radians(horizontal_angle)), 0, 0],
        [np.sin(np.radians(horizontal_angle)), np.cos(np.radians(horizontal_angle)), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    #combining rotation matricies
    rotation_matrix = Rz @ Ry @ Rx

    #apply rotatation
    pl.user_matrix = rotation_matrix

    return pl

def readData(plotter, split, volume,n_evo, x_pos, y_pos, z_pos, cmap, opacity):


    slice = 0 
    volume = 1
    if split:
        fSlice = split
    else:
        fSlice = 154 
    
    while slice < fSlice:
        with h5py.File(f'archive/BraTS2020_training_data/content/data/volume_{volume}_slice_{slice}.h5', 'r') as img:
            brain_scan = np.array(img['image'])
            print(f'archive/BraTS2020_training_data/content/data/volume_{volume}_slice_{slice}.h5')
            volumeRendering(plotter, brain_scan, x_pos, y_pos, z_pos, opacity, cmap,n_evo)

            slice += 1
            z_pos += 1
        
def callback(pos):
    print(f'{pos}')


def main():
    brainSplit = False
    brainSplit = input('Would you like to split the brain? (y/n): ')
    if brainSplit == 'y': 
        brainSplit = True 
        brainSplit = input('Where would you like to split the brain (1 - 154): ')
    else: brainSplit = False
    volume = input('Input Volume number between 1 and 369: ')
    n_evo = input('Number of evolutions (recommended: 2 or 3): ')
    num_evo = int(n_evo)
    plotter = pv.Plotter(shape=(1,int(num_evo)),window_size=[2560,1400])

    

    


    readData(plotter, int(brainSplit), volume,num_evo ,x_pos = -120, y_pos = -120, z_pos = 0, cmap = "coolwarm", opacity = "sigmoid")


    for plot in range(num_evo):
        plotter.subplot(0,plot)
        #plotter.add_bounding_box( color='red', render_lines_as_tubes=True)
        plotter.show_bounds(grid='front', location='outer', minor_ticks=True, bounds=[-240, 240, -240, 240, -240, 240],
                            show_xaxis=True, show_yaxis=True, show_zaxis=True, show_xlabels=True, show_ylabels=True, show_zlabels=True)

        #camera view
        plotter.camera_position = (0,0,20) # 'xz', 'xy', 'zx', 'zy' or set to (0,0,20) to reset
        plotter.camera.roll = 180.0
        #plotter.view_isometric() 

        #plotter settings
        #plotter.set_background('black', top='white')


    #show plotter
    plotter.track_click_position(callback=callback, side='left')
    plotter.add_camera_orientation_widget(animate=False)
    print('Angles: ', sagittal_angle, horizontal_angle, coronal_angle)
    plotter.show()
    


main()