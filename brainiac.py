import numpy as np
import pyvista as pv
from pyvista import themes
import h5py
import scipy.ndimage as sci
from scipy.spatial.transform import Rotation as R

# pv.set_plot_theme(themes.DarkTheme())
sagittal_angle = 0 #x
horizontal_angle = 20 #z
coronal_angle = 0 #y


grid = pv.ImageData()

#visualization using pyvistas ImageData for volume rendering
def volumeRendering(plotter, brain_scan, x_pos, y_pos,z_pos, opac, cmap):
    if not np.all(brain_scan == 0): #to take away top and bottom rectangles (not plotting all zero)

        grid.dimensions = brain_scan.shape[:3] 
        grid.spacing = (1, 1, 1)
        grid.origin = (x_pos, y_pos, z_pos) 
        grid.point_data['normal'] = brain_scan.flatten(order='F')  #flatten 
        #mesh layer
        contour = grid.contour()
        pl = plotter.add_mesh(contour,color='k', show_edges=True)

        #rotation
        pl = rotate_brain(pl, sagittal_angle, horizontal_angle, coronal_angle)

        pl.position = (0,0, 15)

        #adds volume to plotter
        plotter.add_volume(grid, scalars='normal', shade=False, opacity=opac ,cmap=cmap)
        

def rotate_brain(pl, sagittal_angle, horizontal_angle, coronal_angle):


    # Create a rotation object
    r = R.from_euler('xyz', [sagittal_angle, horizontal_angle, coronal_angle], degrees=True)

    #Apply rotation to the points
    pl.rotate_x(sagittal_angle)
    pl.rotate_y(coronal_angle)
    pl.rotate_z(horizontal_angle)
    
    return pl


def readData(plotter, x_pos, y_pos, z_pos, cmap, opacity):


    slice = 0 
    volume = 1 #edit for different volumes
    
    while slice < 154:
        with h5py.File(f'archive/BraTS2020_training_data/content/data/volume_{volume}_slice_{slice}.h5', 'r') as img:
            brain_scan = np.array(img['image'])
            print(f'archive/BraTS2020_training_data/content/data/volume_{volume}_slice_{slice}.h5')
            volumeRendering(plotter, brain_scan, x_pos, y_pos, z_pos, opacity, cmap)

            slice += 1
            z_pos += 1
        


x_pos = 10
y_pos = 0
z_pos = 0
cmap = "bone"
opacity = "sigmoid"

plotter = pv.Plotter()

for i in range(3):
    readData(plotter, x_pos, y_pos, z_pos, cmap, opacity)

#features
plotter.add_bounding_box( color='red', render_lines_as_tubes=True)
plotter.show_bounds(grid='front', location='outer')

#view from axial plane
plotter.camera_position = 'zx'
plotter.camera.roll = 180.0

#plotter settings
#plotter.show_axes()
# plotter.set_background('grey')


#show plotter
plotter.show()