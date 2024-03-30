import numpy as np
import pyvista as pv
import h5py

grid = pv.ImageData()
#visualization using pyvistas ImageData for volume rendering
def volumeRendering(plotter, brain_scan, z_pos):
    if not np.all(brain_scan == 0): #to take away top and bottom rectangles (not plotting all zero)
        #initialize imageData
        
        grid.dimensions = brain_scan.shape[:3]  # Set dimensions based on the shape of the 3D volume
        grid.spacing = (1, 1, 1)
        grid.origin = (10, 0, z_pos * grid.spacing[2])  #adjusts z-axis to form 3d visualization
        grid.point_data['Segmentation'] = brain_scan.flatten(order='F')  #flatten 

        #adds volume to plotter
        plotter.add_volume(grid, scalars='Segmentation', shade=True,)

        contour = grid.contour()  # Adjust the isosurface value based on the segmentation

        # Add the contour mesh to the plotter
        plotter.add_mesh(contour, color='red')


plotter = pv.Plotter()
 
slice = 0 
volume = 1 #edit for different volumes
z_pos = 0
while slice < 100:
    with h5py.File(f'archive/BraTS2020_training_data/content/data/volume_{volume}_slice_{slice}.h5', 'r') as img:
        brain_scan = np.array(img['image'])
        print(f'archive/BraTS2020_training_data/content/data/volume_{volume}_slice_{slice}.h5')

        volumeRendering(plotter, brain_scan, z_pos)

        slice += 1
        z_pos += 1





#view from axial plane
plotter.camera_position = (0, 0, 20)
plotter.camera.roll = 180.0

#plotter settings
plotter.show_axes()
# plotter.set_background('grey')

#show plotter
plotter.show()

