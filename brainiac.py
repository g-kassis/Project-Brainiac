import numpy as np
import pyvista as pv
from pyvista import themes
import h5py

# pv.set_plot_theme(themes.DarkTheme())

grid = pv.ImageData()
#visualization using pyvistas ImageData for volume rendering
def volumeRendering(plotter, brain_scan, z_pos):
    if not np.all(brain_scan == 0): #to take away top and bottom rectangles (not plotting all zero)
        #initialize imageData
        
        grid.dimensions = brain_scan.shape[:3]  # Set dimensions based on the shape of the 3D volume
        grid.spacing = (100, 100, 100)
        grid.origin = (10, 0, z_pos * grid.spacing[2])  #adjusts z-axis to form 3d visualization
        grid.point_data['Segmentation'] = brain_scan.flatten(order='F')  #flatten 

        #adds volume to plotter
        plotter.add_volume(grid, scalars='Segmentation', shade=True,)


plotter = pv.Plotter()
plotter.add_legend_scale(corner_offset_factor=2.0, bottom_border_offset=30, top_border_offset=30, left_border_offset=30, right_border_offset=30, bottom_axis_visibility=True, top_axis_visibility=True, left_axis_visibility=True, right_axis_visibility=True, legend_visibility=True, xy_label_mode=False, render=True, color=None, font_size_factor=0.6, label_size_factor=1.0, label_format=None, number_minor_ticks=0, tick_length=5, minor_tick_length=3, show_ticks=True, tick_label_offset=2)
plotter.add_bounding_box(line_width=10, color='red')
 
slice = 0 
volume = 1 #edit for different volumes
z_pos = 0
while slice < 154:
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

