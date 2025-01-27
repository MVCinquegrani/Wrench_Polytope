import mujoco
import numpy as np
import xml.etree.ElementTree as ET
import pinocchio as pin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch

import polytope_geom as ply

class Polytope: 
    
    def display(self,color):

        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['lightcoral', 'pink', 'violet', 'plum', 'darkred', 'plum', 'purple', 'darkviolet']
        legend_elements = []
        legeend_labels = []
        p_min = np.zeros(3)
        p_max = np.zeros(3)

        verts = np.array(self.vertices)
        # p_min = np.minimum(p_min, np.min(verts,axis = 0))
        # p_max = np.maximum(p_max, np.max(verts,axis = 0))
        p_min = np.min(verts,axis = 0)
        p_max = np.max(verts,axis = 0)



        # polytope
        grouped_vertices = [[self.vertices[idx] for idx in face] for face in self.faces]
        ax.add_collection3d(Poly3DCollection(grouped_vertices, facecolors=color, linewidths=1.0, edgecolors=(1,0,0,0.1), alpha=.25))     
        plot_patch = Patch(color=color) 
        legend_elements.append((plot_patch))
        legeend_labels.append((" force geometry"))

        ax.set_xlim([p_min[0],p_max[0]])
        ax.set_ylim([p_min[1],p_max[1]])
        ax.set_zlim3d([p_min[2],p_max[2]])

        # Set labels in LaTeX
        ax.set_xlabel(r'$ x [m] $')
        ax.set_ylabel(r'$ y [m] $')
        ax.set_zlabel(r'$ z [m] $')


        plt.legend(legend_elements, legeend_labels)
        plt.show()        
