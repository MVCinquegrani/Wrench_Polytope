#!/usr/bin/python3
#   /**
#    * pinocchio state = [basePos(0-2) baseQuad(3-6) q(7-18)]
#    *
#    * basePos: Base position in Origin frame (3x1)
#    * baseQuad: (x,y,z,w) (4x1)
#    * q: Joint angles per leg [HAA, HFE, KFE] (3x1) [4x]
#    *
#    * pinocchio velocity = [v(0-2) w(3-5) qj(6-17)]
#    *
#    * v: Base linear velocity in Base Frame (3x1)
#    * w: Base angular velocity in Base Frame (3x1)
#    * qj: Joint velocities per leg [HAA, HFE, KFE] (3x1)
#    */


import mujoco
import numpy as np
import xml.etree.ElementTree as ET
import pinocchio as pin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
from scipy.optimize import linprog


# from pycapacity.visual import *
import polytope_geom as ply
import polytope_visual



class LimitForce:
    def __init__(self, EE = "", force_limit = np.zeros(3), magnitude = 0):
        self.foot_name = EE
        self.force = force_limit
        self.magnitude = magnitude

    def __iter__(self):
        return iter(self.force)



class ForcePolytopeBuilder:
    def __init__(self, force_limits = None, contact = None, polytope = None, foot_name=None):
        self.polytope = polytope
        self.contact = contact
        self.force_limits = force_limits if force_limits is not None else []
        self.poly_verts_contact = None
        self.foot_name = foot_name 

    def __iter__(self):
        return iter(self.force_limits)
    
    def add_force(self,force):
        if self.foot_name is None:
            self.foot_name = force.foot_name
            self.force_limits.append(force)
        else:
            if force.foot_name != self.foot_name:
                raise ValueError(
                    f"Mismatch in foot name: expected '{self.foot_name}', got '{force.foot_name}'"
                )
            self.force_limits.append(force)
    
    def find_polytope(self):
        if self.force_limits is not None:
            if self.contact is not None:
                self.find_contact_polytope()
            else:
                l = len(self.force_limits)
                verts = np.zeros((l, 3))
                for i, f in enumerate(self.force_limits):
                    point = f.force/1000 # from mm to m
                    verts[i] = point
                verts = np.array(verts).T
                self.polytope = ply.Polytope()
                self.polytope.find_from_point_cloud(verts)
        else:
            print ("no list of limit forces\n")

    def find_contact(self, contacts):
        for c in contacts:
            if c.in_contact is not False: 
                for f in self.force_limits:
                    if f.foot_name == c.foot_name : 
                        self.contact = c
        if self.contact is None:
            print ("No match with contact names \n")

    def find_contact_polytope(self, contacts=None):
        if self.contact is None:
            if contacts is not None:
                self.find_contact(contacts)
            else:
                print ("contact uknown and not specified\n")

        l = len(self.force_limits)
        verts = np.zeros((l, 3))

        for i, f in enumerate(self.force_limits):
            point = self.contact.contact_point + f.force/1000 # from mm to m
            verts[i] = point
        self.poly_verts_contact = np.array(verts).T
        self.polytope = ply.Polytope()
        self.polytope.find_from_point_cloud(self.poly_verts_contact)



def generate_tau_limits(x):
    """
    Generates all combinations of a vector of size 3 with values x and -x .
    
    Args:
        x (float): The value that each element of the vector can take (positive or negative).
        
    Returns:
        list: List of numpy arrays representing all possible combinations.
    """
    vectors = []
    
    # Nested loops to explore all combinations
    for i in [x, -x]:
        for j in [x, -x]:
            for k in [x, -x]:
                vectors.append(np.array([i, j, k]))
    
    return vectors

def read_feet_site_xml(model_path : str):

    # Carica il file XML
    tree = ET.parse(model_path)
    root = tree.getroot()

    # Cerca i nomi dei site che hanno una geom associata con la classe 'foot'
    EE_site = []

    for geom in root.findall(".//geom"):
        class_attr = geom.get('class')
        print(f"Found geom: {geom.attrib}")  # Mostra tutti gli attributi di ogni geom
        if class_attr == 'foot':
            print(f"Found geom with class 'foot': {geom.attrib}")
            # body = geom.find("../body")
            body = geom.findall("..")
            for site in body.findall(".//site"):
                if site.get('name'):
                    EE_site.append(site.get('name'))

    print("End-effector sites with 'foot' class:", EE_site)
    return


def calculate_force_limit(EE_site, pmodel, pdata, q, q_vel, q_acc):
    # Extract the joint space inertial matrix
    full_M = np.zeros((pmodel.nv, pmodel.nv))
    full_M = pin.crba(pmodel, pdata, q)

    # Reduced inertia matrix for the base
    M_b = full_M[:6, :3]

    # Base (COM) acceleration
    base_acc = q_acc[:6]        # COM acc

    # Coriolis, Gentrifugal and Gravity terms 
    c = pin.nonLinearEffects(pmodel, pdata, q, q_vel)

    # Product Mb * vdot
    Mb_vdot = np.dot(M_b.T, base_acc)

    # Get max torque
    # tau = np.array([23.7, 23.7, 23.7])
    taus = generate_tau_limits(23.7)


    # Product Mb * vdot
    Mb_vdot = np.dot(M_b.T, base_acc)

    # List to store force limits
    leg_force_limits = []

    # Update the data and compute the model Jacobian
    pin.framesForwardKinematics(pmodel,pdata,q)
    pin.computeJointJacobians(pmodel,pdata,q)
    pin.updateFramePlacements(pmodel,pdata)

    legs_dof_range = {}
    # Get the iD index and associated joint names     # debug: pmodel.names.tolist()
    for ee in EE_site: 
        legs_dof_range[ee] = [i for i, name in enumerate(pmodel.names) if name.startswith(ee)]


    # Nome del file in cui salvare i risultati
    output_file = "force_limits.csv"


    # for idx, tau in enumerate(taus):
    for EE , indices in legs_dof_range.items():   

        force_limits = []
        force_limits_class = ForcePolytopeBuilder()

        # for EE , indices in legs_dof_range.items():
        for idx, tau in enumerate(taus):
            leg_dof_range = []
            # Find DOF index associated with the joint id
            for i in indices:
                leg_dof_range.append(pmodel.joints[i].idx_v)

            # Reduced inertia matrix and Coriolis/gravity terms for the leg
            M_i = full_M[np.ix_(leg_dof_range, leg_dof_range)]
            c_and_g_i = c[leg_dof_range]

            # Product Mi * q_ddot (null)
            q_ddot = q_acc[leg_dof_range]
            Mi_qddot = np.dot(M_i, q_ddot)

            # inner term eq 3 from paper
            delta = Mb_vdot + Mi_qddot + c_and_g_i

            # Compiute the Jacobian and select the 3x3 submatrix
            frame_name = EE+"_FOOT"                     # when using XML use _foot
            frame_id = pmodel.getFrameId(frame_name)
            J = pin.getFrameJacobian(pmodel, pdata, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)       

            # Compute the transposed Jacobian pseudo-inverse 
            JT_inv = np.linalg.pinv(J[:3,leg_dof_range].T)

            # for i in range(2): 
            #     # Calculate the force limit
            #     force_lim = JT_inv @ (delta +tau*(-1)**(i+1))

            force = JT_inv @ (delta - tau)
            magnitude = np.linalg.norm(force)
            force_lim = LimitForce(EE, force, magnitude)
            force_limits.append(force_lim)

            # print
            print(f"{EE}: {force}   Magnitude: {magnitude}")

        # Compute the polytope
        leg_force_limits.append(ForcePolytopeBuilder(force_limits, foot_name=EE))

    print("\n")
    return leg_force_limits



def read_force_limits_from_file(file_path):
    """
    Reads the CSV file containing the limit forces and recreates the instances of limit forces.
        Args:       file_path (str): The path to the CSV file to be read.
        Returns:    list: A list of LimitForce istances.

    example: force_limits = fwp.read_force_limits_from_file("force_limits.csv")    
    """
    force_limits = []

    with open(file_path, "r") as f:
        
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split(",")
            EE = parts[0] 
            force = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            magnitude = float(parts[4])  
            force_lim = LimitForce(EE,force, magnitude)

            force_limits.append(force_lim)

    return force_limits

def contacts_linking(leg_forces_limit, contacts):
    for f in leg_forces_limit:
        f.find_contact(contacts)

def compute_polytopes(leg_forces_limits):
    poly_list = []
    for f in leg_forces_limits:
        f.find_polytope()
        poly_list.append(f.polytope)
    return poly_list


# Friction cone **********************************************************************************************************************************

def compute_friction_pyramids(contacts):
    """
    Computes the vertices of the friction pyramid for a contact point.
    
    Args:
        mu (float): Coefficient of friction.
        p (array-like): Contact point [p_x, p_y, p_z].
        N (array-like): Normal to the point of contact [N_x, N_y, N_z].
        h (float): Pyramid height.
        n (int): Number of edges of the base (default: 4, for a square pyramid).
        
    Returns:
        np.ndarray: Array of vertices of the friction pyramid (form: (n, 3)).
    """

    # List to store pyramids, one fpr each contact
    pyramids = []
    # pyram_vexs = np.array()

    for contact in contacts:
        if contact.in_contact:
            EE = contact.foot_name
            mu = contact.mu[0]
            n = contact.n
            h = contact.h
            N = contact.normal
            p = contact.contact_point 
        

            # Normalizes N
            N = np.array(N)
            N = N / np.linalg.norm(N)

            # Computes two vectors orthogonal to the normal for the tangent plane
            t1 = np.array([-N[1], N[0], 0]) if N[2] == 0 else np.cross(N, [1, 0, 0])
            t1 = t1 / np.linalg.norm(t1)
            t2 = np.cross(N, t1)
            t2 = t2 / np.linalg.norm(t2)

            # Base radius of the pyramid
            r = mu * h

            # Calculate the vertices of the base [p, n base vertices]
            vertices = np.zeros((n+1, 3))
            vertices[0] = p
            for i in range(n):
                angle = (2 * np.pi / n) * i
                vertex_base = r * (np.cos(angle) * t1 + np.sin(angle) * t2)
                vertex = p + h * N + vertex_base  # Traslazione nel sistema globale
                # vertices.append(vertex)
                vertices[i+1] = vertex

            vertices = np.array(vertices)
            poly = ply.Polytope()  
            poly.find_from_point_cloud(vertices.T)
            poly.find_grouped_vertices()
            pyram = ply.ContactPolytope(contact, EE, poly)
            # pyram = Poly3DCollection(poly.grouped_vertices, facecolors='cyan', linewidths=1.0, edgecolors=(1,0,1,0.1), alpha=.25)
            pyramids.append(pyram)  
            
        else:
            print(f"Warning: {contact.foot_name_mj} foot not in contact")

    
    # visualize_friction_pyramid(pyram_vexs)
    return pyramids




def display_PPolys_with_contacts(Fpoly):

    """
    Visualizes the friction pyramids for multiple contact points.

    Args:
        pyram_vexs (list): List of np.ndarray arrays, each representing the vertices of a friction pyramid.
                            The first point is the contact point, the others are the vertices of the base.
        force_limits (np.ndarray): A 3D vector representing the force limits to visualize.                    
    """


    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color = 'cyan'
    legend_elements = []
    legeend_labels = []
    p_min = np.zeros(3)
    p_max = np.zeros(3)

    if Fpoly is None:
            print("Error: Empty pyramid list found.")
            return

    for j, forces in enumerate(Fpoly):

        if forces is None:
            print("Error: Empty pyramid data found.")
            return
        
        forces.find_grouped_vertices()
        pyram = Poly3DCollection(forces.grouped_vertices, facecolors=color, linewidths=1.0, edgecolors=(1,0,1,0.1), alpha=.25)
        ax.add_collection3d(pyram)

        vertices = np.array(forces.vertices)
        ax.scatter(vertices[0], vertices[1], vertices[2], color='grey') #display vertices
        for i, verts in enumerate(vertices):
            p_min = np.minimum(p_min, np.min(verts,axis = 0))
            p_max = np.maximum(p_max, np.max(verts,axis = 0))
        if j == 0:
            plot_pyram = Patch(color=color)
            legend_elements.append((plot_pyram))
            legeend_labels.append(("Friction pyramids"))

         #  Dispaly the Contact point and force vector
        contact = forces.contact 
        if contact is not None:
            force = (np.array([contact.fx, contact.fy, contact.fz]))/100
            pos = contact.contact_point
            plot_contact = ax.scatter(pos[0], pos[1], pos[2], color='red')
            if j == 0:
                legend_elements.append((plot_contact))
                legeend_labels.append(("Contact points"))
            origin = pos  # Assuming the vector starts at the origin
            plot_force = ax.quiver( origin[0], origin[1], origin[2], force[0], force[1], force[2], color='blue', linewidth=2)
            if j == 0:
                legend_elements.append((plot_force))
                legeend_labels.append(("Contact force"))  
      
        # Draw the vertices of the pyramid base
        # base_vertices = pyram[1:]  # Base vertices
        # ax.scatter(base_vertices[:, 0], base_vertices[:, 1], base_vertices[:, 2], color='blue', label="vertices of the pyramid")

        # # Draw the edges of the pyramid
        # for vertex in base_vertices:
        #     ax.plot([p[0], vertex[0]], [p[1], vertex[1]], [p[2], vertex[2]], color='blue', linestyle='-', linewidth=1.5)

        # # Draw the base of the pyramid
        # base_polygon = Poly3DCollection([base_vertices], alpha=0.2, facecolor='cyan', edgecolor='black')
        # ax.add_collection3d(base_polygon)

    # Set axis limits
    ax.set_xlim([p_min[0],p_max[0]])
    ax.set_ylim([p_min[1],p_max[1]])
    ax.set_zlim3d([0,p_max[2]+ 0.4])

    # Set labels in LaTeX
    ax.set_xlabel(r'$ x [m] $')
    ax.set_ylabel(r'$ y [m] $')
    ax.set_zlabel(r'$ z [m] $')

    ax.set_title("Friction pyramids")
    plt.legend(legend_elements, legeend_labels)
    plt.show()
    print("\n")


def display_more(poly_list, legend_name=' default_legend_name'):

    if len(poly_list) > 7:
        colors = ['lightcoral', 'pink', 'violet', 'plum', 'cyan', 'cyan', 'cyan', 'cyan', 'gray', 'gray']
    else: 
        if len(poly_list) > 5:
            colors = ['lightcoral', 'pink', 'violet', 'plum', 'cyan', 'cyan', 'cyan', 'gray', 'gray']
        else: 
            colors = ['lightcoral', 'pink', 'violet', 'plum', 'cyan', 'gray', 'plum', 'darkred', 'plum', 'purple', 'darkviolet']

    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    legend_elements = []
    legend_labels = []
    p_min = np.zeros(3)
    p_max = np.zeros(3)
    n = len(poly_list)
    contact_poltted = False

    for j, poly in enumerate(poly_list):
        if poly is not None:
            if poly.vertices is None:
                poly.find_vertices()   
            if poly.faces is None:
                poly.face_indices = ply.vertex_to_faces(poly.vertices)
                poly.find_grouped_vertices()
                poly.faces = ply.face_index_to_vertex(poly.vertices,poly.face_indices)

            vertices = np.array(poly.vertices)
            ax.scatter(vertices[0], vertices[1], vertices[2], color='grey')
            for i, verts in enumerate(vertices):
                p_min[i] = np.minimum(p_min[i], np.min(verts))
                p_max[i] = np.maximum(p_max[i], np.max(verts))

            if hasattr(poly,'contact') and poly.contact is not None:
                plot_contact = ax.scatter(poly.contact.contact_point[0], poly.contact.contact_point[1], poly.contact.contact_point[2], color='red')
                if contact_poltted is False:
                    contact_poltted = True
                    legend_elements.append((plot_contact))
                    legend_labels.append(("Contact points"))

            ax.add_collection3d(Poly3DCollection(poly.grouped_vertices, facecolors=colors[j], linewidths=1.0, edgecolors=(1,0,0,0.1), alpha=.25))     
            plot_patch = Patch(color=colors[j]) 
            legend_elements.append((plot_patch))
            if hasattr(poly, 'foot_name') and poly.foot_name is not None:
                legend_labels.append((poly.foot_name + legend_name))
            else:
                legend_labels.append((legend_name))

    ax.set_xlim([p_min[0],p_max[0]])
    ax.set_ylim([p_min[1],p_max[1]])
    ax.set_zlim3d([p_min[2],p_max[2]])

    # Set labels in LaTeX
    ax.set_xlabel(r'$ x [m] $')
    ax.set_ylabel(r'$ y [m] $')
    ax.set_zlabel(r'$ z [m] $')

                                                                                                                                                                                                    
    plt.legend(legend_elements, legend_labels)
    plt.show()        
    print('/n')    

    
def display_FPolys_with_contacts(leg_force_limits):

    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['lightcoral', 'pink', 'violet', 'plum', 'darkred', 'plum', 'purple', 'darkviolet']
    legend_elements = []
    legeend_labels = []
    p_min = np.zeros(3)
    p_max = np.zeros(3)

    for j, forces in enumerate(leg_force_limits):
        contact = forces.contact
        if contact.in_contact:
            plot_contact = ax.scatter(contact.contact_point[0], contact.contact_point[1], contact.contact_point[2], color='red')
            if j == 0:
                legend_elements.append((plot_contact))
                legeend_labels.append(("Contact points"))

            # # ax.scatter(c.contact_point[0], c.contact_point[1], c.contact_point[2], color=colors[j], label=c.foot_name + " geometry")
            verts = np.zeros((8, 3))
            i = 0
            for f in forces:
                if f.foot_name == contact.foot_name : 
                    point = contact.contact_point + f.force/1000 # from mm to m
                    verts[i] = point
                    # ax.scatter(point[0], point[1], point[2], color=(1,0,0,1)) # , label="force point"
                    # ax.plot([c.contact_point[0], point[0]], [c.contact_point[1], point[1]], [c.contact_point[2], point[2]], color=colors[j], linestyle='-', linewidth=1.5)
                    i = i + 1

            verts = np.array(verts)
            p_min = np.minimum(p_min, np.min(verts,axis = 0))
            p_max = np.maximum(p_max, np.max(verts,axis = 0))

            verts = verts.T
            ax.scatter(verts[0], verts[1], verts[2], color='grey') #display vertices

            # polytope
            p = ply.Polytope(verts)
            p.find_grouped_vertices()
            ax.add_collection3d(Poly3DCollection(p.grouped_vertices, facecolors=colors[j], linewidths=1.0, edgecolors=(1,0,0,0.1), alpha=.25))     
            plot_patch = Patch(color=colors[j]) 
            legend_elements.append((plot_patch))
            legeend_labels.append((contact.foot_name + " force geometry"))

    # Set axis limits
    ax.set_xlim([p_min[0],p_max[0]])
    ax.set_ylim([p_min[1],p_max[1]])
    ax.set_zlim3d([p_min[2],p_max[2]])

    # Set labels in LaTeX
    ax.set_xlabel(r'$ x [m] $')
    ax.set_ylabel(r'$ y [m] $')
    ax.set_zlabel(r'$ z [m] $')


    plt.legend(legend_elements, legeend_labels)
    plt.show()        

def compute_intersection(leg_force_limits, Fpoly):
    intersection_polys = []
    for p_FLim in leg_force_limits:
        for p_Pyram in Fpoly:
            if p_FLim.foot_name == p_Pyram.foot_name:
                intersection = p_Pyram.intersection(p_FLim.polytope)
                INTPoly = ply.ContactPolytope(p_Pyram.contact, p_Pyram.foot_name, intersection)
                intersection_polys.append(INTPoly)
    return intersection_polys


def compute_wrench_polytope(intersection_polys ):
    origin = np.zeros(3) # debug
    WPolys = []
    WPoly3Ds = []
    

    def set_equal_aspect(ax, data_points):
        # Calcola i limiti globali basati su tutti i punti
        x_limits = [min(data_points[0, :]), max(data_points[0, :])]
        y_limits = [min(data_points[1, :]), max(data_points[1, :])]
        z_limits = [min(data_points[2, :]), max(data_points[2, :])]
        global_min = min(x_limits[0], y_limits[0], z_limits[0])
        global_max = max(x_limits[1], y_limits[1], z_limits[1])
        ax.set_xlim([global_min, global_max])
        ax.set_ylim([global_min, global_max])
        ax.set_zlim([global_min, global_max])


    for poly in intersection_polys:
        if poly.contact is None: 
            print ("contact uknown and not specified\n")
        else: 
            contact = poly.contact
            foot_name = poly.foot_name
            if hasattr(poly, 'polytope'):
                foot_name = poly.foot_name
                poly = poly.polytope
            else:
                if poly.vertices is None:
                    poly.find_vertices()

            p = contact.contact_point
            torques = []
            for i in range(poly.vertices.shape[1]): 
                vertex = poly.vertices[:, i]
                # vertex = poly.vertices[:, i] - p        # change reference
                # print(f"Vertice {i}: {vertex}")
                torque = np.cross(p,vertex)
                torques.append(torque)


                # # cross product figure ( rendi assi stessa ordine di misura)
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(np.array(torques)[:,0], np.array(torques)[:, 1], np.array(torques)[:, 2], color='grey', label="ciaaa")
                # ax.scatter(torque[0], torque[1], torque[2], color='green', label="ciaaa")
                # ax.quiver( origin[0], origin[1], origin[2], p[0], p[1], p[2], color='blue', linewidth=2)
                # # ax.quiver( p[0], p[1], p[2], vertex[0], vertex[1], vertex[2], color='red', linewidth=2)
                # ax.quiver( origin[0], origin[1], origin[2], vertex[0] - p[0], vertex[1] - p[1], vertex[2] - p[2], color='red', linewidth=2)
                # ax.quiver( origin[0], origin[1], origin[2], torque[0], torque[1], torque[2], color='green', linewidth=2)
                # all_points = np.array([origin, p, vertex, torque]).T
                # set_equal_aspect(ax, all_points)
                # plt.show() 

            torques = np.array(torques).T

            # torques_cited = clean_and_add_point_out_of_plane(torques)
            # wrench = ply.Polytope()
            # wrench.find_from_point_cloud(points=np.array(torques))
            # wrench.display('red')

            wrench_vertices = np.vstack((poly.vertices, torques))
            wrench = ply.Polytope()
            wrench.find_from_point_cloud(points=np.array(wrench_vertices))
            WPoly = ply.ContactPolytope(contact, foot_name, vertices = wrench_vertices)
            WPoly.find_grouped_vertices()
            WPolys.append(WPoly)
            WPoly3D = ply.ContactPolytope(contact, foot_name, vertices = wrench_vertices[3:, :])
            WPoly3D.find_grouped_vertices()
            WPoly3Ds.append(WPoly3D)

    return WPolys,WPoly3Ds

def Minkowski_Sum(Wpoly):
    # sum = Wpoly[0]
    # for poly in Wpoly[1:]:
    #     sum = sum + poly 
    sum1 = Wpoly[0] + Wpoly[1]
    sum2 = Wpoly[2] + Wpoly[3]
    sum = sum1 + sum2

    FWP = ply.ContactPolytope(polytope=sum)
    return FWP    

def clean_and_add_point_out_of_plane(points, distance=0.01):
    
    # Trasporre la matrice per lavorare con i punti come righe
    points_T = points.T

    # Filtra i punti per mantenere solo quelli unici
    filtered_points = np.unique(points_T, axis=0)

    # Verifica che ci siano almeno 3 punti unici
    if len(filtered_points) < 3:
        raise ValueError("La lista deve contenere almeno tre punti distinti.")
    
    # Trova tre punti distinti
    found = False
    for i in range(len(filtered_points) - 2):
        if len(np.unique(filtered_points[i:i + 3], axis=0)) == 3:
            p1, p2, p3 = filtered_points[i], filtered_points[i + 1], filtered_points[i + 2]
            found = True
            break

    if not found:
        raise ValueError("Impossibile trovare tre punti distinti nella lista.")

    # # Converti i punti in array numpy
    # p1, p2, p3 = np.array(filtered_points[0]), np.array(filtered_points[1]), np.array(filtered_points[2])
    
    # Calcola due vettori nel piano
    v1 = p2 - p1
    v2 = p3 - p1

    # Calcola il normale al piano (prodotto vettoriale)
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)  # Normalizza il vettore normale

    # Aggiungi il nuovo punto
    new_point = p1 + distance * normal
    # filtered_points_update = np.vstack([filtered_points, new_point])
    filtered_points[i] = new_point

    # Ritorna la lista dei punti, inclusa il nuovo punto
    return filtered_points.T

def reduction3D(FWP):
    if FWP.vertices is None:
        print ("vertices uknown or not specified\n")
    else: 
        FWP3D = ply.Polytope(vertices = FWP.vertices[:3, :])
        FWP3D.find_grouped_vertices()
    return FWP3D


# def remove_repeated_vertices(poly):
#     """
#     Removes "repeated" generators from the polytope if all six coordinates
#     differ by at most 0.001. In other words, it identifies duplicates
#     among the polytope's generators and removes one of them.

#     :param primal_polytope: A Python object that manages a collection of 6D generators
#                             and supports numberOfGenerators(), getGenerator(), 
#                             and removeGenerator() methods.
#     """
#     i = 0
#     while i < len(poly.vertecies):
#         new_red = False  # Flag to indicate if a generator was found redundant
#         j = i + 1
#         while j < len(poly.vertecies):
#             # Compare each of the 6 coordinates with a small tolerance (0.001)
#             coords_are_close = all(
#                 abs(poly.vertecies[i][k] -
#                     poly.vertecies[i][k]) <= 0.001
#                 for k in range(6)
#             )

#             if coords_are_close:
#                 print(f"vertex {i} is redundant")
#                 # Remove the i-th generator (the "duplicate")
#                 # poly.removeGenerator(i)
#                 new_red = True
#                 # Break because we must restart checking from the new i-th generator
#                 # (all indices have now shifted by 1 after removal).
#                 break
#             else:
#                 j += 1

#         # If a generator was removed, do NOT advance 'i'; re-check the new generator at index i
#         if not new_red:
            # i += 1



def compute_feasibility_metric(FWP, w_GI=np.zeros(6)):
    """
    Computes the feasibility metric `s` based on the feasible wrench polytope (FWP) and the gravito-inertial wrench (w_GI).

    Args:
        FWP (Polytope): The feasible wrench polytope in vertex form.
        w_GI (np.ndarray): The gravito-inertial wrench, a 6D vector.

    Returns:
        float: Feasibility metric `s`.
    """

    if FWP.vertices is None:
        FWP.find_vertices()

    # Extract vertices of the polytope as a matrix (V)
    V = np.array(FWP.vertices).T  # Shape (6, nv), where nv is the number of vertices
    nv = V.shape[1]

    # Define the LP problem
    c = np.zeros(nv + 1)  # Objective vector for LP (LP solvers, talways minimize) : maximize s (last variable)
    c[-1] = -1           # Coefficients: minimize -s, which is equivalent to maximizing s

    # Equality constraint: V * lambda = w_GI
    A_eq = np.hstack((V, -w_GI.reshape(-1, 1)))  # Add -s term as an extra column
    b_eq = w_GI

    # Inequality constraints: lambda_i >= 0 and s <= 1
    A_ub = np.zeros((nv + 1, nv + 1))
    np.fill_diagonal(A_ub[:nv, :nv], -1)  # -lambda_i <= 0
    A_ub[-1, -1] = 1                      # s <= 1
    b_ub = np.zeros(nv + 1)
    b_ub[-1] = 1

    # Solve the LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')

    if res.success:
        s = -res.fun  # LP minimizes -s, so the result needs to be negated
        return s
    else:
        print("LP did not converge. The wrench might be infeasible.")
        return -np.inf  # Indicate infeasibility