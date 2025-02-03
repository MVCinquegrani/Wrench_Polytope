import mujoco_utils as muj_ut
import pinocchio_utils as pin_ut
from contact_analysis import ContactManager
import FW_Polytope as fwp
import numpy as np
import pinocchio as PIN
# import meshcat
import matplotlib.pyplot as plt  
import os

def expand_path(path):
    if path.startswith('~'):
        return path.replace('~', f"/home/{os.environ['USER']}", 1)
    return path

MODEL_PATHS = {
    'mujoco': '~/cernbox/UBUNTU/Desktop/legged_robot/quadruped-control-main/legged_ctrl_ws/src/mujoco/xml_file/scene_w_sensor.xml',
    'pinocchio': '~/cernbox/UBUNTU/Desktop/legged_robot/quadruped-control-main/legged_ctrl_ws/src/unitree-go1-control-framework/legged_examples/legged_unitree/legged_unitree_description/urdf/go1_abs.urdf',
    'end_effector_sites': '~/cernbox/UBUNTU/Desktop/legged_robot/quadruped-control-main/legged_ctrl_ws/src/mujoco/xml_file/go1_sensor.xml',
    'mesh_dir' : '~/cernbox/UBUNTU/Desktop/legged_robot/quadruped-control-main/legged_ctrl_ws/src/unitree-go1-control-framework/legged_examples/legged_unitree/legged_unitree_description/meshes/go1/',
}

MODEL_PATHS = {key: expand_path(path) for key, path in MODEL_PATHS.items()}

# Load the model and initialize mjdata in Mujoco
mjmodel, mjdata = muj_ut.load_model_from_path(MODEL_PATHS['mujoco'])

# Pinocchio use URDF but because of MeshCat the urdf has to have the absolute path for the meshes  
pmodel, pdata = pin_ut.load_model_with_viewer (MODEL_PATHS['pinocchio'], MODEL_PATHS['mesh_dir'])

# Get the initial state from xml 
mjqpos0 = muj_ut.get_initial_qpos(mjmodel)
print("Initial qpos from XML:", mjqpos0)

# Pos update
mjdata.qpos[:] = mjqpos0    # Mujoco
pqpos0 = mjqpos0.copy()     # Pinocchio

# Display robot in Mujoco
view_win = muj_ut.display_in_Mujoco(mjmodel, mjdata)

# Pyramid parameters
h = 0.3  # Altezza della piramide
n = 4  # Numero di lati della base

# Names of the end-effector sites
EE_site = ["LF", "RF", "LH", "RH"]      # change name if use xml 

# Main loop
while True:
    try:

        muj_ut.mujoco_control_and_simulation(mjdata, mjmodel, mjqpos0)
        q, q_vel, q_acc = muj_ut.read_state_from_Mujoco(mjdata,mjmodel)
        leg_force_limits = fwp.calculate_force_limit(EE_site, pmodel, pdata, q, q_vel, q_acc)

        # Contact in Mujoco
        contacts = ContactManager(mjmodel, h, n, EE_site)
        contacts.update(mjmodel, mjdata)
        contacts.print_contact_forces()

        if mjdata.time >= 0.3:

            # Force Geometry
            fwp.contacts_linking(leg_force_limits, contacts)
            Force_Polys = fwp.compute_polytopes(leg_force_limits)
            # fwp.display_more(Force_Polys, " force geometry")

                                            # # Chris
                                            # Wpoly,WPoly3D = fwp.compute_wrench_polytope(leg_force_limits)
                                            # fwp.display_more(WPoly3D, " torque poly")

            # Friction Geometry
            Fpolys = fwp.compute_friction_pyramids(contacts)
            # fwp.display_more(Fpolys, " Friction pyramids")

            # # Polts customized
            # fwp.display_FPolys_with_contacts(leg_force_limits)
            # fwp.display_PPolys_with_contacts(Fpolys)

            # Geometry intersection (ContactPolytope type)
            intersection_polys = fwp.compute_intersection(leg_force_limits, Fpolys)
            # fwp.display_more(intersection_polys, " Intersection Geometry")

            # Add the trorque to the interection polytope
            WPoly,WPoly3D = fwp.compute_wrench_polytope(intersection_polys) 
            # fwp.display_more(WPoly3D, " torque poly")

            # Compute the Minkowski Sum of the intersections (pyc) (line 171 polytope_geom file)
            FWP = fwp.Minkowski_Sum(WPoly)
            FWP3D = fwp.reduction3D(FWP)
            FWP3D.display('green')

            # w_GI = np.array([...])  # Replace with the actual gravito-inertial wrench vector.
            # s = fwp.compute_feasibility_metric(FWP)
            # print("Feasibility metric:", s)


        # Update view window of Mujoco
        view_win.sync()

        # Exit loop condition for debugging (optional)
        if mjdata.time >= 5.0:
            print("Exiting simulation loop.")
            break

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        break
    # except Exception as e:
    #     print(f"Error during simulation: {e}")
    #     break

