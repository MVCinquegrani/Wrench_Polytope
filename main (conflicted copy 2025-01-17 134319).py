import mujoco_utils as muj_ut
import pinocchio_utils as pin_ut
from contact_analysis import ContactManager
import FW_Polytope as fwp
import numpy as np
import pinocchio as PIN
import meshcat
import matplotlib.pyplot as plt


# Configurable paths
MODEL_PATHS = {
    'mujoco': '/home/mavicern/cernbox/UBUNTU/Desktop/legged_robot/quadruped-control-main/legged_ctrl_ws/src/mujoco/xml_file/scene_w_sensor.xml',
    'pinocchio': '/home/mavicern/cernbox/UBUNTU/Desktop/legged_robot/quadruped-control-main/legged_ctrl_ws/src/unitree-go1-control-framework/legged_examples/legged_unitree/legged_unitree_description/urdf/go1_abs.urdf',
    'end_effector_sites': '/home/mavicern/cernbox/UBUNTU/Desktop/legged_robot/quadruped-control-main/legged_ctrl_ws/src/mujoco/xml_file/go1_sensor.xml'
}


# Load the model and initialize mjdata in Mujoco
mjmodel, mjdata = muj_ut.load_model_from_path(MODEL_PATHS['mujoco'])

# Pinocchio use URDF but because of MeshCat the urdf has to have the absolute path for the meshes  
pmodel, pdata, viz = pin_ut.load_model_with_viewer (MODEL_PATHS['pinocchio'])
# fwp.read_feet_site_xml(MODEL_PATHS['end_effector_sites']) # not working

# Display robot configuration
# pin_ut.display_in_Meshcat(viz)

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

        polys = []

        if mjdata.time >= 0.3:

            # Force Geometry
            fwp.contacts_linking(leg_force_limits, contacts)
            Force_Polys = fwp.compute_polytopes(leg_force_limits, True)
            # fwp.display_more(Force_Polys, " force geometry")

            # Friction Geometry
            Fpolys = fwp.compute_friction_pyramids(contacts)
            # fwp.display_more(Fpoly, " Friction pyramids")

            # Polts customized
            # fwp.display_FPolys_with_contacts(leg_force_limits)
            # fwp.display_PPolys_with_contacts(Fpolys)


            polys = list(Force_Polys)       # copy the surfece, the object inside still points to the same memory
            polys.extend(Fpolys)
            # fwp.display_more(polys)

            # Geometry intersection (ContactPolytope type)
            intersection_polys = fwp.compute_intersection(leg_force_limits, Fpolys)
            # fwp.display_more(intersection_polys, " Intersection Geometry")

            # Add the trorque to the interection polytope
            Wpoly,WPoly3D = fwp.compute_wrench_polytope(intersection_polys) 
            fwp.display_more(WPoly3D, " torque poly")
            # blabla = []
            # blabla.append(Wpoly[0])
            # blabla.append(Wpoly[0])
            # ciaaa = fwp.Minkowski_Sum(blabla)

            # Compute the Minkowski Sum of the intersections (pyc) (line 171 polytope_geom file)
            FWP = fwp.Minkowski_Sum(WPoly3D)
            # FWP3D = fwp.reduction3D(FWP)
            FWP.display('green')    

            # # w_GI = np.array([...])  # Replace with the actual gravito-inertial wrench vector.
            # s = fwp.compute_feasibility_metric(Wpoly)
            # print("Feasibility metric:", s)



            

        # Update Meshcat
        # viz.display(q)

        # for frame in pmodel.frames:
        #     # Ottieni il nome e la posizione del frame
        #     frame_name = frame.name
        #     frame_placement = frame.placement

        #     # Aggiungi un frame al visualizzatore
        #     viz.viewer["frames/" + frame_name].set_transform(frame_placement.homogeneous)
        #     viz.viewer["frames/" + frame_name].set_object(meshcat.geometry.Box([0.01, 0.01, 0.01]))  # Una piccola rappresentazione del frame


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



# # Jacobian in Mujoco
# efc_J = mjdata.efc_J
# expected_size = 36 * 18
# if efc_J.size != expected_size:
#     raise ValueError(f"La dimensione dell'array efc_J ({efc_J.size}) non corrisponde a quella prevista ({expected_size}).")
# dense_matrix = efc_J.reshape((36, 18))
# np.set_printoptions(suppress=True,linewidth=500)
# dense_matrix_pretty = np.array2string(dense_matrix,precision=6,floatmode='fixed')
# print(dense_matrix_pretty)