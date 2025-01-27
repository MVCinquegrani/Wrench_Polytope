import mujoco
from mujoco import viewer
import numpy as np
from time import sleep

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path('/home/mavicern/cernbox/UBUNTU/Desktop/legged_robot/quadruped-control-main/legged_ctrl_ws/src/mujoco/scene_w_sensor.xml')
data = mujoco.MjData(model)

qpos0 = model.key_qpos[0]
print("Initial qpos from XML:", qpos0)
data.qpos[:] = qpos0
view_win = viewer.launch_passive(model, data)

sensor_names = []
for i in range(model.nsensor):
    sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    sensor_names.append(sensor_name)


def get_contact_forces(data, geom1, geom2):
    contact_forces = []
    for i in range(data.ncon):
        contact = data.contact[i]
        if (contact.geom1 == geom1 and contact.geom2 == geom2) or (contact.geom1 == geom2 and contact.geom2 == geom1):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            contact_forces.append(force[:3])  # Only take the force part, not the torque
    return contact_forces    


class ContactStruct:
    def __init__(self, floor_id, foot_name):
        self.floor_id = floor_id
        self.foot_name = foot_name
        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
        self.in_contact = False
        self.frame = "world? (verify)"
        self.reset()

    def reset(self):
        self.in_contact = False
        self.fx = 0
        self.fy = 0
        self.fz = 0

    def parse_forces(self, data, contact_id):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_id, force)
        self.in_contact = True
        self.fx = force[2]
        self.fy = force[1]
        self.fz = force[0]    



class ContactData:
    def __init__(self):
        self.floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.FL = ContactStruct(self.floor_id, "FL")
        self.FR = ContactStruct(self.floor_id, "FR")
        self.RL = ContactStruct(self.floor_id, "RL")
        self.RR = ContactStruct(self.floor_id, "RR")
        self.f_struct = [self.FL, self.FR, self.RL, self.RR]

    def get_foot_contact_forces(self, data):
        for f in self.f_struct:
            f.reset()
            for i in range(data.ncon):
                contact = data.contact[i]
                if (contact.geom1 == f.id or contact.geom2 == f.id) and (contact.geom1 == self.floor_id or contact.geom2 == self.floor_id):
                    f.parse_forces(data,i)
                    
    def print_contact_forces(self):
        for f in self.f_struct:
            print(f"{f.foot_name} forces: ({f.fx}, {f.fy}, {f.fz})")
    
    def update(self, data):
        self.get_foot_contact_forces(data)


def extract_intertia_matrices(model, data):
    full_M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, full_M, data.qM)

    M_b = full_M[:6,:6]
    M_i_list = []
    leg_dofs = [range(6,9), range(9,12), range(12,15), range(15,18)]

    for dofs in leg_dofs:
        M_i = full_M[np.ix_(dofs,dofs)]
        M_i_list.append(M_i)

    return M_b, M_i_list


M_b, M_i_list = extract_intertia_matrices(model,data)

# print inertia matrices
print("M_b: ", M_b)
print("\n")
for idx, M_i in enumerate(M_i_list):
    print(f"M_i leg {idx + 1}:", M_i)
    print("\n")

# grf visualization 
view_win.opt.flags[7] = 0
view_win.opt.flags[8] = 0
view_win.opt.flags[13] = 0
view_win.opt.flags[14] = 1
view_win.opt.flags[15] = 0
view_win.opt.flags[16] = 1
view_win.opt.flags[18] = 1
view_win.opt.flags[23] = 0


while True:
    # Apply control
    data.ctrl[:] = qpos0[7:]

    mujoco.mj_step(model, data)
    # sleep(0.1)

    # GRFs
    grfs = data.cfrc_ext

    # Joint torques direct from sim
    joint_torques = data.qfrc_actuator

    # Joint pos direct from sim
    joint_positions = data.qpos

    # Joint vel direct from sim
    joint_velocities = data.qvel

    # COM state from sim
    com_pos = data.subtree_com[0]
    com_vel = data.subtree_linvel[0]

    # Joint torques direct from sim
    actuator_forces = data.sensordata[:model.nsensor]

    # contacts = ContactData()
    # contacts.update(data)
    # contacts.print_contact_forces()

    # # Print values
    # print("GRFs:", grfs)
    # print("\n")
    # print("Joint Torques:", joint_torques)
    # print("\n")
    # print("Joint Positions:", joint_positions)
    # print("\n")
    # print("Joint Velocities:", joint_velocities)
    # print("\n")
    # print("COM Position:", com_pos)
    # print("\n")
    # print("COM Velocity:", com_vel)
    # print("\n")
    # print("Actuator Forces:", actuator_forces)
    # print("\n")

    # for i, sensor_name in enumerate(sensor_names):
    #     print(f"{sensor_name}: {data.sensor(sensor_name)}")
    # print("***********************************")
    # print("\n")

    # Update view window
    view_win.sync()

