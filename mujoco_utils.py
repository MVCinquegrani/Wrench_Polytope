#   /**
#    * mujoco state = [basePos(0-2) baseQuad(3-6) q(7-18)]
#    *
#    * basePos: Base position in Global frame (3x1)
#    * baseQuad: Base orientation in Global frame (w,x,y,z) (4x1)
#    * q: Joint angles per leg [HAA, HFE, KFE] (3x1) [4x]
#    *
#    * mujoco velocity = [v(0-2) w(3-5) qj(6-17)]
#    *
#    * v: Base linear velocity in Global Frame (3x1)
#    * w: Base angular velocity in Base/Local Frame (3x1)
#    * qj: Joint velocities per leg [HAA, HFE, KFE] (3x1)
#    *
#    * mujoco Accelerations same space as the corresponding velocities
#    *
#    */

import mujoco
from mujoco import viewer
import numpy as np

def load_model_from_path(path):
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    print(mujoco.__version__)

    return model, data

def get_initial_qpos(model):
    return model.key_qpos[0]

def get_sensor_names(model):
    sensor_names = []
    for i in range(model.nsensor):
        sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        sensor_names.append(sensor_name)
    return sensor_names

def get_contact_forces(data, model, geom1, geom2):
    contact_forces = []
    for i in range(data.ncon):
        contact = data.contact[i]
        if (contact.geom1 == geom1 and contact.geom2 == geom2) or (contact.geom1 == geom2 and contact.geom2 == geom1):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            contact_forces.append(force[:3])  # Only take the force part, not the torque
    return contact_forces


def mujoco_control_and_simulation(mjdata, mjmodel, mjqpos0):
    mjdata.ctrl[:] = mjqpos0[7:]
    mujoco.mj_step(mjmodel, mjdata) # one step simulation in Mujoco
    return

def display_in_Mujoco(mjmodel, mjdata):
    view_win = viewer.launch_passive(mjmodel, mjdata)

    # grf visualization 
    view_win.opt.flags[7] = 0
    view_win.opt.flags[8] = 0
    view_win.opt.flags[13] = 0
    view_win.opt.flags[14] = 1
    view_win.opt.flags[15] = 0
    view_win.opt.flags[16] = 1
    view_win.opt.flags[18] = 1
    view_win.opt.flags[23] = 0    

    return view_win


def read_state_from_Mujoco(mjdata,mjmodel):

    q = mjdata.qpos.copy()
    # Quaternion wxyz -> xyzw
    q[3] = mjdata.qpos[4]
    q[4] = mjdata.qpos[5]
    q[5] = mjdata.qpos[6]
    q[6] = mjdata.qpos[3]

    q_vel = mjdata.qvel.copy()
    mujoco.mju_rotVecQuat(q_vel[:3], mjdata.qvel.copy()[:3], mjdata.qpos.copy()[3:7])

    # debug
    # bho = np.array([ 0.0007963, 0.9999997, 0, 0 ])
    # mujoco.mju_rotVecQuat(q_vel[:3], mjdata.qvel.copy()[:3], bho)

    q_acc = mjdata.qacc.copy()
    mujoco.mju_rotVecQuat(q_acc[:3], mjdata.qacc.copy()[:3], mjdata.qpos.copy()[3:7])

    return q, q_vel, q_acc
