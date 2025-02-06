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



import numpy as np
import pinocchio as pin
from time import sleep 
from pinocchio.visualize import MeshcatVisualizer
import sys
# import meshcat.geometry as g
# import meshcat.transformations as tf


def load_model_with_viewer (model_path_pin, mesh_dir):

    #  = "/home/mavi/cernbox/UBUNTU/Desktop/legged_robot/quadruped-control-main/legged_ctrl_ws/src/unitree-go1-control-framework/legged_examples/legged_unitree/legged_unitree_description/meshes/go1/"
    # pmodel = pin.buildModelFromUrdf(model_path_pin, pin.JointModelFreeFlyer())
    pmodel, cm, vm = pin.buildModelsFromUrdf(model_path_pin, mesh_dir, pin.JointModelFreeFlyer())
    pdata = pmodel.createData()
    # viz = MeshcatVisualizer(pmodel,cm,vm)
    return pmodel, pdata #, viz


def display_in_Meshcat(viz) :
    try:
        viz.initViewer(open=True)
    except ImportError as err:
        print(
            "Error while initializing the viewer. It seems you should install Python meshcat"
        )
        print(err)
        sys.exit(0)
 
    # Load the robot in the viewer.
    viz.loadViewerModel()
    sleep(3)
    viz.displayVisuals(True)

def p_WGI(pmodel, pdata,q, q_vel, q_acc):

    # pin.computeAllTerms(pmodel, pdata, q, q_vel)

    pin.forwardKinematics(pmodel, pdata, q)
    pin.crba(pmodel, pdata, q)
    pin.centerOfMass(pmodel, pdata, 0)  
    com_pos = pdata.com[0]
    pin.centerOfMass(pmodel, pdata, 2)
    com_acc = pdata.acom[0]

    m = pin.computeTotalMass(pmodel)
    g = g = pmodel.gravity.linear
    force_G = (m * g)/1000
    torque_G = np.cross(com_pos, force_G)
    w_G = np.hstack((force_G, torque_G)) 

    pin.computeCentroidalMomentumTimeVariation(pmodel, pdata, q, q_vel, q_acc)
    h_dot = (pdata.dhg)/1000
    placement = pin.SE3(np.eye(3), com_pos)
    h_dot_w = np.zeros(h_dot.vector.shape)
    h_dot_w = placement.act(h_dot)

    # pin.changeReferenceFrame(h_dot,1,0, h_dot_w) 

    # # pdata.dAg
    # # pdata.Ag
    # Kw_dot = q_acc[3:6]
    # h_dot = np.hstack((m*com_acc, Kw_dot)) 

    w_GI = h_dot_w + w_G
    return w_GI

