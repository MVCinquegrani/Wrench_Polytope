import mujoco
import numpy as np

class ContactStruct:
    def __init__(self, model, floor_id, foot_name):
        self.floor_id = floor_id
        self.foot_name = foot_name
        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
        self.in_contact = False
        self.frame = "world"  # Placeholder
        self.reset()

    def reset(self):
        self.in_contact = False
        self.fx = 0
        self.fy = 0
        self.fz = 0

    def parse_forces(self, model, mjdata, contact_id):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, mjdata, contact_id, force)
        self.in_contact = True
        self.fx = force[2]
        self.fy = force[1]
        self.fz = force[0]

class ContactData:
    def __init__(self, model):
        self.floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.FL = ContactStruct(model, self.floor_id, "FL")
        self.FR = ContactStruct(model, self.floor_id, "FR")
        self.RL = ContactStruct(model, self.floor_id, "RL")
        self.RR = ContactStruct(model, self.floor_id, "RR")
        self.f_struct = [self.FL, self.FR, self.RL, self.RR]

    def get_foot_contact_forces(self, model, mjdata):
        for f in self.f_struct:
            f.reset()
            for i in range(mjdata.ncon):
                contact = mjdata.contact[i]
                if (contact.geom1 == f.id or contact.geom2 == f.id) and (contact.geom1 == self.floor_id or contact.geom2 == self.floor_id):
                    f.parse_forces(model, mjdata, i)
                    
    def print_contact_forces(self):
        for f in self.f_struct:
            magnitude = np.linalg.norm(np.array([f.fx, f.fy, f.fz]))
            print(f"{f.foot_name} forces: ({f.fx}, {f.fy}, {f.fz})      Magnitude: {magnitude}")
        print("\n")    

        
    def update(self, model, mjdata):
        self.get_foot_contact_forces(model, mjdata)
