import mujoco
import numpy as np

class ContactPoint:
    def __init__(self, model, floor_id, foot_name_mj, foot_name, h, n):
        self.floor_id = floor_id
        self.foot_name = foot_name
        self.foot_name_mj = foot_name_mj
        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, foot_name_mj)
        self.frame = "world"  # Placeholder
        self.mu = model.geom_friction[self.id]
        self.h = h 
        self.n = n   
        self.reset()


    def reset(self):
        self.in_contact = False
        self.fx = 0
        self.fy = 0
        self.fz = 0
        self.normal = np.zeros(3)                   # Vector normal to the surface in the global grame
        self.contact_point = np.zeros(3) 

    def parse_forces(self, model, mjdata, contact_id):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, mjdata, contact_id, force)
        self.in_contact = True
        self.fx = force[2]
        self.fy = force[1]
        self.fz = force[0]

        contact = mjdata.contact[contact_id]
        self.normal = np.array(contact.frame[:3])   # The first three values are the vector normal to the surface
        self.contact_point = np.array(contact.pos)  # Contact point in global coordinates

class ContactManager:
    def __init__(self, model, h, n, EE_site):
        # mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 0)  # debug
        self.floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")   
        setattr(self, EE_site[0], ContactPoint(model, self.floor_id, "FL", EE_site[0], h, n))
        setattr(self, EE_site[1], ContactPoint(model, self.floor_id, "FR", EE_site[1], h, n))
        setattr(self, EE_site[2], ContactPoint(model, self.floor_id, "RL", EE_site[2], h, n))
        setattr(self, EE_site[3], ContactPoint(model, self.floor_id, "RR", EE_site[3], h, n))
        self.contactpoints = [getattr(self, name) for name in EE_site]

    def __iter__(self):
        return iter(self.contactpoints)

    def get_foot_contact_forces(self, model, mjdata):
        for f in self.contactpoints:
            f.reset()
            for i in range(mjdata.ncon):
                contact = mjdata.contact[i]
                if (contact.geom1 == f.id or contact.geom2 == f.id) and (contact.geom1 == self.floor_id or contact.geom2 == self.floor_id):
                    f.parse_forces(model, mjdata, i)
                    
    def print_contact_forces(self):
        for f in self.contactpoints:
            magnitude = np.linalg.norm(np.array([f.fx, f.fy, f.fz]))
            print(f"{f.foot_name} forces: ({f.fx}, {f.fy}, {f.fz})      Magnitude: {magnitude}")
        print("\n")    

        
    def update(self, model, mjdata):
        self.get_foot_contact_forces(model, mjdata)
