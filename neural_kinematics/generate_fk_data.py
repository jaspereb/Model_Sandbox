import math
import pytorch_kinematics as pk
import rerun as rr
from rerun_utils import log_xyz_quat
import numpy as np
import random
import pickle

DEBUG = False
MIN_JOINT = -math.pi
MAX_JOINT = math.pi

rr.init("fk_data", spawn=True)
chain = pk.build_serial_chain_from_urdf(open("ur5e.urdf", "rb").read(), "ee_link")
print(chain)
print(chain.get_joint_parameter_names())

full_data = []
for i in range(5000):
    # specify joint values (can do so in many forms)
    joints = [random.uniform(MIN_JOINT, MAX_JOINT) for _ in range(6)]

    # do forward kinematics and get transform objects; end_only=False gives a dictionary of transforms for all links
    ret = chain.forward_kinematics(joints, end_only=False)

    tg = ret['ee_link']
    m = tg.get_matrix()
    pos = m[:, :3, 3]
    rot = pk.matrix_to_quaternion(m[:, :3, :3])

    if DEBUG:
        for link in ret:
            m = ret[link].get_matrix()
            pos = m[:, :3, 3]
            rot = pk.matrix_to_quaternion(m[:, :3, :3])
            log_xyz_quat(pos, rot, link, in_mm=False, axis_length=0.1)

    data_point = joints
    ee_matrix = m
    
    # Convert tensors to numpy arrays
    pos_tensor = ee_matrix[:, :3, 3]
    rot_tensor = pk.matrix_to_quaternion(ee_matrix[:, :3, :3])
    
    # Convert to numpy arrays and flatten
    pos_np = pos_tensor.detach().numpy().flatten()
    rot_np = rot_tensor.detach().numpy().flatten()
    
    data_point.extend(list(pos_np))
    data_point.extend(list(rot_np))

    print(data_point)
    data_point = np.asarray(data_point)
    print(f"Got data point: {data_point}")

    full_data.append(data_point)

random.shuffle(full_data)
split_idx = int(0.8 * len(full_data))
train_data = full_data[:split_idx]
test_data = full_data[split_idx:]

with open('train.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open('test.pkl', 'wb') as f:
    pickle.dump(test_data, f)