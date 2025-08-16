import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation as R

# Quaternions are W-last in all cases

def log_matrix(m: np.ndarray, path: str, in_mm=True, static=False, axis_length: int | float = 150, from_parent=False):
    rmat = m[:3, :3]
    t = m[:3, 3]
    if in_mm:
        # because rerun uses m
        t = t / 1000
        float_axis_length = float(axis_length) / 1000
    else:
        float_axis_length = float(axis_length)
    
    relation = rr.TransformRelation.ChildFromParent if from_parent else None
    rr.log(
        path,
        rr.Transform3D(translation=t, mat3x3=rmat, axis_length=float_axis_length, relation=relation),
        static=static,
    )

def log_xyz_quat(trans, quat, path: str, in_mm=True, static=False, axis_length: int | float = 15, from_parent=False):
    t = np.array(trans)
    if in_mm:
        # because rerun uses m
        t = t / 1000
        float_axis_length = float(axis_length) / 1000
    else:
        float_axis_length = float(axis_length)

    relation = rr.TransformRelation.ChildFromParent if from_parent else None
    rr.log(
        path,
        rr.Transform3D(translation=t, quaternion=quat, axis_length=float_axis_length, relation=relation),
        static=static,
    )

def log_xyz_rpy(trans, rpy, path: str, in_mm=True, static=False, axis_length: int | float = 150, from_parent=False):
    # Convert translation and RPY to quaternion
    rot = R.from_euler('xyz', rpy)
    quat = rot.as_quat()  # (x, y, z, w) for rerun
    t = np.array(trans)
    if in_mm:
        # because rerun uses m
        t = t / 1000
        float_axis_length = float(axis_length) / 1000
    else:
        float_axis_length = float(axis_length)
    
    relation = rr.TransformRelation.ChildFromParent if from_parent else None
    rr.log(
        path,
        rr.Transform3D(
            translation=t,
            quaternion=quat,
            axis_length=float_axis_length,
            relation=relation
        ),
        static=static,
    )

if __name__ == "__main__":
    rr.init("rerun_test", spawn=True)

    m = np.eye(4)
    m[0, 3] = 1
    m[1, 3] = 2
    m[2, 3] = 3
    log_matrix(m, "test", in_mm = False)
    
    # Test pose logging
    trans = [0.5, 0.3, 0.7]
    rpy = [0.1, 0.2, 0.3]
    log_xyz_rpy(trans, rpy, "test_pose")

    quat = [0.1, 0.2, 0.3, 0.4]
    log_xyz_quat(trans, quat, "test_pose")
