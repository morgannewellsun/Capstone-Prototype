
import numpy as np
import pandas as pd


def quaternion_to_rotation_matrix(quaternion_xyzw_np):
    # xyzw_np.shape = (4,)
    products = quaternion_xyzw_np[:, np.newaxis] * quaternion_xyzw_np[np.newaxis, :]
    xx, xy, xz, wx, _, yy, yz, wy, _, _, zz, wz, _, _, _, ww = products.flatten()
    n = np.trace(products)
    s = 0 if (n == 0) else (2 / n)
    rotation_matrix = np.array([
        [yy + zz, xy - wz, xz + wy],
        [xy + wz, xx + zz, yz - wx],
        [xz - wy, yz + wx, xx + yy]])
    rotation_matrix *= s
    rotation_matrix = np.eye(3) - rotation_matrix
    return rotation_matrix


def transform_position_cameraframe_to_inertialframe(
        camera_pose_pos_xyz_np, camera_pose_rot_xyzw_np, pos_to_transform_xyz_np):
    # camera_pose_pos_xyz_np.shape = (3,)
    # camera_pose_rot_xyzw_np.shape = (4,), w = real part
    # pos_cameraframe_xyz_np.shape = (3,)
    rotation_matrix_inertial_camera = quaternion_to_rotation_matrix(camera_pose_rot_xyzw_np)
    # rotation_matrix_inertial_camera = rotation_matrix_inertial_camera.T
    return np.ndarray.flatten(
        rotation_matrix_inertial_camera @ pos_to_transform_xyz_np[:, np.newaxis]
        + camera_pose_pos_xyz_np[:, np.newaxis])


class SingleObjectTracker:

    def __init__(self):

        # =====================================================================
        # HYPERPARAMETERS
        # =====================================================================

        # buffers for smoothing out detections
        self.buffer_size_entry = 2  # only trust detections after an object is seen this number of times consecutively
        self.buffer_size_exit = 2  # keep trusting detections until an object is missing for this number of updates

        # =====================================================================
        # MEMORY
        # =====================================================================


class MultiObjectTracker:

    def __init__(self, tracked_object_ids):
        """
        :param tracked_object_ids: List[int], list of object class ids to track
        """

        # =====================================================================
        # HYPERPARAMETERS
        # =====================================================================

        # list of object class ids to track
        self.tracked_object_ids = tracked_object_ids

        # =====================================================================
        # MEMORY
        # =====================================================================
        self.single_object_trackers_dict = dict()
        for tracked_object_id in self.tracked_object_ids:
            self.single_object_trackers_dict.update({tracked_object_id: SingleObjectTracker()})


    def update(
            self,
            camera_pose_xyz_np, camera_pose_rot_xyzw_np,
            detection_time, detection_objects_ids, detection_objects_pos_xyz_np
    ):

        # convert all detections of interest to the inertial frame
        updated_objects_id = []
        updated_objects_inertial_pos = []
        for object_id, object_pos_xyz_np in zip(detection_objects_ids, detection_objects_pos_xyz_np):
            if object_id in self.tracked_object_ids:
                object_pos_xyz_np = np.array([object_pos_x, -1 * object_pos_y, object_pos_z])  # flip y-coord
                updated_objects_id.append(object_id)
                updated_objects_inertial_pos.append(transform_position_cameraframe_to_inertialframe(
                    camera_pose_xyz_np, camera_pose_rot_xyzw_np, object_pos_xyz_np))

        #




def main(filepath_combined, filepath_unpacked):

    # load and parse data
    combined_df = pd.read_csv(filepath_combined, index_col=None)
    camera_poses_pos_xyz_np = combined_df[
        ["slam_pose_pos_x", "slam_pose_pos_y", "slam_pose_pos_z"]].values
    camera_poses_rot_xyzw_np = combined_df[
        ["slam_pose_rot_x", "slam_pose_rot_y", "slam_pose_rot_z", "slam_pose_rot_w"]].values
    detection_indices_np = combined_df["detection_idx"].values
    detection_times_np = combined_df["detection_time"].values
    unpacked_df = pd.read_csv(filepath_unpacked, index_col=None)
    detection_idx_objects_map = dict()
    for detection_idx, detection_df in unpacked_df.groupby("detection_idx"):
        detection_idx_objects_map.update({detection_idx: list(zip(
            detection_df["detection_object_id"].values,
            detection_df[["detection_object_pos_x", "detection_object_pos_y", "detection_object_pos_z"]].values))})

    # make an object tracker
    my_mot = MultiObjectTracker(tracked_object_ids=[39, 41])

    # send each detection to the tracker
    for detection_idx in detection_indices_np:
        detection_objects = detection_idx_objects_map.get(detection_idx, [])
        detection_objects_id = []
        detection_objects_pos_xyz_np = []
        for detection_object_id, detection_object_pos_xyz_np in detection_objects:
            detection_objects_id.append(detection_object_id)
            detection_objects_pos_xyz_np.append(detection_object_pos_xyz_np)
        my_mot.update(
            camera_poses_pos_xyz_np[detection_idx], camera_poses_rot_xyzw_np[detection_idx],
            detection_times_np[detection_idx], detection_objects_id, detection_objects_pos_xyz_np)


if __name__ == "__main__":

    # inputs
    filepath_combined = r"D:\Documents\Academics\ROB498\Capstone-Prototype\data\combined.csv"
    filepath_unpacked = r"D:\Documents\Academics\ROB498\Capstone-Prototype\data\unpacked.csv"

    main(filepath_combined, filepath_unpacked)
