import glob
import os

import numpy as np
import pandas as pd
import cv2

flip_flop = True


def quaternion_to_rotation_matrix(quaternion_np):
    quaternion_imag_np = quaternion_np[0:3]
    i, j, k = quaternion_imag_np
    quaternion_real = quaternion_np[3]
    quaternion_cross = np.array([[0, -k, j], [k, 0, -i], [-j, i, 0]])
    rotation_matrix = (
            ((np.square(quaternion_real) - np.sum(np.square(quaternion_imag_np))) * np.eye(3))
            + (2 * quaternion_imag_np[:, np.newaxis] @ quaternion_imag_np[np.newaxis, :])
            + (2 * quaternion_real * quaternion_cross))
    return rotation_matrix


def transform_position_inertialframe_to_cameraframe(
        camera_pose_pos_np, camera_pose_rot_np, pos_to_transform_np):
    # camera_pose_pos_np.shape = (3,)
    # camera_pose_rot_np.shape = (4,), w = real part
    # pos_to_transform_np.shape = (3,)
    c = quaternion_to_rotation_matrix(camera_pose_rot_np).T
    return np.ndarray.flatten(c @ (pos_to_transform_np[:, np.newaxis] - camera_pose_pos_np[:, np.newaxis]))


def transform_position_cameraframe_to_inertialframe(
        camera_pose_pos_np, camera_pose_rot_np, pos_to_transform_np):
    # camera_pose_pos_np.shape = (3,)
    # camera_pose_rot_np.shape = (4,), w = real part
    # pos_to_transform_np.shape = (3,)
    c = quaternion_to_rotation_matrix(camera_pose_rot_np)
    return np.ndarray.flatten(c @ pos_to_transform_np[:, np.newaxis] + camera_pose_pos_np[:, np.newaxis])


class SingleObjectTracker:

    def __init__(self):

        # =====================================================================
        # HYPERPARAMETERS
        # =====================================================================

        # buffers for smoothing out detections
        self.buffer_size_entry = 2  # only trust detections after an object is seen this number of times consecutively
        self.trust_entry_buffer = False  # should the initial consecutive detections be "replayed" upon gaining trust?
        self.buffer_size_exit = 2  # keep trusting detections until an object is missing for this number of updates

        # =====================================================================
        # MEMORY
        # =====================================================================

        # buffers
        self.timesteps_since_last_detected = np.inf
        self.timesteps_consecutively_detected = 0
        self.trust_current_detections = False
        self.detections_buffer = []

        # filter
        self.object_inertialframe_pos_np = None

    def update_filter(self, detection_time, object_inertialframe_pos_np):
        self.object_inertialframe_pos_np = object_inertialframe_pos_np

    def update_detected(self, camera_pose_pos_np, camera_pose_rot_np, detection_time, object_pos_np):

        # convert from camera frame to inertial frame
        object_inertialframe_pos_np = transform_position_cameraframe_to_inertialframe(
            camera_pose_pos_np, camera_pose_rot_np, object_pos_np)

        # update buffers
        self.timesteps_since_last_detected = 0
        self.timesteps_consecutively_detected += 1
        if self.trust_current_detections:
            self.update_filter(detection_time, object_inertialframe_pos_np)
        else:
            if self.trust_entry_buffer:
                self.detections_buffer.append((detection_time, object_inertialframe_pos_np))
            if self.timesteps_consecutively_detected >= self.buffer_size_entry:
                self.trust_current_detections = True
                if self.trust_entry_buffer:
                    for buffered_detection in self.detections_buffer:
                        self.update_filter(*buffered_detection)
                    self.detections_buffer = []

    def update_undetected(self):

        # update buffers
        self.timesteps_since_last_detected += 1
        self.timesteps_consecutively_detected = 0
        if self.trust_current_detections:
            if self.timesteps_since_last_detected >= self.buffer_size_exit:
                self.trust_current_detections = False
        else:
            self.detections_buffer = []

    def get_pos_cameraframe(self, camera_pose_pos_np, camera_pose_rot_np):
        if self.object_inertialframe_pos_np is not None:
            return transform_position_inertialframe_to_cameraframe(
                camera_pose_pos_np, camera_pose_rot_np, self.object_inertialframe_pos_np)
        else:
            return None


class MultiObjectTracker:

    def __init__(self, tracked_objects_id):
        """
        :param tracked_object_ids: List[int], list of object class ids to track
        """

        # =====================================================================
        # HYPERPARAMETERS
        # =====================================================================

        # list of object class ids to track
        self.tracked_objects_id = tracked_objects_id

        # =====================================================================
        # MEMORY
        # =====================================================================
        self.single_object_trackers = dict()
        for tracked_object_id in self.tracked_objects_id:
            self.single_object_trackers.update({tracked_object_id: SingleObjectTracker()})

    def update(
            self,
            camera_pose_pos_np, camera_pose_rot_np,
            detection_time, detection_objects_id, detection_objects_pos_np):
        """
        Make sure that the y-coordinates in detection_objects_pos_np are flipped for consistency with RHR.

        :param camera_pose_pos_np: ndarray[float], shape (3,)
        :param camera_pose_rot_np: ndarray[float], shape (4,)
        :param detection_time: List[float]
        :param detection_objects_id: List[int]
        :param detection_objects_pos_np: List[ndarray[float]], each ndarray has shape (3,)
        """

        # update single object trackers for detected objects
        for object_id, object_pos_np in zip(detection_objects_id, detection_objects_pos_np):
            for tracked_object_id in self.tracked_objects_id:
                if object_id == tracked_object_id:
                    self.single_object_trackers[object_id].update_detected(
                        camera_pose_pos_np, camera_pose_rot_np, detection_time, object_pos_np)

        # update single object trackers for non-detected objects
        for tracked_object_id in self.tracked_objects_id:
            if tracked_object_id not in detection_objects_id:
                self.single_object_trackers[tracked_object_id].update_undetected()

        # return cameraframe positions for tracked objects
        return_values = []
        for tracked_object_id in self.tracked_objects_id:
            return_values.append(
                self.single_object_trackers[tracked_object_id].get_pos_cameraframe(
                    camera_pose_pos_np, camera_pose_rot_np))
        return return_values


def main(filepath_combined, filepath_unpacked, directory_frames, directory_annotated, filepath_video_annotated):

    # camera parameters
    intrinsic_matrix = np.array([[600.48120117, 0., 200.15670776], [0., 599.72149658, 214.27590942], [0., 0., 1.]])

    # video parameters
    fps = 8.25
    frame_size = (416, 416, 3)

    # load and parse data
    combined_df = pd.read_csv(filepath_combined, index_col=None)
    camera_poses_pos_np = combined_df[
        ["slam_pose_pos_x", "slam_pose_pos_y", "slam_pose_pos_z"]].values
    camera_poses_rot_np = combined_df[
        ["slam_pose_rot_x", "slam_pose_rot_y", "slam_pose_rot_z", "slam_pose_rot_w"]].values
    detection_indices_np = combined_df["detection_idx"].values
    detection_times_np = combined_df["detection_time"].values
    unpacked_df = pd.read_csv(filepath_unpacked, index_col=None)
    detection_idx_objects_map = dict()
    for detection_idx, detection_df in unpacked_df.groupby("detection_idx"):
        detection_idx_objects_map.update({detection_idx: list(zip(
            detection_df["detection_object_id"].values,
            detection_df[["detection_object_pos_x", "detection_object_pos_y", "detection_object_pos_z"]].values))})

    # load video frames
    frame_filepaths = sorted(glob.glob(os.path.join(directory_frames, "*.png")))
    frame_times = []
    for frame_filepath in frame_filepaths:
        frame_times.append(int(os.path.splitext(os.path.split(frame_filepath)[-1])[0]) / 1e9)

    # make an object tracker
    my_mot = MultiObjectTracker(tracked_objects_id=[41, 39])
    colors = [(255, 0, 0), (0, 0, 255)]

    # send each detection to the tracker, and annotate frames
    my_video_writer = cv2.VideoWriter(
        filepath_video_annotated, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_size[0], frame_size[1]))
    current_frame_idx = 0
    for detection_idx in detection_indices_np:

        # find out which frames have been added since the last detection
        frame_indices_to_annotate = []
        while frame_times[current_frame_idx] <= detection_times_np[detection_idx]:
            frame_indices_to_annotate.append(current_frame_idx)
            current_frame_idx += 1

        # prepare data for tracking
        detection_objects = detection_idx_objects_map.get(detection_idx, [])
        detection_objects_id = []
        detection_objects_pos_np = []
        for detection_object_id, detection_object_pos_np in detection_objects:
            detection_objects_id.append(detection_object_id)
            detection_objects_pos_np.append(detection_object_pos_np)

        # feed data to tracker
        cameraframe_annotations_pos = my_mot.update(
            camera_poses_pos_np[detection_idx], camera_poses_rot_np[detection_idx],
            detection_times_np[detection_idx], detection_objects_id, detection_objects_pos_np)
        for frame_idx_to_annotate in frame_indices_to_annotate:
            frame_image_np = cv2.imread(frame_filepaths[frame_idx_to_annotate])
            for color, cameraframe_annotation_pos_np in zip(colors, cameraframe_annotations_pos):
                if cameraframe_annotation_pos_np is None:
                    continue
                imageframe_annotation_pos = (intrinsic_matrix @ cameraframe_annotation_pos_np[:, np.newaxis]).flatten()
                y, x = imageframe_annotation_pos[0:2] / imageframe_annotation_pos[2]
                cv2.line(frame_image_np, (208, 208), (int(y), int(x)), color, 2)
            cv2.imwrite(
                os.path.join(directory_annotated, f"{int(frame_times[frame_idx_to_annotate] * 1e9)}.png"),
                frame_image_np)
            my_video_writer.write(frame_image_np)
    my_video_writer.release()


if __name__ == "__main__":
    # inputs
    filepath_combined = r"D:\Documents\Academics\ROB498\Capstone-Prototype\data\combined.csv"
    filepath_unpacked = r"D:\Documents\Academics\ROB498\Capstone-Prototype\data\unpacked.csv"
    directory_frames = r"D:\Documents\Academics\ROB498\Capstone-Prototype\data\frames"

    # outputs
    directory_frames_annotated = r"D:\Documents\Academics\ROB498\Capstone-Prototype\data\frames_annotated"
    filepath_video_annotated = r"D:\Documents\Academics\ROB498\Capstone-Prototype\data\video_annotated.avi"

    main(filepath_combined, filepath_unpacked, directory_frames, directory_frames_annotated, filepath_video_annotated)
