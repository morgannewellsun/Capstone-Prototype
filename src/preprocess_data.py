
import numpy as np
import pandas as pd


def main(filepath_slam, filepath_detection, filepath_combined, filepath_unpacked):

    # load data
    slam_df = pd.read_csv(filepath_slam)
    slam_times_np = slam_df["Time"].values
    slam_seq_np = slam_df["header.seq"].values
    slam_poses_np = slam_df[[
        "pose.position.x", "pose.position.y", "pose.position.z",
        "pose.orientation.x", "pose.orientation.y", "pose.orientation.z", "pose.orientation.w"]].values
    detection_df = pd.read_csv(filepath_detection)
    detection_times_np = detection_df["Time"].values
    detection_seq_np = detection_df["header.seq"].values
    detection_strings_np = detection_df["detections"].values

    # compute deltatime for both files
    slam_deltatimes_np = slam_times_np[1:] - slam_times_np[:-1]
    print(f"[INFO] SLAM deltatimes:      "
          f"mean={np.mean(slam_deltatimes_np)}, stdev={np.std(slam_deltatimes_np)}")
    detection_deltatimes_np = detection_times_np[1:] - detection_times_np[:-1]
    print(f"[INFO] Detection deltatimes: "
          f"mean={np.mean(detection_deltatimes_np)}, stdev={np.std(detection_deltatimes_np)}")

    # first timestamp for both files
    print(f"[INFO] First SLAM timestamp:      {slam_times_np[0]}")
    print(f"[INFO] First detection timestamp: {detection_times_np[0]}")

    # SLAM runs faster
    # first SLAM timestep is earlier than first detection timestamp

    # find most recent SLAM timestamp prior to each detection timestamp
    # this simulates a system where the most recent timestep for both SLAM and detection are buffered
    slam_idx = 0
    matched_slam_indices = []
    for detection_time in detection_times_np:
        while slam_times_np[slam_idx] <= detection_time:
            slam_idx += 1
        slam_idx -= 1
        matched_slam_indices.append(slam_idx)
    matched_slam_indices_np = np.array(matched_slam_indices)

    # combine dataframes
    combined_df = pd.DataFrame({
        "slam_time": slam_times_np[matched_slam_indices_np],
        "slam_seq": slam_seq_np[matched_slam_indices_np],
        "slam_pose_pos_x": slam_poses_np[matched_slam_indices_np, 0],
        "slam_pose_pos_y": slam_poses_np[matched_slam_indices_np, 1],
        "slam_pose_pos_z": slam_poses_np[matched_slam_indices_np, 2],
        "slam_pose_rot_x": slam_poses_np[matched_slam_indices_np, 3],
        "slam_pose_rot_y": slam_poses_np[matched_slam_indices_np, 4],
        "slam_pose_rot_z": slam_poses_np[matched_slam_indices_np, 5],
        "slam_pose_rot_w": slam_poses_np[matched_slam_indices_np, 6],
        "detection_idx": range(len(detection_times_np)),
        "detection_time": detection_times_np,
        "detection_seq": detection_seq_np,
    })
    combined_df.to_csv(filepath_combined, index=False)

    # unpack detected objects
    detection_indices = []
    detection_objects_id = []
    detection_objects_pos_x = []
    detection_objects_pos_y = []
    detection_objects_pos_z = []
    for detection_idx, detection_str in enumerate(detection_strings_np):
        if detection_str == "[]":
            continue
        parts = detection_str.split("\n")
        parser_state = "id"  # {"id", "position", "x", "y", "z"}
        for part in parts:
            if (parser_state == "id") and ("id:" in part) and ("_id:" not in part):
                object_id = int(part.split(" ")[-1])
                detection_indices.append(detection_idx)
                detection_objects_id.append(object_id)
                parser_state = "position"
            elif (parser_state == "position") and ("position:" in part):
                parser_state = "x"
            elif (parser_state == "x") and ("x:" in part):
                object_pos_x = float(part.split(" ")[-1])
                detection_objects_pos_x.append(object_pos_x)
                parser_state = "y"
            elif (parser_state == "y") and ("y:" in part):
                object_pos_y = float(part.split(" ")[-1]) * -1  # for consistency with SLAM frames
                detection_objects_pos_y.append(object_pos_y)
                parser_state = "z"
            elif (parser_state == "z") and ("z:" in part):
                object_pos_z = float(part.split(" ")[-1])
                detection_objects_pos_z.append(object_pos_z)
                parser_state = "id"
            else:
                continue
    unpacked_df = pd.DataFrame({
        "detection_idx": detection_indices,
        "detection_object_id": detection_objects_id,
        "detection_object_pos_x": detection_objects_pos_x,
        "detection_object_pos_y": detection_objects_pos_y,
        "detection_object_pos_z": detection_objects_pos_z,
    })
    unpacked_df.to_csv(filepath_unpacked, index=False)


if __name__ == "__main__":

    # inputs
    filepath_slam = r"D:\Documents\Academics\ROB498\Capstone-Prototype\data\slam.csv"
    filepath_detection = r"D:\Documents\Academics\ROB498\Capstone-Prototype\data\detections.csv"

    # outputs
    filepath_combined = r"D:\Documents\Academics\ROB498\Capstone-Prototype\data\combined.csv"
    filepath_unpacked = r"D:\Documents\Academics\ROB498\Capstone-Prototype\data\unpacked.csv"

    main(filepath_slam, filepath_detection, filepath_combined, filepath_unpacked)
