
import os

from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr
import numpy as np
import cv2


def main(filepath_bag, filepath_video, directory_frames):

    # =========================================================================
    # GET VIDEO FRAMES AND VIDEO
    # =========================================================================

    topic_name = '/yolov4_publisher/color/image'
    topic_name_calib = '/yolov4_publisher/color/camera_info'
    frame_size = (416, 416, 3)
    fps = 8.25

    # create reader instance
    with Reader(filepath_bag) as reader:

        # # inspect topics
        # for connection in reader.connections.values():
        #     print("Connection topic: ", connection.topic)
        #     print("Connection message type: ", connection.msgtype)
        #     print("")
        # raise Exception

        # # get fps
        # n_frames = 0
        # timestamps = []
        # for connection, timestamp, rawdata in reader.messages():
        #     if connection.topic == topic_name:
        #         n_frames += 1
        #         timestamps.append(timestamp)
        # print(n_frames / ((timestamps[-1] - timestamps[0]) / 1e9))
        # raise Exception

        # # get calibration
        # for connection, timestamp, rawdata in reader.messages():
        #     if connection.topic == topic_name_calib:
        #         msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
        #         print(msg.k.reshape((3, 3)))
        #         raise Exception
        # # [[600.48120117   0.         200.15670776]
        # #  [  0.         599.72149658 214.27590942]
        # #  [  0.           0.           1.        ]]

        # extract the video and frames
        my_video_writer = cv2.VideoWriter(
            filepath_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_size[0], frame_size[1]))
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == topic_name:
                msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                frame_np = np.array(msg.data).reshape(frame_size)
                cv2.imwrite(os.path.join(directory_frames, f"{timestamp}.png"), frame_np)
                my_video_writer.write(frame_np)
        my_video_writer.release()


if __name__ == "__main__":

    # input
    filepath_bag = r"D:\Documents\Academics\ROB498\Capstone-Prototype\data\test1.bag"

    # output
    filepath_video = r"D:\Documents\Academics\ROB498\Capstone-Prototype\data\video.avi"
    directory_frames = r"D:\Documents\Academics\ROB498\Capstone-Prototype\data\frames"

    main(filepath_bag, filepath_video, directory_frames)