import streamlit as st
import cv2 as cv
import numpy as np
from collections import defaultdict
import tempfile
from ultralytics import YOLO


def annotate(results, track_history, default_size):
    # Get the boxes and track IDs
    boxes = results[0].obb.xywhr.cpu()
    try:
        track_ids = results[0].obb.id.int().cpu().tolist()
    except Exception as e:
        print(f"Error encountered: {e}")  # 可选：打印错误信息以便调试
        track_ids = []
        boxes = []
    try:
        # Visualize the results on the frame
        annotated_frame = results[0].plot(line_width=2, conf=False, labels=True)
    except Exception as e:
        print(f"Error encountered while plotting results: {e}")
        # Create a default black frame with the specified size
        height, width = default_size
        annotated_frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Visualize the results on the frame
    # annotated_frame = results[0].plot(line_width = 2,conf = False)
    # frame_list.append(annotated_frame)
    # Plot the tracks
    if len(track_ids) != 0:
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h, r = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 90:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv.polylines(annotated_frame, [points], isClosed=False, color=(230, 0, 230), thickness=4)
    else:
        pass
    return annotated_frame

# Streamlit 应用
def main():
    st.title("视频识别与追踪")

    # 文件选择框
    uploaded_file = st.file_uploader("选择视频文件", type=["mp4", "avi"])

    if uploaded_file is not None:
        # 保存上传的文件到临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        temp_file.close()

        # 设置模型路径和保存路径

        # 设置模型路径
        model_path = st.secrets["model_path"]
        model = YOLO(model_path)

        # 显示处理后的视频
        cap = cv.VideoCapture(temp_file.name)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv.CAP_PROP_FPS)
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        track_history = defaultdict(lambda: [])
        frame_count = 0

        st.write("正在处理视频...")
        progress_bar = st.progress(0)

        # 创建一个可更新的占位符
        image_placeholder = st.empty()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Run YOLO11 tracking on the frame
            results = model.track(frame, persist=True)
            annotated_frame = annotate(results, track_history, (width, height))

            # 更新图片
            image_placeholder.image(annotated_frame, channels="BGR", caption=f"Frame {frame_count}")

            frame_count += 1
            progress_bar.progress(frame_count / total_frames)

        cap.release()
        st.success("视频处理完成！")

if __name__ == "__main__":
    main()
