import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings
from face_train import Model

class FaceRecognitionProcessor(VideoProcessorBase):
    def __init__(self, model_path, student_data) -> None:
        super().__init__()
        self.model = Model()
        self.model.load_model(file_path=model_path)
        self.student_data = student_data

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # 进行人脸识别，识别出人脸后查询学生数据，判断是否签到
        # 这里需要你根据你的具体情况来实现
        return img

def main():
    st.title("人脸识别签到")

    st.markdown("### 视频捕获")
    st.markdown("请在下方查看摄像头捕获的视频流，并进行人脸识别签到。")

    model_path = "F:/app_project/web_face_identification/face_identification/face_project/model/face_model_03.keras"
    student_data = {}  # 学生数据，例如 {"张三": "001", "李四": "002"}

    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=lambda: FaceRecognitionProcessor(model_path, student_data),
        client_settings=ClientSettings(
            requesting_video=True,
            webrtc={"media_devices": "video-capture"}
        ),
        async_processing=True
    )

    if webrtc_ctx.video_processor:
        st.write("处理中...")
    else:
        st.write("等待摄像头连接...")

if __name__ == "__main__":
    main()
