import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings

class FaceCaptureProcessor(VideoProcessorBase):
    def __init__(self, path_name, top_num, sign="frontalface") -> None:
        super().__init__()
        self.path_name = path_name
        self.top_num = top_num
        self.sign = sign
        self.classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.num_faces = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if self.num_faces < self.top_num:
                self.save_face(img, x, y, w, h)
                self.num_faces += 1
        return img

    def save_face(self, frame, x, y, w, h):
        img_name = f'{self.path_name}/{self.sign}_{self.num_faces}.jpg'
        face_image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
        cv2.imwrite(img_name, face_image)

def main():
    st.title("人脸信息录入")

    st.markdown("### 视频捕获")
    st.markdown("请在下方查看摄像头捕获的视频流，并进行人脸信息录入。")

    path_name = "./data/2021302021175"
    top_num = 50
    sign = "frontalface"

    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=lambda: FaceCaptureProcessor(path_name, top_num, sign),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    if webrtc_ctx.video_processor:
        st.write("处理中...")
    else:
        st.write("等待摄像头连接...")

if __name__ == "__main__":
    main()
