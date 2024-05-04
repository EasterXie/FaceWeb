"""
录入人脸
"""
import cv2
from load_data import resize_image
import os
import shutil
import streamlit as st


def CatchUsbVideo(window_name, camera_id, path_name, top_num, classer="F:/app_project/web_face_identification/face_identification/face_project/haarcascade_frontalface_alt2.xml", sign="frontalface"):
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(camera_id)
    # 准备人脸识别分类器
    classfier = cv2.CascadeClassifier(classer)
    # 识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)
    num = 0
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            print("摄像头堵塞,结束进程")
            break
            # 将当前帧转换成灰度图像
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) == 1:  # 检测到一张人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect

                img_name = f'{path_name}/{sign}_{num}.jpg'
                print(img_name)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                image = resize_image(image)
                issaved = cv2.imwrite(img_name, image)
                print(f"保存成功? == {issaved}")

                num += 1
                if num > top_num:  # 如果超过指定最大保存数量退出循环
                    print("收集数量达标,采集结束")
                    break

                # 画出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                # 显示当前捕捉到了多少人脸图片了
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f'{num}', (x + 30, y + 30), font, 1, (255, 0, 255), 4)

                # 超过指定最大保存数量结束程序
        if num > top_num:
            break

        # 显示图像
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

            # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

def create_folder(directory, folder_name):
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 拼接文件夹路径
    folder_path = f"{directory}/{folder_name}"
    
    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        st.success(f"文件夹 '{folder_name}' 已成功创建在 '{directory}' 下。")
    else:
        st.warning(f"文件夹 '{folder_name}' 在 '{directory}' 下已存在。")



def delete_folder(folder_path):
    try:
        # 递归删除文件夹及其所有内容
        shutil.rmtree(folder_path)
        st.success(f"文件夹 '{folder_path}' 及其内容已成功删除。")
    except OSError as e:
        st.warning(f"删除文件夹 '{folder_path}' 及其内容时出错: {e}")


if __name__ == '__main__':
    # #  调用函数收集正脸人脸
    CatchUsbVideo("识别人脸区域", 0, "F:/app_project/web_face_identification/face_identification/face_project/data/2021302021175", 50)
    # 调用函数收集测脸(左)人脸
    # CatchUsbVideo("识别人脸区域", 0, "F:/app_project/web_face_identification/face_identification/face_project/data/2021302021175", 10, classer="F:/app_project/web_face_identification/face_identification/face_project/haarcascade_profileface.xml", sign="profileface")
    # delete_folder("F:/app_project/web_face_identification/face_identification/face_project/data/2021302021175")