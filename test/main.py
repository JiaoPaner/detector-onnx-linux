import cv2
import ctypes
import base64
from PIL import Image
from io import BytesIO
import json
from datetime import datetime

def frame2base64(frame):
    img = Image.fromarray(frame)  # 将每一帧转为Image
    output_buffer = BytesIO()  # 创建一个BytesIO
    img.save(output_buffer, format='JPEG')  # 写入output_buffer
    byte_data = output_buffer.getvalue()  # 在内存中读取
    base64_data = base64.b64encode(byte_data)  # 转为BASE64
    output_buffer.close()
    return base64_data


if __name__ == '__main__':
    so = ctypes.cdll.LoadLibrary("/home/jiaopan/projects/c++/detector-onnx-linux/cmake-build-debug/libdetector.so")
    model_path = bytes("/home/jiaopan/projects/c++/detector-onnx-linux/model/yolov5s.onnx", "utf-8")
    status = so.init(model_path, 1)
    print(status)
    image = bytes("/home/jiaopan/Downloads/bus.jpg", "utf-8")
    detectFile = so.detectByFile
    detectFile.restype = ctypes.c_char_p
    result = detectFile(image, ctypes.c_float(0.5))
    result = ctypes.string_at(result, -1).decode("utf-8")
    print(result)

    cap = cv2.VideoCapture("/home/jiaopan/Downloads/jd2.mp4")
    index = 0;
    detect = so.detectByBase64
    detect.restype = ctypes.c_char_p
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            #frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA)
            if index % 1 == 0:
                start = datetime.now()
                #cv2.imwrite("frame.jpg",frame)
                #result = detectFile(bytes("frame.jpg","utf-8"), ctypes.c_float(0.6))
                image = frame2base64(frame)
                result = detect(image, ctypes.c_float(0.4))
                end = datetime.now()
                print("time cost:",(end-start).total_seconds())
                result = ctypes.string_at(result, -1).decode("utf-8")
                result = json.loads(result)
                data = result["data"]
                print(data)
                for box in data:
                    x = int(box["location"]["x"])
                    y = int(box["location"]["y"])
                    width = int(box["location"]["width"])
                    height = int(box["location"]["height"])
                    cv2.putText(frame, box["label"], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.rectangle(frame,(x,y),(x+width,y+height),(255,0,0),2)
            cv2.imshow('image', frame)
            k = cv2.waitKey(20)
            index += 1
            # q键退出
            if (k & 0xff == ord('q')):
                break

    cap.release()
    cv2.destroyAllWindows()

