import numpy as np
import cv2
import scipy
from scipy.spatial import distance as dist
import serial
arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=.1) 

Known_distance = 30
Known_width = 5.7
thres = 0.5
nms_threshold = 0.2

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 242)
GOLDEN = (32, 218, 165)
LIGHT_BLUE = (255, 9, 2)
PURPLE = (128, 0, 128)
CHOCOLATE = (30, 105, 210)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)

font = cv2.FONT_HERSHEY_PLAIN
fonts = cv2.FONT_HERSHEY_COMPLEX
fonts2 = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
fonts3 = cv2.FONT_HERSHEY_COMPLEX_SMALL
fonts4 = cv2.FONT_HERSHEY_TRIPLEX

cap = cv2.VideoCapture(2)
face_model = cv2.CascadeClassifier("config/haarcascade_frontalface_default.xml")
Distance_level = 0
classNames = []
with open("config/coco.names", "r") as f:
    classNames = f.read().splitlines()
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

weightsPath = "config/frozen_inference_graph.pb"
configPath = "config/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

fourcc = cv2.VideoWriter_fourcc(*"XVID")

face_detector = cv2.CascadeClassifier("config/haarcascade_frontalface_default.xml")

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length


def disparity_map(left_image, right_image):
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(left_gray, right_gray)

    disparity_8bit = cv2.convertScaleAbs(disparity)
    cv2.imshow("Disparity Map", disparity_8bit)

    return disparity


def Distance_finder(Focal_Length, real_face_width, face_width_in_frame, disparity):
    distance = (real_face_width * Focal_Length) / face_width_in_frame
    disparity_distance = disparity * 0.05
    distance = distance - disparity_distance
    return distance[0][0]


def face_data(image, CallOut, Distance_level):
    face_width = 0
    face_x, face_y = 0, 0
    face_center_x = 0
    face_center_y = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for x, y, h, w in faces:
        line_thickness = 2
        LLV = int(h * 0.12)

        cv2.line(image, (x, y + LLV), (x + w, y + LLV), (GREEN), line_thickness)
        cv2.line(image, (x, y + h), (x + w, y + h), (GREEN), line_thickness)
        cv2.line(image, (x, y + LLV), (x, y + LLV + LLV), (GREEN), line_thickness)
        cv2.line(
            image, (x + w, y + LLV), (x + w, y + LLV + LLV), (GREEN), line_thickness
        )
        cv2.line(image, (x, y + h), (x, y + h - LLV), (GREEN), line_thickness)
        cv2.line(image, (x + w, y + h), (x + w, y + h - LLV), (GREEN), line_thickness)

        face_width = w
        face_center = []
        face_center_x = int(w / 2) + x
        face_center_y = int(h / 2) + y
        if Distance_level < 10:
            Distance_level = 10

        if CallOut == True:
            cv2.line(image, (x, y - 11), (x + 180, y - 11), (ORANGE), 28)
            cv2.line(image, (x, y - 11), (x + 180, y - 11), (YELLOW), 20)
            cv2.line(image, (x, y - 11), (x + Distance_level, y - 11), (GREEN), 18)

    return face_width, faces, face_center_x, face_center_y


ref_image = cv2.imread("config/lena.png")

ref_image_face_width, _, _, _ = face_data(ref_image, False, Distance_level)
Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)
print(Focal_length_found)

while True: 
    _, frameIn = cap.read()
    width = frameIn.shape[1]
    frame = frameIn[:, : width // 2]
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)

    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs)) 
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    face_width_in_frame, Faces, FC_X, FC_Y = face_data(frame, True, Distance_level)

    if len(classIds) != 0:
        for i in indices:
            i = i
            box = bbox[i]
            confidence = str(round(confs[i], 2))
            color = Colors[classIds[i] - 1]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
            cv2.putText(
                frame,
                classNames[classIds[i] - 1] + " " + confidence,
                (x + 10, y + 20),
                font,
                1,
                color,
                2,
            )

    for face_x, face_y, face_w, face_h in Faces:
        if face_width_in_frame != 0:
            left_image = frame[:, :320, :]
            right_image = frame[:, 320:, :]
            disparity = disparity_map(left_image, right_image)
            Distance = Distance_finder(
                Focal_length_found, Known_width, face_width_in_frame, disparity
            )
            Distance = round(Distance, 2)
            Distance_level = int(Distance * 2.54)
            cv2.putText(
                frame,
                f"Distance {Distance} Cm",
                (face_x - 6, face_y - 6),
                fonts,
                0.5,
                (BLACK),
                2,
            )

    if cv2.waitKey(1) == ord("q"):
        break

    status, photo = cap.read()
    l = len(bbox)
    frame = cv2.putText(
        frame,
        str(len(bbox)) + " Object",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    stack_x = []
    stack_y = []
    stack_x_print = []
    stack_y_print = []
    global D

    if len(bbox) == 0:
        pass
    else:
        for i in range(0, len(bbox)):
            x1 = bbox[i][0]
            y1 = bbox[i][1]
            x2 = bbox[i][0] + bbox[i][2]
            y2 = bbox[i][1] + bbox[i][3]

            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            stack_x.append(mid_x)
            stack_y.append(mid_y)
            stack_x_print.append(mid_x)
            stack_y_print.append(mid_y)

            frame = cv2.circle(frame, (mid_x, mid_y), 3, [0, 0, 255], -1)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 2)

        if len(bbox) == 2:
            D = int(
                dist.euclidean(
                    (stack_x.pop(), stack_y.pop()), (stack_x.pop(), stack_y.pop())
                )
            )
            frame = cv2.line(
                frame,
                (stack_x_print.pop(), stack_y_print.pop()),
                (stack_x_print.pop(), stack_y_print.pop()),
                [0, 0, 255],
                2,
            )
        else:
            D = 0

        if D < 100 and D != 0:
            num = "s"
            arduino.write(num.encode())
            frame = cv2.putText(
                frame,
                "!!MOVE AWAY!!",
                (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                [0, 0, 255],
                4,
            )
        
        if D >= 100:
            num = "q"
            arduino.write(num.encode())
            pass

        frame = cv2.putText(
            frame,
            str(D / 10) + " cm",
            (300, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Output", frame)
        if cv2.waitKey(100) == 13:
            break

cap.release()
cv2.destroyAllWindows()
