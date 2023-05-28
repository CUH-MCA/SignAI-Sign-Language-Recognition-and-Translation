import csv
import copy
import argparse
import itertools
import sys
import cv2 as cv
import numpy as np
import mediapipe as mp
import argostranslate.package
import argostranslate.translate
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
mp_drawing = mp.solutions.drawing_utils
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import time
from gingerit.gingerit import GingerIt



expression = ' '
stored_inputs = []

from_code = "en"
to_code = "hi"
# Download and install Argos Translate package


print('checking network ...')
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(
        lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
    )
)
argostranslate.package.install_from_path(package_to_install.download())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.4)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    args = parser.parse_args()
    return args

def main():
    try:

        def cls():
            global expression
            expression = ' '
            e.delete(0, "end")

        def enter_text(x):
            global expression
            e.delete(0, "end")
            expression = expression + str(x)
            e.get()
            e.insert(0, expression)
        

        def add_input(input_text):
            if input_text:
                stored_inputs.append(input_text)
                update_text()

        def update_text():
            text.delete(1.0, END)
            for input_text in stored_inputs:
                text.insert(END, input_text + '\n')

        def cls_stored():
            global stored_inputs
            stored_inputs = []
            add_input("Predictions :")
    

        args = get_args()
        cap_device = args.device
        cap_width = args.width
        cap_height = args.height
        use_static_image_mode = args.use_static_image_mode
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence

        
        root = Tk()
        root.title("Sign.AI")
        root.geometry("1200x600")
        photo = PhotoImage(file = "images\logo.png")
        root.iconphoto(False,photo)
        root.config(bg="#1C1C1C")

        Label(root,text="Sign Language Detection", font = ("times new roman",30,"bold"), bg ="#1C1C1C",fg="#78F093").grid(row = 0,column = 0, columnspan=5)
        f1 = LabelFrame(root,bg = "#DEDEDE")
        f1.grid(row = 1, column = 1, columnspan = 3)
        l1 = Label(f1,bg = "#1C1C1C")
        l1.grid(row = 1, column = 1, columnspan= 3)
        e = Entry(root, width=53,fg = "#DEDEDE", font = ("Courier",18,"bold"), bg = "#1C1C1C", background="#1C1C1C")
        e.grid(row = 2, column = 1)
        button_clear = Button(root, text='Clear', height=1, width=20, bd=3, command=cls)
        button_clear.grid(row = 3, column = 1, columnspan= 3)



        frame = Frame(root, bg="#282c34")
        frame.grid(row=1, column=0, padx=10, pady=10)


        text = Text(frame, wrap="word", bg="#282c34", fg="white", font=('times new roman', 14))
        text.config(spacing1=10)
        text.config(spacing2=10)
        text.config(spacing3=10)
        text.grid(row=0, column=0, sticky="nsew")

        frame.columnconfigure(0,weight=1)
        frame.columnconfigure(1,weight=6)

        root.grid_rowconfigure(1, weight=4)
        root.grid_columnconfigure(0, weight=6)

        image = Image.open("images/signstpw.png")
        image = image.resize((320, 540))
        image = image.convert("RGBA")
        photo = ImageTk.PhotoImage(image)
        label = Label(root, image=photo, bg="#1C1C1C")
        label.grid(row = 1, column = 4, rowspan=3)

        add_input("Predictions : ")
        button_clear_predictions = Button(root, text='Clear History', height=1, width=20, bd=3, command = cls_stored)
        button_clear_predictions.grid(row = 2, column = 0)

        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        keypoint_classifier = KeyPointClassifier()

        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]

        mode = 0
        times_captured = 0
        prevchar=' '
        count=0
        spcount=0


        while True:
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break

            number, mode = select_mode(key, mode)

            ret, image = cap.read()
            if not ret:
                break

            image = cv.flip(image, 1)
            debug_image = copy.deepcopy(image)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks is not None:
                no_of_hands = len(results.multi_hand_landmarks)
                pre_processed_landmark_list = []
                if no_of_hands == 2:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                        pre_processed_landmark_list = pre_processed_landmark_list + pre_process_landmark(landmark_list)
                        mp_drawing.draw_landmarks(image=debug_image, landmark_list=hand_landmarks,
                                                connections=mp_hands.HAND_CONNECTIONS)
                elif no_of_hands == 1:
                    pre_processed_landmark_list = [0.0]*42
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                        pre_processed_landmark_list = pre_processed_landmark_list + pre_process_landmark(landmark_list)
                        mp_drawing.draw_landmarks(image=debug_image, landmark_list=hand_landmarks,
                                                connections=mp_hands.HAND_CONNECTIONS)
                else:
                    pre_processed_landmark_list = [0.0]*84

                times_captured = logging_csv(number, mode, pre_processed_landmark_list, times_captured)
                hand_sign_id, confidence = keypoint_classifier(pre_processed_landmark_list)

                sign = keypoint_classifier_labels[hand_sign_id] 
                
                if confidence >=0.85 :#0.996:
                    spcount=0
                    if prevchar == sign:
                        count = count+1
                    else:
                        prevchar=sign
                        count=0
                    if  count>=10 and expression[-1] !=  sign:
                        prevchar=' '
                        count=0
                        if sign == "fullstop":
                            var = expression
                            parser = GingerIt()
                            res = parser.parse(var)
                            result = res['result']
                            translatedText = argostranslate.translate.translate(result, from_code, to_code)
                            add_input(f'{len(stored_inputs)} : {result}\nHindi : {translatedText}')
                            cls()
                            sign = ""
                        enter_text(sign)

                debug_image = draw_info_text(
                
                    debug_image,
                    cap_width,
                    confidence,
                    keypoint_classifier_labels[hand_sign_id]                
                )
            else:
                spcount =spcount+1
                if spcount>=10 and expression[-1] !=  " ":
                        enter_text(" ")
                        spcount=0

            debug_image = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)  
            debug_image= draw_info(debug_image, mode, number, times_captured)
            img = ImageTk.PhotoImage(Image.fromarray(debug_image))
            l1['image'] = img
            root.update()

            # cv.imshow('Hand Gesture Recognition', debug_image)

        cap.release()
        cv.destroyAllWindows()
    except:
        class DevNull:
            def write(self, msg):
                pass

        sys.stderr = DevNull()




def select_mode(key, mode):
    number = 0
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        data = list(reader)
        number = len(data)-1
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
        sign_name = input("enter sign name:")
        sign_name = sign_name
        sign_name = [sign_name]
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', mode="a", newline="", encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(sign_name)
    return number, mode


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list


def logging_csv(number, mode, landmark_list, times_captured):
    if mode == 0:
        times_captured = 0
    if mode == 1:
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
        times_captured = times_captured+1
    return times_captured


def draw_info_text(image, img_width, confidence, hand_sign_text):
    info_text = ""
    if hand_sign_text != "":
        info_text = info_text + hand_sign_text
    if confidence >= 0.7:
        cv.putText(image, info_text, (int(img_width/2 + 10), 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

    return image


def draw_info(image, mode, number, times_captured):
    if mode == 1:
        cv.putText(image, "Reading Your Hand...", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(image, "Label:" + str(number), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(image, "Captured:" + str(times_captured), (10, 75), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv.LINE_AA)
    if mode == 0:
        cv.putText(image, "Predicting...", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
