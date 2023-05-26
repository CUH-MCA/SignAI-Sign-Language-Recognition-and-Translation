import csv
import copy
import argparse
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
mp_drawing = mp.solutions.drawing_utils


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
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

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
            debug_image = draw_info_text(
                debug_image,
                cap_width,
                confidence,
                keypoint_classifier_labels[hand_sign_id]
            )

        debug_image = draw_info(debug_image, mode, number, times_captured)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


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
    if confidence >= 0.3:
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
