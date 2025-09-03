import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


def detect_player_gesture(cap) -> str:
    # Gesture recognizer
    base_options = BaseOptions(model_asset_path="gesture_recognizer.task")
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # Aliases
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame.flags.writeable = False
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            recognized_gesture = recognizer.recognize(mp_img)

            if recognized_gesture.gestures:
                gesture_label = recognized_gesture.gestures[0][0].category_name
                print(gesture_label)
                return gesture_label

            # if results.multi_hand_landmarks:
            #     for hand_landmarks in results.multi_hand_landmarks:
            #         mp_drawing.draw_landmarks(
            #             frame,
            #             hand_landmarks,
            #             mp_hands.HAND_CONNECTIONS,
            #             mp_drawing_styles.get_default_hand_landmarks_style(),
            #             mp_drawing_styles.get_default_hand_connections_style(),
            #         )

            # cv2.imshow("MediaPipe Hands", cv2.flip(frame, 1))
            # if cv2.waitKey(5) & 0xFF == 27:
            #     break

    # cap.release()
    # cv2.destroyAllWindows()


def pc_gesture():
    cap = cv2.VideoCapture(0)
    window_name = "Rock, paper, scissors"
    # TODO: Window is not being resized at all
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 500, 200)
    cv2.resizeWindow(window_name, 300, 300)
    while True:
        player_move = detect_player_gesture(cap)
        if player_move == "None":
            continue
        else:
            img_path = f"assets/{player_move}.png"
            print(img_path)
            img = cv2.imread(img_path)

            cv2.imshow(window_name, img)
            cv2.waitKey(1)


if __name__ == "__main__":
    # cap = cv2.VideoCapture(0)
    # detect_player_gesture(cap)
    pc_gesture()

    cap.release()
    cv2.destroyAllWindows()
