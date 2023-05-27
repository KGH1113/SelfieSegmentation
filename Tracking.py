import cv2
import mediapipe as mp
import numpy as np

class Tracking:
    def __init__(self, mp_drawing, mp_selfie_segmentation, selfie_segmentation):
        self.mp_drawing = mp_drawing
        self.mp_selfie_segmentation = mp_selfie_segmentation
        self.selfie_segmentation = selfie_segmentation
    
    def process(self, image):
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = self.selfie_segmentation.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return (image, results)
    
    def segmentate(self, image, results, BG_COLOR, bg_image):
        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack(
        (results.segmentation_mask,) * 3, axis=-1) > 0.1
        # The background can be customized.
        #   a) Load an image (with the same width and height of the input image) to
        #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
        #   b) Blur the input image by applying image filtering, e.g.,
        #      bg_image = cv2.GaussianBlur(image,(55,55),0)

        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
        output_image = np.where(condition, image, bg_image)

        return output_image

# def select_bg_image():
#     while True:
#         input_num = input("Select the background: 1. color 2. blur 3. image input: ")
#         if input_num == 1:
#             bg_image = None
#         elif input_num == 2:
#             bg_image = "blur"
#         elif input_num == 3:
#             bg_image_path = input("Please write the image path: ")
#             bg_image = cv2.imread(bg_image_path)
#         else:
#             continue
#         break

#     return bg_image

# if __name__ == '__main__':
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# For webcam input:
BG_COLOR = (0, 0, 0) # gray

bg_image = None

cap = cv2.VideoCapture(0)
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    tracking = Tracking(mp_drawing, mp_selfie_segmentation, selfie_segmentation)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        image, results = tracking.process(image)

        output_image = tracking.segmentate(image, results, BG_COLOR, bg_image)

        cv2.imshow('MediaPipe Selfie Segmentation', output_image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
