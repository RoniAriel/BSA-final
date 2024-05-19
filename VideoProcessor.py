import cv2
import numpy as np


class VideoProcessor:
    def __init__(self, file, output_video):
        """
              Parameters:
              - input_video (str): Path to the input video file.
              - output_video (str): Path to save the processed output video file.
              """
        self.webm_file = f'{file}.webm'
        self.cap = cv2.VideoCapture(self.webm_file) # Open the input video file for reading
        self.output_video = output_video
        self.out = None  # Initialize the VideoWriter object
        self.fps = None
        self.std_sum_array = []
    def process_video(self):
        """
                Process the input video by cropping frames, detecting motion, and saving the processed video.
                """
        if not self.cap.isOpened():
            print("Error: Could not open the video file.")
            return

        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Define codec and VideoWriter object for saving the processed video
        codec = cv2.VideoWriter_fourcc(*'VP80')
        self.out = cv2.VideoWriter(self.output_video, codec, self.fps, (370, 280))

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break
            # Crop the frame to the desired region of interest
            cropped_frame = frame[0:280, 10:380]

            # Calculate the standard deviation of pixel values across color channels
            std_dev = np.std(cropped_frame, axis=2)

            # Create a mask based on the standard deviation threshold
            mask = (std_dev > 5).astype(np.uint8)

            # Apply a color (red) to regions with high standard deviation
            cropped_frame[std_dev > 5] = [0, 0, 255]
            self.std_sum_array.append(np.sum(std_dev > 5))

            # Write the processed frame to the output video
            self.out.write(cropped_frame)

        self.cap.release()
        self.out.release()
        return self.output_video, self.std_sum_array

