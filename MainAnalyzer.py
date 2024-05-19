from AudioAnalyzer import AudioAnalyzer
from DataProcessor import DataProcessor
from VideoProcessor import VideoProcessor
from BirdAnalyzer import BirdAnalyzer
import pandas as pd

class MainAnalyzer:
    def __init__(self, file, output_video_file):
        self.file = file
        self.std_sum_array = []
        self.output_video_file = output_video_file
        self.resampled_max_freqs = None
        self.resampled_max_freqs2 = None
        self.data_dict= None

    def main(self):
        # Process video and analyze audio
        video = VideoProcessor(self.file, self.output_video_file)
        audio = AudioAnalyzer(self.file, video.std_sum_array)
        self.load_data()
        self.output_video, self.std_sum_array = video.process_video()
        audio.plot_waveform()
        audio.analyze_spectrogram()
    
        audio.plot_spectrogram()

        self.resampled_max_freqs, self.resampled_max_freqs2 = audio.resample_freqs()
        birds_calls = BirdAnalyzer(self.std_sum_array, audio.D_filtered, self.resampled_max_freqs,self.resampled_max_freqs2,  video.fps, audio.max_freqs, self.data_dict)
  
        birds_calls.plot_tweet_occurrences()
  
        print(birds_calls.calculate_frequency_statistics())
   
        birds_calls.plot_tweets_per_bird()
  
        birds_calls.plot_mean_signal()
   
        birds_calls.plot_tweet_frequencies()
  

    def load_data(self):
        csv_file = 'call_points.csv'
        processor = DataProcessor(csv_file)
        try:
            processor.load_data()
            processor.process_data()
            self.data_dict = processor.get_data_dict()
        except FileNotFoundError as e:
            print("Error: File not found -", e)
        except ValueError as e:
            print("Error: Value error -", e)

def main():
    file = "2023-08-30_08-06-50pol"
    analyzer = MainAnalyzer(file, "output_video1.webm")  # Pass the output video file as an argument
    analyzer.main()  # Call the main method of MainAnalyzer

if __name__ == "__main__":
    main()
