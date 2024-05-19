import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
from AudioAnalyzer import AudioAnalyzer
from DataProcessor import DataProcessor

class BirdAnalyzer:
    threshold = 100  # Class attribute for the threshold value

    def __init__(self, std_sum_array, D_filtered, resampled_max_freqs, resampled_max_freqs2, fps, max_freqs,dic):
        """
        Initialize BirdAnalyzer with necessary variables.

        Parameters:
        - std_sum_array (numpy array): Array representing the sum of pixel intensities.
        - D_filtered (numpy array): Filtered spectrogram data.
        - fps (int): Frames per second.
        - max_freqs (numpy array): Array of maximum frequencies.
        """
        self.std_sum_array = std_sum_array
        self.resampled_max_freqs2 = resampled_max_freqs2
        self.resampled_max_freqs = resampled_max_freqs
        self.fps = fps
        self.time_array = None
        self.time = 0
        self.occurrence_array = None
        self.max_freqs = max_freqs
        self.D_filtered = D_filtered
        self.data_dict = dic
        self.birds_frames_arrays1 = []
        self.birds_frames_arrays2 = []
        self.__cal_time()  # Calculate time related variables upon initialization
        self.__occurrences_per_bird()

    def __cal_time(self):
            """
            Calculate time-related variables based on the FPS and array length.
            """
            self.time = len(self.std_sum_array) / self.fps
            self.time_array = np.linspace(0, self.time, len(self.std_sum_array))
            self.occurrence_array = np.array(self.std_sum_array) > self.threshold

    def __occurrences_per_bird(self):

## This is not a real dict , just for testing
        bird_dict = {
            "1": [(0, self.time-55), (self.time-47,self.time-46), (self.time-45.3, self.time-37.2), (self.time-33.2, self.time-33), (self.time-32.05, self.time-31.835), (self.time-30.85, self.time-30.7), (self.time-30.065, self.time-29.7), (self.time-29.05, self.time-28.855), (self.time-27, self.time-26.835), (self.time-25.2, self.time-25), (self.time-20.795, self.time-20.575), (self.time-19.8,self.time-18), (self.time-14.465, self.time-13.455), (self.time-12, self.time-9), (self.time-5.05, self.time-4.8)],
            "2": [(self.time-53,self.time-50), (self.time-45.8, self.time-45.32), (self.time-37, self.time-34), (self.time-32.255, self.time-32.09), (self.time-31.815, self.time-33.7), (self.time-26.4, self.time-25.3), (self.time-17.3, self.time-17), (self.time-8.5, self.time-7.5), (self.time-1.5, self.time)],
            "3": [(self.time-33.7, self.time-33.25), (self.time-33.8, self.time-32.255), (self.time-31.695, self.time-30.9), (self.time-30.675, self.time-30.075), (self.time-29.65, self.time-29.1), (self.time-28.855, self.time-27.75),  (self.time-26.815, self.time-26.5), (self.time-24.9, self.time-20.8), (self.time-20.5, self.time-20), (self.time-17.9, self.time-17.4), (self.time-16.455, self.time-14.475), (self.time-13.425, self.time-13), (self.time-7.4, self.time-5.1), (self.time-4.7, self.time-4)]
        }


        desired_length2 = self.D_filtered.shape[1]  
        resampling_factor = desired_length2 / len(self.occurrence_array)  
        resampled_occurence_array_array = signal.resample(self.occurrence_array.astype(float), desired_length2)  
        resampled_occurence_array_array = resampled_occurence_array_array > 0.5   
        for key in self.data_dict.keys():
            frames1 = np.zeros(len(self.occurrence_array))
            frames2 = np.zeros(self.D_filtered.shape)
            for value in self.data_dict[key]:
                frames1[value] = self.occurrence_array[value]
            self.birds_frames_arrays1.append(frames1)
        for bird,values in bird_dict.items() :
            for start, end in values:
                frames2[:,int(start*self.D_filtered.shape[1]/self.time):int(end*self.D_filtered.shape[1]/self.time)] = resampled_occurence_array_array[int(start*self.D_filtered.shape[1]/self.time):int(end*self.D_filtered.shape[1]/self.time)]
            self.birds_frames_arrays2.append(frames2)

    def calculate_frequency_statistics(self):
        """
        Calculate frequency statistics for Tweet Frames, Non-Tweet Frames, and Total.
        """
        # Create a DataFrame with condition (tweet or non-tweet) and frequency columns
        data = {'Condition': self.occurrence_array, 'Frequency': self.resampled_max_freqs}
        df = pd.DataFrame(data)

        # Calculate statistics for tweet frames, non-tweet frames, and total frames
        statistics_true = df[df['Condition']].describe()
        statistics_false = df[~df['Condition']].describe()
        statistics_total = df.describe()

        # Concatenate statistics into a single table
        statistics_table = pd.concat([statistics_true, statistics_false, statistics_total], axis=1)
        statistics_table.columns = ['Tweets', 'No Tweets', 'Total']

        print(statistics_table)

    def plot_tweet_occurrences(self):
        """
        Plot the tweet occurrence over time.
        """
        # Plot the number of colored pixels and tweet occurrences over time
        plt.figure(figsize=(10, 4))
        plt.plot(self.time_array, self.std_sum_array)
        plt.xlabel('Time [s]')
        plt.ylabel('Number of colored pixels')
        plt.title('Number of Colored Pixels over time')
        plt.axhline(y=self.threshold, color='r', linestyle='--', label='Tweet Threshold')
        plt.xlim(0, self.time)
        plt.grid(True)
        plt.legend()
        plt.show(block=False)

        plt.figure(figsize=(10, 4))
        plt.plot(self.time_array, self.occurrence_array)
        plt.xlabel('Time [s]')
        plt.ylabel('Tweets Detection Over Time')
        plt.title('Tweet occurrences')
        plt.xlim(0, self.time)
        plt.yticks([0, 1])
        plt.grid(True)
        plt.show()

    def plot_tweets_per_bird(self):
        """
        Plot tweet occurrences for each bird.

        Parameters:
        - dic (dict): Dictionary containing bird-specific data.
        """
        
        plt.figure(figsize=(10, 6))

        # Plot tweet occurrences for each bird
        index = 1
        for bird in self.data_dict.keys():
            tweet_occurrences = [0] * len(self.time_array)
            for frame in self.data_dict[bird]:
                tweet_occurrences[frame] = 1

            plt.subplot(len(self.data_dict), 1, index)
            plt.plot(self.time_array, tweet_occurrences)
            plt.xlabel('Time [s]')
            plt.ylabel('Tweets')
            plt.title(f'Bird {index} - Tweet Occurrence')
            plt.yticks([0, 1])
            plt.grid(True)
            index += 1

        plt.tight_layout()
        plt.show()
   

    def plot_mean_signal(self):
        """
        Plot the mean signal of detected tweets.
        """
        plt.figure()
        for index, value in enumerate(self.birds_frames_arrays2):
            max_magnitude_index = np.argmax(np.mean(self.D_filtered[:,value[1].astype(bool)], axis=1))
            frequency_highest_magnitude = self.resampled_max_freqs2[max_magnitude_index]
            highest_magnitude = np.mean(self.D_filtered[:,value[1].astype(bool)], axis=1)[max_magnitude_index]
            plt.subplot(3, 1, index+1)
            plt.plot(self.resampled_max_freqs2, np.mean(self.D_filtered[:,value[1].astype(bool)], axis=1))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Magnitude [dB]')
            plt.title(f'Bird {index+1} - Mean Signal')
            plt.grid(True)
            print(f"The Highest Frequency for Bird {index+1} is {frequency_highest_magnitude} [Hz] with Magnitude of {highest_magnitude} [dB] ")
        plt.tight_layout()
        plt.show()

       
# # # Use call_indices for further calculations
        #     max_magnitude_index = np.argmax(np.mean(self.D_filtered[:, call_indices], axis=0))
        #     max_magnitude_index = np.max(self.D_filtered[:, np.array(value).astype(bool)], axis=0)
        #     frequency_highest_magnitude = self.resampled_max_freqs2[max_magnitude_index]
        #     highest_magnitude = np.mean(self.D_filtered[:,value[1].astype(bool)], axis=1)[max_magnitude_index]

        #     # Plotting mean signal of detected tweets
        #     plt.subplot(3, 1, index+1)
        #     plt.plot(self.resampled_max_freqs2, np.mean(self.D_filtered[:,value[1].astype(bool)], axis=1))
        #     plt.xlabel('Frequency [Hz]')
        #     plt.ylabel('Magnitude [dB]')
        #     plt.title(f'Bird {index+1} - Mean Signal')
        #     plt.grid(True)
        #     print(f"The Highest Frequency for Bird {index+1} is {frequency_highest_magnitude} [Hz] with Magnitude of {highest_magnitude} [dB] ")
        # plt.tight_layout()
        # plt.show(block=False)

    def plot_tweet_frequencies(self):
        """
        Plot the tweet frequencies for each bird.
        """
        colors = ['red', 'blue', 'orange']

        # Plot tweet frequencies for each bird
        for index, values in enumerate(self.birds_frames_arrays1):
            freqs_copy = np.copy(self.resampled_max_freqs)
            freqs_copy[~values.astype(bool)] = 0
            plt.plot(self.time_array, freqs_copy, color=colors[index], label=f"Bird {index+1}")

        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.title('Bird Tweets Frequencies')
        plt.xlim(0, self.time_array[-1])
        plt.grid(True)
        plt.legend()
        plt.show()