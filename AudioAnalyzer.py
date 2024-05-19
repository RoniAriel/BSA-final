import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

class AudioAnalyzer:
    def __init__(self, file, std_sum_array):
        self.file_path = f"{file}.wav"  # File path of the audio file
        self.y, self.sr = librosa.load(self.file_path)  # Load audio data and sample rate
        self.freqs = None  # Initialize frequency array
        self.D = None  # Initialize spectrogram
        self.D_filtered = None  # Initialize filtered spectrogram
        self.max_freq_indices = None  # Initialize array for indices of maximum frequencies
        self.max_freqs = None  # Initialize array for maximum frequencies
        self.resampled_max_freqs = None  
        self.resampled_max_freqs2 = None  
        self.std_sum_array = std_sum_array  # Standard sum array used for resampling

    def plot_waveform(self):
        """
        Plot the audio waveform.
        """
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(self.y, sr=self.sr)
        plt.title('Audio Sound Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.show()

    def plot_spectrogram(self):
        """
        Plot the spectrogram of the audio.
        """
        if self.D is not None:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(self.D, sr=self.sr, x_axis='time', y_axis='linear')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            plt.show()
        else:
            print("Spectrogram data has not been calculated yet. Call analyze_spectrogram first.")

    def analyze_spectrogram(self, freq_threshold=1000):
        """
        Analyze the spectrogram of the audio.
        """
        self.D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref=np.max)  # Compute spectrogram
        self.freqs = librosa.core.fft_frequencies(sr=self.sr)  # Get frequency array
        idx_low = np.argmax(self.freqs > freq_threshold) - 1  # Find index for frequency threshold
        D_filtered = np.copy(self.D)  
        D_filtered[:idx_low, :] = -80  # Apply frequency filtering
        self.D_filtered = D_filtered

        # Derive the maximum frequencies associated with each frame in the spectrogram.
        self.max_freq_indices = np.argmax(self.D_filtered, axis=0)
        self.max_freqs = self.freqs[self.max_freq_indices]
        

    def resample_freqs(self):
        """
        Resample the frequencies array to match the size of the number of frames and Spectrogram's Shape.

        Returns:
        - resampled_max_freqs (numpy array): Resampled maximum frequencies array.
        - resampled_max_freqs2 (numpy array): Second resampled maximum frequencies array.
        """
        desired_length = len(self.std_sum_array)  # desired length for resampling
        indices1 = np.arange(len(self.max_freqs))  # Create indices for original maximum frequencies
        indices2 = np.arange(len(self.freqs))  # Create indices for original frequencies

        # Perform linear interpolation for resampling
        interp_func1 = interp1d(indices1, self.max_freqs, kind='linear')
        interp_func2 = interp1d(indices2, self.freqs, kind='linear')

        # Resample the maximum frequencies arrays
        resampled_max_freqs = interp_func1(np.linspace(0, len(self.max_freqs) - 1, desired_length))
        resampled_max_freqs2 = interp_func2(np.linspace(0, len(self.freqs) - 1, self.D.shape[0]))

        return resampled_max_freqs, resampled_max_freqs2
