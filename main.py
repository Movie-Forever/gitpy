import os
from tkinter import Tk, filedialog, Button, Label, StringVar
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment


class AudioAnalyzer:
    def __init__(self, root):
        self.root = root
        self.file_path = ""
        self.load_button = Button(root, text="Load Audio", command=self.load_audio)
        self.load_button.pack()

    def load_audio(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3;*.aac")])

        if self.file_path:
            self.process_audio()

    def process_audio(self):
        audio = AudioSegment.from_file(self.file_path)

        # Check and convert to wav
        if audio.channels > 1:
            audio = audio.set_channels(1)

        if not self.file_path.lower().endswith(".wav"):
            wav_file_path = os.path.splitext(self.file_path)[0] + ".wav"
            audio.export(wav_file_path, format="wav")
            self.file_path = wav_file_path

        # Remove metadata
        audio = AudioSegment.from_wav(self.file_path)

        # Display audio file name
        print(f"File Name: {os.path.basename(self.file_path)}")

        # Display time in seconds
        time_in_seconds = len(audio) / 1000.0
        print(f"Time: {time_in_seconds:.2f} seconds")

        # Compute highest resonance frequency
        samples = np.array(audio.get_array_of_samples())
        frequencies, amplitudes = self.compute_highest_resonance(samples)
        max_freq_index = np.argmax(amplitudes)
        max_freq = frequencies[max_freq_index]
        print(f"Highest Resonance Frequency: {max_freq:.2f} Hz")

        # Compute Low, Mid, High Frequency
        low, mid, high = self.compute_frequency_ranges(samples, audio.frame_rate)
        print(f"Low Frequency: {low} Hz")
        print(f"Mid Frequency: {mid} Hz")
        print(f"High Frequency: {high} Hz")

        # Plot waveform
        plt.subplot(2, 3, 1)
        plt.plot(np.linspace(0, time_in_seconds, len(samples)), samples)
        plt.title("Waveform")

        # Plot RT60 for Low, Mid, High frequencies
        for i, freq_range in enumerate([(0, low), (low, mid), (mid, high)]):
            rt60 = self.compute_rt60(samples, audio.frame_rate, freq_range)
            plt.subplot(2, 3, i + 2)
            plt.plot(rt60, label=f"{freq_range[0]}-{freq_range[1]} Hz")
            plt.title(f"RT60 - {freq_range[0]}-{freq_range[1]} Hz")
            plt.legend()

        plt.show()

    def compute_highest_resonance(self, samples):
        frequencies, amplitudes = plt.psd(samples, NFFT=1024, Fs=44100)
        return frequencies, amplitudes

    def compute_frequency_ranges(self, samples, frame_rate):
        n = len(samples)
        freq_range = np.fft.fftfreq(n, d=1 / frame_rate)

        low = 20
        mid = 1000
        high = 5000

        low_range = np.where((freq_range >= 0) & (freq_range < low))[0]
        mid_range = np.where((freq_range >= low) & (freq_range < mid))[0]
        high_range = np.where((freq_range >= mid) & (freq_range < high))[0]

        return low, mid, high

    def compute_rt60(self, samples, frame_rate, freq_range):
        n = len(samples)
        freq_range_indices = np.where((freq_range[0] <= np.fft.fftfreq(n, d=1 / frame_rate)) &
                                      (np.fft.fftfreq(n, d=1 / frame_rate) < freq_range[1]))[0]
        freq_samples = np.fft.fft(samples)
        decay = np.abs(freq_samples[freq_range_indices])
        rt60 = -60 / np.gradient(decay)
        return rt60


if __name__ == "__main__":
    root = Tk()
    root.title("Audio Analyzer")
    analyzer = AudioAnalyzer(root)
    root.mainloop()
