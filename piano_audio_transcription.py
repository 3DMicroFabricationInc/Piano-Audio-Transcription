import librosa
import numpy as np
from music21 import stream, note

# 1. Load audio
y, sr = librosa.load("./sound/morning_piano_progression_118bpm_C_minor.wav", sr=22050, mono=True)
#y, sr = librosa.load("piano.wav", sr=22050, mono=True)

# 2. Detect onsets (note start times)
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

# 3. Pitch detection using librosa's piptrack
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

# Helper: convert frequency to MIDI note number
def hz_to_midi(hz):
    return int(round(69 + 12 * np.log2(hz / 440.0)))

# 4. Build a music21 stream
score = stream.Stream()

for t in onset_frames:
    pitch_slice = pitches[:, t]
    mag_slice = magnitudes[:, t]

    if mag_slice.any():
        # Pick the strongest pitch at this onset
        idx = mag_slice.argmax()
        freq = pitch_slice[idx]
        if freq > 0:
            midi_num = hz_to_midi(freq)
            n = note.Note(midi_num)
            n.quarterLength = 1  # default duration
            score.append(n)

# 5. Export to MusicXML
score.write("musicxml", fp="output.xml")
print("Transcription saved to output.xml")
