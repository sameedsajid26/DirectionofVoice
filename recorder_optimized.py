import pyaudio
import wave
import numpy as np
import sys

RESPEAKER_RATE = 48000

CHANNELS = 6
WIDTH = 2

RESPEAKER_INDEX = 2 
CHUNK = 1024
RECORD_SECONDS = 2

OUTPUT_PATH = "../data/pyrecorder/"

p = pyaudio.PyAudio()

def find_device_index():
   found = -1
 
   for i in range(p.get_device_count()):
       dev = p.get_device_info_by_index(i)
       name = dev['name'].encode('utf-8')
       print(i, name, dev['maxInputChannels'], dev['maxOutputChannels'])
       if name.lower().find(b'respeaker') >= 0 and dev['maxInputChannels'] > 0:
           found = i
           break
 
   return found
 
 
device_index = find_device_index()
if device_index < 0:
   print('No ReSpeaker USB device found')
   sys.exit(1)



stream = p.open(
    format = p.get_format_from_width(WIDTH),
    #format = pyaudio.paInt16,
    rate = RESPEAKER_RATE,
    channels = CHANNELS,
    input = True,
    output = False,
    input_device_index=RESPEAKER_INDEX,
)


print("...........Recording Started............")

frames = [[] for i in range(CHANNELS)]

for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)

    for j in range(CHANNELS):
        ch_data = np.frombuffer(data, dtype=np.int16)[j::CHANNELS]
        frames[j].append(ch_data.tobytes())

print (".........Done Recording..........")

stream.stop_stream()
stream.close()
p.terminate()


for i in range (CHANNELS):
    WAVE_OUTPUT_FILENAME = f"channel_{i}.wav"
    WAVE_OUTPUT_FILEPATH = OUTPUT_PATH + WAVE_OUTPUT_FILENAME

    wf = wave.open(WAVE_OUTPUT_FILEPATH, 'wb')

    wf.setnchannels(1)
    wf.getsampwidth(p.get_sample_size(
        p.get_format_from_width(WIDTH)))
    wf.setframerate(RESPEAKER_RATE)
    wf.writeframes(b''.join(frames[i]))
    wf.close()