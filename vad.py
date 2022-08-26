import webrtcvad
vad = webrtcvad.Vad()

vad.set_mode(1)

sample_rate = 16000
frame_duration = 10  # ms
frame = b'\x00\x00' * int(sample_rate * frame_duration / 1000)
print(f'Contains speech: %s' % (vad.is_speech(frame, sample_rate)))
