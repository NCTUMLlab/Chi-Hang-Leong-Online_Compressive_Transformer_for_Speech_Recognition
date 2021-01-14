import pyaudio
import time
from multiprocessing import Queue
import wave
import numpy
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()
qq = Queue()
def callback(in_data, frame_count, time_info, status):
    global qq
    qq.put(in_data)
    return (in_data, pyaudio.paContinue)

with wave.open("demo/demo001.wav",'rb') as a:
    pp = a.readframes(a.getnframes())
aa = [int.from_bytes(pp[2*i:2*i+2], byteorder='little',signed=True) for i in range(78391)]
bb = numpy.frombuffer(pp, dtype=numpy.int16)
pass

# stream = p.open(format=pyaudio.paInt16,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 frames_per_buffer=1024,
#                 input=True,
#                 output=False,
#                 stream_callback=callback)

# stream.start_stream()

# while stream.is_active():
#     if not qq.empty():
#         a = qq.get()
#     time.sleep(0.1)

# stream.stop_stream()
# stream.close()

# p.terminate()