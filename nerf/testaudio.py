import pyaudio
import numpy as np
from queue import Queue
from threading import Thread, Event

from pynput import keyboard

def _read_frame(stream, exit_event, queue, chunk):

    while True:
        if exit_event.is_set():
            print('[INFO] read frame thread ends')
            break
        frame = stream.read(chunk, exception_on_overflow=False)
        frame = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32767 # [chunk]
        queue.put(frame)
        #print(queue.qsize())

def _play_frame(stream, exit_event, queue, chunk):

    while True:
        if exit_event.is_set():
            print('[INFO] play frame thread ends')
            break
        print(queue.qsize())
        frame = queue.get()
        frame = (frame * 32767).astype(np.int16).tobytes()
        stream.write(frame, chunk)

def on_press(key):
    if key == keyboard.Key.enter:
        print('you press Enter')
        return False

def on_release(key):
    if key == keyboard.Key.enter:
        print('you release Enter')

exit_event = Event()
audio_instance = pyaudio.PyAudio()

input_stream = audio_instance.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, output=False, frames_per_buffer=320)
queue = Queue()
process_read_frame = Thread(target=_read_frame, args=(input_stream, exit_event, queue, 320))
output_stream = audio_instance.open(format=pyaudio.paInt16, channels=1, rate=16000, input=False, output=True, frames_per_buffer=320)
output_queue = Queue()
process_play_frame = Thread(target=_play_frame, args=(output_stream, exit_event, output_queue, 320))

process_read_frame.start()
process_play_frame.start()

with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
    listener.join()

#keyboard.wait('q')

exit_event.set()
output_stream.stop_stream()
output_stream.close()
process_play_frame.join()
input_stream.stop_stream()
input_stream.close()
process_read_frame.join()
