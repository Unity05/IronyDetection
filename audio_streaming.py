import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt


def audio_streaming(output_queue):
    CHUNK = 1024
    SAMPLE_FORMAT = pyaudio.paInt16
    #  SAMPLE_FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000

    REQUIRED_SILENCE_LENGTH = (RATE / CHUNK) * 0.3
    REQUIRED_SILENCE_LENGTH_FOR_SHUTDOWN = 50 * REQUIRED_SILENCE_LENGTH

    p = pyaudio.PyAudio()

    print('Recording.')

    stream = p.open(
        format=SAMPLE_FORMAT,
        channels=CHANNELS,
        rate=RATE,
        frames_per_buffer=CHUNK,
        input=True
    )

    SILENCE_COUNTER = 0
    save_frames = []
    last_frame = []
    i = 0
    while SILENCE_COUNTER < REQUIRED_SILENCE_LENGTH_FOR_SHUTDOWN:
        data = stream.read(CHUNK)
        data_int = struct.unpack(str(CHUNK) + 'h', data)
        frames = list(data_int)
        if np.mean(np.absolute(data_int)) < 100:
            SILENCE_COUNTER += 1
            if save_frames != []:
                if len(save_frames) < 1500:
                    save_frames = []
                else:
                    save_frames += frames
            else:
                last_frame = frames
        else:
            if save_frames == []:
                save_frames += last_frame
            save_frames += frames
            SILENCE_COUNTER = 0
        if SILENCE_COUNTER >= REQUIRED_SILENCE_LENGTH:
            if i > (REQUIRED_SILENCE_LENGTH + 1):
                output_queue.put(save_frames)
                save_frames = []
            i = 0
        i += 1

    stream.stop_stream()
    stream.close()

    p.terminate()

    print('Finished recording.')

    return
