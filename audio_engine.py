import os
import gtts
from playsound import playsound
import threading
class AudioEngine():
    def __init__(self, objects):
        if not os.path.exists("audio"):
            os.makedirs("audio")
        for object in objects:
            tts = gtts.gTTS(object, lang="en")
            tts.save("audio/" + object + ".mp3")

    def _playAsnc(self, fname):
        threading.Thread(target=playsound, args=(fname,)).start()


    def playsound(self, obj_name):
        self._playAsnc("audio/" + obj_name + ".mp3")

    def playIncorrectSound(self):
        self._playAsnc("wrong.wav")

    def playCorrectSound(self):
        self._playAsnc("correct.wav")
