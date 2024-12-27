import sounddevice as sd
from playsound import playsound
import scipy.io.wavfile as wavfile

from pynput import keyboard
import numpy as np
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from faster_whisper import WhisperModel
from openai import OpenAI

