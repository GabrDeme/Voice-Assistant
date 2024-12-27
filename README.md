# Voice-Assistant

## Overview

This project is a Python-based voice recorder system that integrates advanced functionalities like audio transcription and interaction with a Large Language Model (LLM). Users can record their voice, transcribe the audio, and receive responses from the LLM. The project also includes a text-to-speech feature to read out the LLM's responses.

## Technologies 

The project utilizes the following libraries and tools:

*sounddevice*: For audio recording.

*playsound*: To play audio files.

*scipy.io.wavfile*: For saving audio recordings as WAV files.

*pynput*: For capturing keyboard events.

*dotenv*: For environment variable management.

*langchain_groq*: For interacting with the ChatGroq LLM.

*faster_whisper*: For audio transcription.

*openai*: For generating text-to-speech outputs.

## Features

- **Voice Recording**: Records audio input from the user's microphone.

- **Audio Transcription**: Transcribes the recorded audio to text using the Faster Whisper library.

- **LLM Interaction**: Sends the transcription to the ChatGroq LLM and retrieves responses.

- **Text-to-Speech**: Converts the LLM's responses into audio output and plays it.

- **Keyboard Interaction**: Controlled via simple keyboard inputs (e.g., 'r' to record/stop).
