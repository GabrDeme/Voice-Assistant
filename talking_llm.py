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

# Cria o modelo Whisper para transcrição de áudio 
whisper_model = WhisperModel("small", 
                                compute_type="int8",  # Tipo de computação (int8 otimiza o desempenho)
                                cpu_threads=os.cpu_count(),  # Usa o número de núcleos de CPU disponíveis
                                num_workers=os.cpu_count())  # Define o número de threads para o modelo

# Carrega as variáveis de ambiente de um arquivo .env
load_dotenv()

# Inicializa o cliente da API OpenAI
client = OpenAI()

# Define a classe VoiceRecorder
class VoiceRecorder:
    # Método construtor da classe
    def __init__(self, file_path='recording.wav', 
                 sample_rate=16000):
        self.file_path = file_path  # Caminho do arquivo de gravação
        self.sample_rate = sample_rate  # Taxa de amostragem para a gravação
        self.is_recording = False  # Variável de controle para saber se está gravando
        self.audio_data = []  # Lista para armazenar os dados de áudio
        self.stream = None  # Inicializa a variável de fluxo de áudio

        # Inicializa o LLM para processar a transcrição
        self.llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
        

    # Método para iniciar a gravação
    def start_recording(self):
        print("Recording started...") 
        self.audio_data = []  # Limpa os dados de áudio anteriores
        # Cria o stream de entrada de áudio
        self.stream = sd.InputStream(samplerate=self.sample_rate, 
                                     channels=1, callback=self.audio_callback)
        self.stream.start()  # Inicia a gravação

    # Método para parar a gravação
    def stop_recording(self):
        if self.stream:  # Verifica se o stream está ativo
            self.stream.stop()
            self.stream.close()  
        audio_np = np.concatenate(self.audio_data, axis=0)  # Junta os dados de áudio gravados
        # Salva os dados de áudio no formato WAV
        wavfile.write(self.file_path, self.sample_rate, audio_np)

    # Método de callback para processar os dados de áudio enquanto está gravando
    def audio_callback(self, indata, frames, time, status):
        if status:  # Se houver erro no status, exibe uma mensagem
            print(status, "Audio callback status error.")
        self.audio_data.append(indata.copy())  # Adiciona o pedaço de áudio gravado à lista de dados

    # Método para transcrever o áudio gravado para texto
    def transcribe_audio(self):
        segments, _ = whisper_model.transcribe(self.file_path, language="pt")  # Transcreve o áudio em português
        clean_prompt = "".join(segment.text for 
                           segment in segments).strip()  # Junta os textos transcritos em uma única string
        return clean_prompt  # Retorna o texto transcrito

    # Método para lidar com pressionamento de teclas
    def on_press(self, key):
        try:
            if key.char == 'r':  # Se a tecla 'r' for pressionada
                if not self.is_recording:  # Se não estiver gravando
                    self.is_recording = True  # Inicia a gravação
                    self.start_recording()
                else:  # Se já estiver gravando
                    self.is_recording = False  # Para a gravação
                    self.stop_recording()  # Para a gravação e salva o arquivo
                    transcript = self.transcribe_audio()  # Transcreve o áudio
                    print("User:", transcript)  # Exibe o texto transcrito

                    # Envia a transcrição para o LLM e exibe a resposta
                    llm_response = self.llm.invoke(transcript).content
                    print("LLM:", llm_response)
                    self.speak(llm_response)  # Reproduz a resposta do LLM

        except AttributeError:  # Caso seja uma tecla especial (como shift, ctrl, etc.)
            pass  # Ignora as teclas especiais
        
    # Método para gerar fala a partir do texto
    def speak(self, text):
        resposta = client.audio.speech.create(
            model='tts-1',  # Modelo de texto para fala (TTS)
            voice='echo',  # Voz utilizada para a fala
            input=text  # Texto que será convertido em fala
        )
        resposta.stream_to_file("out.wav")  # Salva o áudio gerado em um arquivo WAV
        playsound('out.wav')  # Reproduz o arquivo de áudio gerado

    # Método principal para iniciar o programa e aguardar a interação do usuário
    def start(self):
        print("Press 'r' to start/stop recording...")  # Instrução para o usuário
        with keyboard.Listener(on_press=self.on_press) as listener:  # Inicia o ouvinte de teclas
            listener.join()  # Aguarda a interação do usuário

# Exemplo de uso:
voice_recorder = VoiceRecorder()  # Cria uma instância do gravador de voz
voice_recorder.start()  # Inicia o processo, pressionando 'r' para começar/parar a gravação