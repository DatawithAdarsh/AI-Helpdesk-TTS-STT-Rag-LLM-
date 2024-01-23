import streamlit as st
import sounddevice as sd
import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import AutoProcessor, AutoModelForCTC
import openai
import os
from streamlit_chat import message
import streamlit as st
import torchaudio
print(torch.__version__)
print(torchaudio.__version__)
from aksharamukha import transliterate
from IPython.display import Audio, display
from transformers import AutoProcessor, SpeechT5ForTextToSpeech,SpeechT5Processor, SpeechT5HifiGan
import pyaudio
import wave
import sys
from array import array
from struct import pack


openai.api_key =  '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'



def parse_transcription(wav_file , lang):
  
    if(lang=="Hindi"):
      processor = Wav2Vec2Processor.from_pretrained("Harveenchadha/vakyansh-wav2vec2-hindi-him-4200")
      model = Wav2Vec2ForCTC.from_pretrained("Harveenchadha/vakyansh-wav2vec2-hindi-him-4200")
    if(lang=="English"):
      processor = AutoProcessor.from_pretrained("Harveenchadha/vakyansh-wav2vec2-indian-english-enm-700")
      model = AutoModelForCTC.from_pretrained("Harveenchadha/vakyansh-wav2vec2-indian-english-enm-700")
    if(lang=="Telgu"):
      processor = AutoProcessor.from_pretrained("Harveenchadha/vakyansh-wav2vec2-telugu-tem-100")
      model = AutoModelForCTC.from_pretrained("Harveenchadha/vakyansh-wav2vec2-telugu-tem-100" )
    if(lang=="Kannada"):
      processor = AutoProcessor.from_pretrained("Harveenchadha/vakyansh-wav2vec2-kannada-knm-560")
      model = AutoModelForCTC.from_pretrained("Harveenchadha/vakyansh-wav2vec2-kannada-knm-560")
    audio_input, sample_rate = sf.read(wav_file)
    input_values = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    print("Thankyou For Waiting")
    return transcription

sample_rate = 48000
#######################

def generate_audio(message,lang):
    if(lang=="English"):
      return ;
    model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language='indic',
                                     speaker='v3_indic')
    #orig_text = "व्यापार की वृद्धि सफलता की ओर अग्रसर होने का प्रमुख कारण है। सही नीतियों, संगठन, नवीनता और उत्कृष्टता से व्यापार बढ़ाएं। उत्पादों और सेवाओं की गुणवत्ता में सुधार करें। विपणन, बाजारीकरण और ग्राहक संबंधों को मजबूत करें। नए अवसरों का पता लगाएं और संचार का प्रभावी उपयोग करें।"
    
    if(lang=="Telgu"):
       roman_text = 	transliterate.process('Telugu', 'ISO', message)
       audio = model.apply_tts(roman_text, speaker='telugu_female')

    if(lang=="Hindi"):
       roman_text = 	transliterate.process('Devanagari', 'ISO', message)
       audio = model.apply_tts(roman_text, speaker='hindi_female')

    if(lang=="English"):
       roman_text = 	transliterate.process('Telugu', 'ISO', message)
       audio = model.apply_tts(roman_text, speaker='telugu_female')
       
    if(lang=="Kannada"):
       roman_text = 	transliterate.process('Kannada', 'ISO', message)
       audio = model.apply_tts(roman_text, speaker='kannada_female')
    #print(roman_text)
    

    torchaudio.save('female.wav', audio.unsqueeze(0), sample_rate)
    display(Audio(audio, rate=sample_rate))
    st.audio('female.wav', format='audio/wav')

###################

def generate_response(prompt):
    completions = openai.Completion.create (
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = completions.choices[0].text
    return message
    

st.title("Sasuke Uchiha")



if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'user_input' not in st.session_state:
    st.session_state.user_input = ''

def get_text(lang):
    input_type = st.radio("Input Type", ("Text", "Voice"), key="input_type")
    if input_type=="Text":
      input_text = st.text_input("You: ", placeholder="Type your Msg", key="input", value=st.session_state.user_input)
      return input_text 
    if input_type=="Voice" :
        audio_file = st.file_uploader("Upload Audio", type=["wav"], key="audio_input")
        return parse_transcription( audio_file , lang)

lang = st.selectbox("Select Language", ["English", "Hindi", "Telgu" , "Kannada"])

user_input = get_text(lang)

if user_input:
    output = generate_response(user_input)
    generate_audio(output,lang)
    st.session_state.past.append(user_input)
    #st.audio('female.wav')
    st.session_state.generated.append(output)
    st.session_state.user_input = ''  # clear text input


if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')