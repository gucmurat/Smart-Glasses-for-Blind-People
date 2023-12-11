import os
import platform
from fastapi import FastAPI, HTTPException, Request

def text_to_speech(tx):
    tx = repr(tx)
    syst = platform.system()
    if syst == 'Linux':
        os.system('gtts-cli %s --output init.mp3 & cvlc --play-and-exit init.mp3' % tx)
    elif syst == 'Windows':
        os.system('PowerShell -Command "Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(%s);"' % tx)
    elif syst == 'Darwin':
        os.system('say %s' % tx)
    else:
        raise RuntimeError("Operating System '%s' is not supported" % syst)

app = FastAPI()

@app.post("/send_text")
async def send_text_to_client(info:Request):
    info_json = await info.json()
    text = info_json['text']
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    return text_to_speech(text)

"""
import requests
url = "http://127.0.0.1:8000/send_text"
data = {"text": "HelloFromClient"}
response = requests.post(url, json=data)
"""

#uvicorn speech:app --port 8000