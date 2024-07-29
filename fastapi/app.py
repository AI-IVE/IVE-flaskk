from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import config
import hashlib
import hmac
import time

import utils
import prompt
import dotenv
dotenv.load_dotenv(dotenv.find_dotenv())

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class TranslationRequest(BaseModel):
    text: str
    model: str = "gpt4"

def remove_first_sentence(text):
    parts = text.split('\n\n', 1)
    if len(parts) > 1:
        return parts[1].strip()
    else:
        return text.strip()

@app.post("/translate")
@app.get("/translate")
async def translate(request: Request, data: TranslationRequest = None):
    seed_number = "187"  # 기존값 187 81 20 17 
    
    if request.method == "POST":
        if data is None:
            raise HTTPException(status_code=400, detail="No data provided")
        text = data.text
        model = data.model
    elif request.method == "GET":
        text = request.query_params.get('text')
        model = request.query_params.get('model', 'gpt4')
    else:
        raise HTTPException(status_code=405, detail="Method not allowed")

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    lang = utils.get_language(text)
    return_lang = "ko" if lang == "jp" else "jp"
    system_prompt_translate = utils.load_prompt(seed_number, lang="ja")

    try:
        if model.lower() == 'gpt4':
            translation = remove_first_sentence(utils.api_openai_native(system_prompt_translate, text, "gpt-4o"))
        elif model.lower() == 'claude':
            translation = remove_first_sentence(utils.api_claude_native(system_prompt_translate, text))
        else:
            raise HTTPException(status_code=400, detail="Invalid model specified")

        return {"translation": translation, "return_lang": return_lang}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    import uvicorn
    # http://127.0.0.1:5000 # 기존 flask 주소 
    uvicorn.run(app, host="127.0.0.1", port=5000)