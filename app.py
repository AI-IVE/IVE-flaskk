from flask import Flask, request, jsonify
import os
import config
import hashlib
import hmac
import time

import utils
import prompt
import dotenv
dotenv.load_dotenv(dotenv.find_dotenv())

app = Flask(__name__)




def remove_first_sentence(text):
    # \n\n을 기준으로 문자열을 분할합니다
    parts = text.split('\n\n', 1)
    
    # 분할된 부분이 2개 이상이면 (즉, \n\n이 존재하면)
    if len(parts) > 1:
        # 두 번째 부분부터 끝까지 반환합니다
        return parts[1].strip()
    else:
        # \n\n이 없으면 원래 문자열을 그대로 반환합니다
        return text.strip()

@app.route('/translate', methods=['GET', 'POST'])
def translate():
    seed_number = "187"  # 기존값 187 81 20 17 
    system_prompt_translate, system_prompt_review, glossary, word_list = utils.load_prompt(seed_number, lang="ja")
    
    if request.method == 'POST':
        data = request.json
        korean_text = data.get('text')
        model = data.get('model', 'gpt4')  # 기본값은 'gpt4'
    elif request.method == 'GET':
        korean_text = request.args.get('text')
        model = request.args.get('model', 'gpt4')  # 기본값은 'gpt4'
    else:
        return jsonify({"error": "Method not allowed"}), 405

    if not korean_text:
        return jsonify({"error": "No text provided"}), 400



    try:
        if model.lower() == 'gpt4':
            translation = remove_first_sentence ( utils.api_openai(system_prompt_translate, korean_text) )
        elif model.lower() == 'claude':
            translation = remove_first_sentence ( utils.api_claude_native(system_prompt_translate, korean_text) )
        else:
            return jsonify({"error": "Invalid model specified"}), 400

        return jsonify({"translation": translation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()

