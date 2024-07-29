import anthropic
import openai
import tiktoken
from lxml import etree
import re
import html
import time
import random
import json
import boto3

from bs4 import BeautifulSoup
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from langchain.callbacks.tracers import LangChainTracer

client_anthropic = anthropic.Anthropic()
client_openai = openai.OpenAI()


def get_language(text):
    systemPrompt = """사용자의 메시지를 분석하여 사용된 언어가 한국어인 경우 'ko', 일본어인 경우 'jp', 영어인 경우 'en'으로만 응답하세요. 다른 설명이나 추가 텍스트 없이 해당 코드만 출력하세요.
                    사용자 질문 : """
    userprompt = text
                    
    completion = client_openai .chat.completions.create(
    model="gpt-4o-2024-05-13",
    messages=[
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": userprompt}
    ]

    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content
    




def api_claude_native(system_prompt, text, model="claude-3-sonnet", verbose=False, max_retries=5, initial_delay=2):
    if model == "claude-3-opus":
        model = "claude-3-opus-20240229"
    elif model == "claude-3-sonnet":
        model = "claude-3-sonnet-20240229"
    elif model == "claude-3.5-sonnet":
        model = "claude-3-5-sonnet-20240620"
    else:
        model = "claude-3-haiku-20240307"

    retry_count = 0
    delay = initial_delay
    
    while retry_count < max_retries:
        try:
            message = client_anthropic.messages.create(
                model=model,  # claude-3-opus-20240229 claude-3-sonnet-20240229 claude-3-haiku-20240307
                max_tokens=4000,
                temperature=0,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text
                            }
                        ]
                    }
                ]
            )
            translated_text = message.content[0].text
            if verbose:
                print("Anthropic", model, "\n\n<<system_prompt>>\n", system_prompt, "\n\n<<user_prompt>>\n", text, "\n\n<<response>>\n", translated_text)
            return translated_text
        except Exception as e:
            retry_count += 1
            print(f"{model} 호출 중 오류 발생 (Attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                time.sleep(delay + random.uniform(0, 1))
                delay *= 2
            else:
                raise e


def api_openai_native(system_prompt, text, model="gpt-4-turbo", verbose=False, max_retries=5, initial_delay=2):
    global seed_number
    if model == "gpt-4-turbo":
        model = "gpt-4-turbo"
    elif model == "gpt-4o":
        model = "gpt-4o"
    else:
        model = "gpt-3.5-turbo"

    retry_count = 0
    delay = initial_delay

    while retry_count < max_retries:
        try:
            message = client_openai.chat.completions.create(
                model=model,
                max_tokens=4000,
                temperature=0,
                # seed=int(seed_number),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
            )
            translated_text = message.choices[0].message.content
            if verbose:
                print("OpenAI", model, "\n\n<<system_prompt>>\n", system_prompt, "\n\n<<user_prompt>>\n", text, "\n\n<<response>>\n", translated_text)
            return translated_text
        except Exception as e:
            retry_count += 1
            print(f"{model} 호출 중 오류 발생 (Attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                time.sleep(delay + random.uniform(0, 1))
                delay *= 2
            else:
                raise e




def load_prompt(seed_number, lang="jp"):
    with open(f"./prompt/prompt_glossary_{lang}.txt", "r", encoding="utf-8") as file:
        glossary_lines = file.readlines()

    glossary = ''.join('   ' + line for line in glossary_lines)

    
    with open(f"./prompt/prompt_word_{lang}.txt", "r", encoding="utf-8") as file:
        word_list_lines = file.readlines()

    word_list = ''.join('   ' + line for line in word_list_lines)

    with open(f"./prompt/prompt_translate1_{lang}.txt", "r", encoding="utf-8") as file:
        system_prompt_translate = file.read()

    system_prompt_translate = system_prompt_translate.replace("{glossary}", glossary)
    system_prompt_translate = system_prompt_translate.replace("{word}", word_list)
    system_prompt_translate = system_prompt_translate.replace("{seed}", seed_number)

   
    return system_prompt_translate 


