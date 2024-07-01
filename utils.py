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

def clean_text(text):
    # print('clean_text[IN]:',text)
    if text is None:
        print("clean_text: parameter is None");
        return ""
    text_content = html.unescape(text)
    # print('clean_text[1]:',text_content)
    soup = BeautifulSoup(text_content, 'html.parser')
    
    for br in soup.find_all('br'):
        br.replace_with('\n')
    
    for p in soup.find_all('p'):
        p.replace_with(str(p.get_text()) + '\n')
    
    text_content = soup.get_text(separator='')
    text_content = text_content.strip()
    if text_content:
        # print('clean_text[OUT-1]:',text_content)
        return text_content
    else:
        root = etree.fromstring(text)
        text_content = ''
        for elem in root.iter():
            # for attr, value in elem.attrib.items():
            #     text_content += value + '\n'
            if 'value' in elem.attrib:
                soup = BeautifulSoup(elem.attrib['value'], 'html.parser')
                
                for br in soup.find_all('br'):
                    br.replace_with('\n')
                
                for p in soup.find_all('p'):
                    p.replace_with(str(p.get_text()) + '\n')
                
                text_content = soup.get_text(separator='')
                text_content = text_content.strip() + '\n'
            if 'description' in elem.attrib:
                soup = BeautifulSoup(elem.attrib['description'], 'html.parser')
                
                for br in soup.find_all('br'):
                    br.replace_with('\n')
                
                for p in soup.find_all('p'):
                    p.replace_with(str(p.get_text()) + '\n')
                
                text_content = soup.get_text(separator='')
                text_content = text_content.strip() + '\n'
        
        text_content = text_content.strip()
        if text_content:
            # print('clean_text[OUT-2]:', text_content)
            return text_content
        else:
            # print('clean_text[OUT-ELSE]:', text)
            return text

# 한국어 element 추출
def extract_korean_elements(element):
    if element.text and re.search(r'[\uAC00-\uD7A3]', element.text):
        yield element, 'text', ''

    for attr, value in element.attrib.items():
        if re.search(r'[\uAC00-\uD7A3]', value):
            yield element, attr, ''

    for child in element:
        yield from extract_korean_elements(child)

def extract_korean_elements_from_json(json_data, parent_key=""):
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            current_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, str) and re.search(r'[\uAC00-\uD7A3]', value):
                yield json_data, key, current_key
            elif isinstance(value, (dict, list)):
                yield from extract_korean_elements_from_json(value, current_key)
    elif isinstance(json_data, list):
        for i, item in enumerate(json_data):
            current_key = f"{parent_key}[{i}]"
            if isinstance(item, str) and re.search(r'[\uAC00-\uD7A3]', item):
                yield item, i, current_key
            elif isinstance(item, (dict, list)):
                yield from extract_korean_elements_from_json(item, current_key)

# 한글 존재 여부 확인
def is_korean(text):
    # 한글 존재 여부 확인
    if re.search(r'[\uAC00-\uD7A3]', text):
        return True
    else:
        decoded_text = html.unescape(text)
        if re.search(r'[\uAC00-\uD7A3]', decoded_text):
            return True
        else:
            return False

# 한국어 체크
def check_korean(text, file_path=None, verbose=False):
    lines = text.split('\n')
    korean_exists = False

    # 한글 존재 여부 확인
    total_cnt = 0
    korean_cnt = 0
    for i, line in enumerate(lines, start=1):
        total_cnt += 1
        if re.search(r'[\uAC00-\uD7A3]', line):
            if file_path is None:
                print(f"한글이 {i}번째 줄에 존재함: {line}")
            else:
                print(file_path, f"한글이 {i}번째 줄에 존재함: {line}")
            korean_exists = True
            korean_cnt += 1
        elif re.search(r'[\uAC00-\uD7A3]', html.unescape(line)):
            if file_path is None:
                print(f"인코딩된 한글이 {i}번째 줄에 존재함: {line}")
            else:
                print(file_path, f"인코딩된 한글이 {i}번째 줄에 존재함: {line}")
            korean_exists = True
            korean_cnt += 1

    if not korean_exists and verbose:
        print(file_path, "한글이 존재하지 않음")
    return korean_exists, korean_cnt, total_cnt

def check_japanese(text, file_path=None, verbose=False):
    lines = text.split('\n')
    japanese_exists = False

    # 일본어 존재 여부 확인 (히라가나, 가타카나, 한자)
    total_cnt = 0
    japanese_cnt = 0
    for i, line in enumerate(lines, start=1):
        total_cnt += 1
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', line):
            if file_path is None:
                print(f"일본어가 {i}번째 줄에 존재함: {line}")
            else:
                print(file_path, f"일본어가 {i}번째 줄에 존재함: {line}")
            japanese_exists = True
            japanese_cnt += 1

    if not japanese_exists and verbose:
        if file_path is None:
            print("일본어가 존재하지 않음")
        else:
            print(file_path, "일본어가 존재하지 않음")
    return japanese_exists, japanese_cnt, total_cnt

def check_chinese(text, file_path=None, verbose=False):
    lines = text.split('\n')
    chinese_exists = False

    # 일본어 한자를 제외한 중국어 간체 존재 여부 확인
    total_cnt = 0
    chinese_cnt = 0
    chinese_pattern = re.compile(r'[\u4E00-\u9FFF]')
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]')

    for i, line in enumerate(lines, start=1):
        total_cnt += 1
        chinese_chars = chinese_pattern.findall(line)
        japanese_chars = japanese_pattern.findall(line)

        if chinese_chars and not japanese_chars:
            if file_path is None:
                print(f"중국어가 {i}번째 줄에 존재함: {line}")
            else:
                print(file_path, f"중국어가 {i}번째 줄에 존재함: {line}")
            chinese_exists = True
            chinese_cnt += 1

    if not chinese_exists and verbose:
        if file_path is None:
            print("중국어가 존재하지 않음")
        else:
            print(file_path, "중국어가 존재하지 않음")
    return chinese_exists, chinese_cnt, total_cnt

def check_english_only(text, file_path=None, verbose=False):
    lines = text.split('\n')
    english_only = True

    # 영어로만 되어있는지 확인
    total_cnt = 0
    english_only_cnt = 0
    for i, line in enumerate(lines, start=1):
        total_cnt += 1
        if not re.search(r'^[a-zA-Z0-9\s\.\,\?\!\'\"\:\;\-\(\)\[\]\{\}\<\>\\/\|\@\#\$\%\^\&\*\_\+\=\~\`]+$', line):
            if file_path is None:
                print(f"영어 이외의 문자가 {i}번째 줄에 포함되어 있음: {line}")
            else:
                print(file_path, f"영어 이외의 문자가 {i}번째 줄에 포함되어 있음: {line}")
            english_only = False
        else:
            english_only_cnt += 1

    if english_only and verbose:
        if file_path is None:
            print("영어로만 되어 있음")
        else:
            print(file_path, "영어로만 되어 있음")
    return english_only, english_only_cnt, total_cnt

def check_language(text, file_path=None, verbose=False):
    check_korean(text, file_path, verbose=verbose)
    check_japanese(text, file_path, verbose=verbose)
    check_chinese(text, file_path, verbose=verbose)
    check_english_only(text, file_path, verbose=verbose)

def get_xml_string(element, pretty_print=False, with_tail=False):
    element_xml_string = etree.tostring(element, encoding='unicode', method='xml', pretty_print=pretty_print, with_tail=with_tail)
    element_xml_string = re.sub(r'&#x([0-9a-fA-F]+);', lambda match: chr(int(match.group(1), 16)), element_xml_string)
    return element_xml_string

def get_json_value_by_path(json_data, path):
    keys = path.split('.')
    value = json_data
    for key in keys:
        if re.search(r'\[\d+\]$', key):
            key_parts = re.split(r'(\[\d+\])$', key)
            key = key_parts[0]
            index = int(key_parts[1][1:-1])
            # print('index:',key_parts[1][1:-1], ', isinstance(value, dict):', isinstance(value, dict), ', key:', key, ', value:', value)
            if isinstance(value, dict) and key in value:
                value = value[key]
                # print("value0:", value)
                if isinstance(value, list) and 0 <= index < len(value):
                    value = value[index]
                    # print("value1:", value)
                else:
                    return None
            else:
                return None
        else:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
    # if re.search(r'\[\d+\]', path):
    #     print("value:", value)
    return value

def extract_chunked_text(elements, filename, max_tokens=1000 ):
    file_tokens_len = 0
    element_texts = []
    for element, attr, _ in elements:
        extracted_texts = []
        try:
            if isinstance(element, dict):
                text = element[attr]
            elif isinstance(element, list):
                text = element[attr]
            elif hasattr(element, 'text') and attr == "text":
                text = element.text.strip() if element.text else ""
            elif attr == "value" and hasattr(element, 'attrib') and "hashkey" in element.attrib and element.attrib["hashkey"] == "author":
                text = "Inswave"
            elif attr == "value" and hasattr(element, 'tag') and element.tag == "author":
                text = "Inswave"
            elif hasattr(element, 'attrib'):
                text = element.attrib.get(attr, "")
            else:
                text = ""
        except AttributeError as e:
            print("    ", f"오류 발생 (extract_text): {str(e)}, Element: {element}, Attr: {attr}")
            text = ""

        tokens_len = tokens_len_from_string(text, "cl100k_base")
        file_tokens_len += tokens_len

        if tokens_len > max_tokens:
            text_parts = split_text_into_parts(text, max_tokens, "cl100k_base")

            for part in text_parts:
                part_tokens_len = tokens_len_from_string(part, "cl100k_base")
                print("    ", f"텍스트 분할, File: {filename}, Tokens:{tokens_len}, Part tokens: {part_tokens_len}")
                extracted_texts.append(part)
        else:
            extracted_texts.append(text)
        element_texts.append((extracted_texts, element, attr))
    return element_texts, file_tokens_len

def unescape_str(text):
    if '&lt;' in text[9:-10] and '&gt;' in text[9:-10] and '<' not in text[9:-10] and '>' not in text[9:-10] and "<content>" == text[:9] and "</content>" == text[-10:]:
        text_enc = html.unescape(text)
        text_enc = re.sub(r'^<content>', '<content_encode>', text_enc)
        text_enc = re.sub(r'</content>$', '</content_encode>', text_enc)
        return text_enc
    else:
        return text

def escape_str(text):
    if '<content_encode>' in text or '</content_encode>' in text:
        text_dec = html.escape(text)
        text_dec = re.sub(r'^&lt;content_encode&gt;', '<content>', text_dec)
        text_dec = re.sub(r'&lt;/content_encode&gt;$', '</content>', text_dec)
        return text_dec
    else:
        return text

def api_claude_langchain(system_prompt, text, model="claude-3-sonnet", verbose=False, max_retries=5, initial_delay=2):
    if model == "claude-3-opus":
        model = "claude-3-opus-20240229"
    elif model == "claude-3-sonnet":
        model = "claude-3-sonnet-20240229"
    else:
        model = "claude-3-haiku-20240307"

    retry_count = 0
    delay = initial_delay

    while retry_count < max_retries:
        try:
            chat = ChatAnthropic(temperature=0, model=model)

            chat_messages = [SystemMessage(content=system_prompt), HumanMessage(content=text)]
            prompt = ChatPromptTemplate.from_messages(chat_messages)

            chain = (prompt | chat).with_config({"tags": ["anthropic", "claude-3", model], "metadata": {"model_id": model}})
            tracer = LangChainTracer(project_name="AI-Translator")
            translated_text = chain.invoke({}, config={"callbacks": [tracer]}).content.strip()

            if verbose:
                print("Anthropic", model, "\n\n<<system_prompt>>\n", system_prompt, "\n\n<<user_prompt>>\n", text, "\n\n<<response>>\n", translated_text)

            return translated_text

        except Exception as e:
            retry_count += 1
            print(f"{model} 호출 중 오류 발생 (Attempt {retry_count}/{max_retries}): {type(e).__name__} - {str(e)}")
            if retry_count < max_retries:
                time.sleep(delay + random.uniform(0, 1))
                delay *= 2
            else:
                raise e

def api_claude(system_prompt, text, model="claude-3-sonnet", use_langchain=True, bedrock=True, verbose=False, max_retries=5, initial_delay=2):
    if use_langchain == False:
        return api_claude_native(system_prompt, text, model=model, verbose=verbose, max_retries=max_retries, initial_delay=initial_delay)
    if bedrock == False:
        return api_claude_langchain(system_prompt, text, model=model, verbose=verbose, max_retries=max_retries, initial_delay=initial_delay)
    if model == "claude-3-opus":
        model = "anthropic.claude-3-opus-20240229-v1:0"
    elif model == "claude-3-sonnet":
        model = "anthropic.claude-3-sonnet-20240229-v1:0"
    else:
        model = "anthropic.claude-3-haiku-20240307-v1:0"

    retry_count = 0
    delay = initial_delay

    # https://medium.com/@dminhk/building-with-anthropics-claude-3-on-amazon-bedrock-and-langchain-%EF%B8%8F-2b842f9c0ca8
    client = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
    model_kwargs =  { 
        "max_tokens": 4000,
        "temperature": 0.0
    }
    
    chat = ChatBedrock(
        client=client,
        model_id=model,
        model_kwargs=model_kwargs,
    )

    while retry_count < max_retries:
        try:
            chat_messages = [SystemMessage(content=system_prompt), HumanMessage(content=text)]
            prompt = ChatPromptTemplate.from_messages(chat_messages)
            
            chain = (prompt | chat).with_config({"tags": ["bedrock", "claude-3", model], "metadata": {"model_id": model}})
            tracer = LangChainTracer(project_name="AI-Translator")
            translated_text = chain.invoke({}, config={"callbacks": [tracer]}).content.strip()

            if verbose:
                print("Anthropic_bedrock", model, "\n\n<<system_prompt>>\n", system_prompt, "\n\n<<user_prompt>>\n", text, "\n\n<<response>>\n", translated_text)

            return translated_text

        except Exception as e:
            retry_count += 1
            print(f"{model} 호출 중 오류 발생 (Attempt {retry_count}/{max_retries}): {type(e).__name__} - {str(e)}")
            if retry_count < max_retries:
                time.sleep(delay + random.uniform(0, 1))
                delay *= 2
            else:
                raise e

def api_claude_native(system_prompt, text, model="claude-3-sonnet", verbose=False, max_retries=5, initial_delay=2):
    if model == "claude-3-opus":
        model = "claude-3-opus-20240229"
    elif model == "claude-3-sonnet":
        model = "claude-3-sonnet-20240229"
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

def api_claude_multi_langchain(system_prompt, messages, model="claude-3-sonnet", verbose=False, max_retries=5, initial_delay=2):
    if model == "claude-3-opus":
        model = "claude-3-opus-20240229"
    elif model == "claude-3-sonnet":
        model = "claude-3-sonnet-20240229"
    else:
        model = "claude-3-haiku-20240307"

    retry_count = 0
    delay = initial_delay

    while retry_count < max_retries:
        try:
            chat = ChatAnthropic(max_tokens=4000, temperature=0, model=model)
            
            system_message = SystemMessage(content=system_prompt)
            chat_messages = [system_message]

            for msg in messages:
                role = msg['role']
                content = msg['content'][0]['text']
                if role == 'user':
                    chat_messages.append(HumanMessage(content=content))
                elif role == 'assistant':
                    chat_messages.append(AIMessage(content=content))

            prompt = ChatPromptTemplate.from_messages(chat_messages)
            chain = (prompt | chat).with_config({"tags": ["anthropic", "claude-3", model, "multi"], "metadata": {"model_id": model}})
            tracer = LangChainTracer(project_name="AI-Translator")
            translated_text = chain.invoke({}, config={"callbacks": [tracer]}).content.strip()

            if verbose:
                print(f"Anthropic {model}\n\n<<system_prompt>>\n{system_prompt}")
                for msg in messages:
                    role = msg['role']
                    content = msg['content'][0]['text']
                    print(f"\n\n<<{role}_prompt>>\n{content}")
                print(f"\n\n<<response>>\n{translated_text}")

            return translated_text

        except Exception as e:
            retry_count += 1
            print(f"{model} 호출 중 오류 발생 (Attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                time.sleep(delay + random.uniform(0, 1))
                delay *= 2
            else:
                raise e

def api_claude_multi(system_prompt, messages, model="claude-3-sonnet", use_langchain=True, bedrock=True, verbose=False, max_retries=5, initial_delay=2):
    if use_langchain == False:
        return api_claude_multi_native(system_prompt, messages, model=model, verbose=verbose, max_retries=max_retries, initial_delay=initial_delay)
    if bedrock == False:
        return api_claude_multi_langchain(system_prompt, messages, model=model, verbose=verbose, max_retries=max_retries, initial_delay=initial_delay)
    if model == "claude-3-opus":
        model = "anthropic.claude-3-opus-20240229-v1:0"
    elif model == "claude-3-sonnet":
        model = "anthropic.claude-3-sonnet-20240229-v1:0"
    else:
        model = "anthropic.claude-3-haiku-20240307-v1:0"

    retry_count = 0
    delay = initial_delay

    # https://medium.com/@dminhk/building-with-anthropics-claude-3-on-amazon-bedrock-and-langchain-%EF%B8%8F-2b842f9c0ca8
    client = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
    model_kwargs =  { 
        "max_tokens": 4000,
        "temperature": 0.0
    }
    
    chat = ChatBedrock(
        client=client,
        model_id=model,
        model_kwargs=model_kwargs,
    )

    while retry_count < max_retries:
        try:
            system_message = SystemMessage(content=system_prompt)
            chat_messages = [system_message]

            for msg in messages:
                role = msg['role']
                content = msg['content'][0]['text'] if msg['content'] and msg['content'][0] and 'text' in msg['content'][0] else ''
                if role == 'user':
                    chat_messages.append(HumanMessage(content=content))
                elif role == 'assistant':
                    chat_messages.append(AIMessage(content=content))

            prompt = ChatPromptTemplate.from_messages(chat_messages)
            chain = (prompt | chat).with_config({"tags": ["bedrock", "claude-3", model, "multi"], "metadata": {"model_id": model}})
            tracer = LangChainTracer(project_name="AI-Translator")
            translated_text = chain.invoke({}, config={"callbacks": [tracer]}).content.strip()

            if verbose:
                print(f"Anthropic_bedrock {model}\n\n<<system_prompt>>\n{system_prompt}")
                for msg in messages:
                    role = msg['role']
                    content = msg['content'][0]['text']
                    print(f"\n\n<<{role}_prompt>>\n{content}")
                print(f"\n\n<<response>>\n{translated_text}")

            return translated_text

        except Exception as e:
            retry_count += 1
            print(f"{model} 호출 중 오류 발생 (Attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                time.sleep(delay + random.uniform(0, 1))
                delay *= 2
            else:
                raise e

def api_claude_multi_native(system_prompt, messages, model="claude-3-sonnet", verbose=False, max_retries=5, initial_delay=2):
    if model == "claude-3-opus":
        model = "claude-3-opus-20240229"
    elif model == "claude-3-sonnet":
        model = "claude-3-sonnet-20240229"
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
                messages=messages
            )

            translated_text = message.content[0].text

            if verbose:
                print(f"Anthropic {model}\n\n<<system_prompt>>\n{system_prompt}")
                for msg in messages:
                    role = msg['role']
                    content = msg['content'][0]['text']
                    print(f"\n\n<<{role}_prompt>>\n{content}")
                print(f"\n\n<<response>>\n{translated_text}")

            return translated_text

        except Exception as e:
            retry_count += 1
            print(f"{model} 호출 중 오류 발생 (Attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                time.sleep(delay + random.uniform(0, 1))
                delay *= 2
            else:
                raise e

def api_openai(system_prompt, text, model="gpt-4-turbo", use_langchain=True, verbose=False, max_retries=5, initial_delay=2):
    if use_langchain == False:
        return api_openai_native(system_prompt, text, model=model, verbose=verbose, max_retries=max_retries, initial_delay=initial_delay)
    retry_count = 0
    delay = initial_delay

    while retry_count < max_retries:
        try:
            chat = ChatOpenAI(temperature=0, model_name=model)
            chat_messages = [SystemMessage(content=system_prompt), HumanMessage(content=text)]
            prompt = ChatPromptTemplate.from_messages(chat_messages)

            chain = (prompt | chat).with_config({"tags": ["openai", "chatgpt", model], "metadata": {"model_id": model}})
            tracer = LangChainTracer(project_name="AI-Translator")
            translated_text = chain.invoke({}, config={"callbacks": [tracer]}).content.strip()

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

def api_openai_native(system_prompt, text, model="gpt-4-turbo", verbose=False, max_retries=5, initial_delay=2):
    global seed_number

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

def api_openai_multi(system_prompt, messages, model="gpt-4-turbo", use_langchain=True, verbose=False, max_retries=5, initial_delay=2):
    global seed_number
    if use_langchain == False:
        return api_openai_multi_native(system_prompt, messages, model=model, verbose=verbose, max_retries=max_retries, initial_delay=initial_delay)
    retry_count = 0
    delay = initial_delay

    while retry_count < max_retries:
        try:
            chat = ChatOpenAI(temperature=0, model_name=model)
            system_message = SystemMessage(content=system_prompt)
            chat_messages = [system_message]

            for msg in messages:
                role = msg['role']
                content = msg['content'][0]['text'] if msg['content'] and msg['content'][0] and 'text' in msg['content'][0] else ''
                if role == 'user':
                    chat_messages.append(HumanMessage(content=content))
                elif role == 'assistant':
                    chat_messages.append(AIMessage(content=content))

            prompt = ChatPromptTemplate.from_messages(chat_messages)
            chain = (prompt | chat).with_config({"tags": ["openai", "chatgpt", model, "multi"], "metadata": {"model_id": model}})
            tracer = LangChainTracer(project_name="AI-Translator")
            translated_text = chain.invoke({}, config={"callbacks": [tracer]}).content.strip()

            if verbose:
                print(f"OpenAI {model}\n\n<<system_prompt>>\n{system_prompt}")
                for msg in messages:
                    role = msg['role']
                    content = msg['content'][0]['text']
                    print(f"\n\n<<{role}_prompt>>\n{content}")
                print(f"\n\n<<response>>\n{translated_text}")

            return translated_text

        except Exception as e:
            retry_count += 1
            print(f"{model} 호출 중 오류 발생 (Attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                time.sleep(delay + random.uniform(0, 1))
                delay *= 2
            else:
                raise e

def api_openai_multi_native(system_prompt, messages, model="gpt-4-turbo", verbose=False, max_retries=5, initial_delay=2):
    global seed_number
    retry_count = 0
    delay = initial_delay

    while retry_count < max_retries:
        try:
            openai_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                role = msg['role']
                content = msg['content'][0]['text']
                openai_messages.append({"role": role, "content": content})

            message = client_openai.chat.completions.create(
                model=model,
                max_tokens=4000,
                temperature=0,
                # seed=int(seed_number),
                messages=openai_messages
            )

            translated_text = message.choices[0].message.content

            if verbose:
                print(f"OpenAI {model}\n\n<<system_prompt>>\n{system_prompt}")
                for msg in messages:
                    role = msg['role']
                    content = msg['content'][0]['text']
                    print(f"\n\n<<{role}_prompt>>\n{content}")
                print(f"\n\n<<response>>\n{translated_text}")

            return translated_text

        except Exception as e:
            retry_count += 1
            print(f"{model} 호출 중 오류 발생 (Attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                time.sleep(delay + random.uniform(0, 1))
                delay *= 2
            else:
                raise e

def tokens_len_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def split_text_into_parts(text, max_tokens, model_name):
    # 텍스트를 최대 토큰 수에 맞게 여러 부분으로 분할
    tokens = text.split("\n")
    parts = []
    current_part = []
    current_count = 0
    
    for token in tokens:
        if current_count + 1 > max_tokens:
            parts.append('\n'.join(current_part))
            current_part = []
            current_count = 0
        current_part.append(token)
        current_count += tokens_len_from_string(token, model_name)
    
    if current_part:
        parts.append('\n'.join(current_part))  # 남은 부분 추가
    
    return parts

def load_prompt(seed_number, lang="jp"):
    with open(f"./prompt/prompt_glossary_{lang}.txt", "r", encoding="utf-8") as file:
        glossary_lines = file.readlines()

    glossary = ''.join('   ' + line for line in glossary_lines)

    with open(f"./prompt/prompt_word_{lang}.txt", "r", encoding="utf-8") as file:
        word_list_lines = file.readlines()

    word_list = ''.join('   ' + line for line in word_list_lines)

    with open(f"./prompt/prompt_translate_{lang}.txt", "r", encoding="utf-8") as file:
        system_prompt_translate = file.read()

    system_prompt_translate = system_prompt_translate.replace("{glossary}", glossary)
    system_prompt_translate = system_prompt_translate.replace("{word}", word_list)
    system_prompt_translate = system_prompt_translate.replace("{seed}", seed_number)

    with open(f"./prompt/prompt_review_{lang}.txt", "r", encoding="utf-8") as file:
        system_prompt_review = file.read()

    system_prompt_review = system_prompt_review.replace("{prompt_translation}", system_prompt_translate)
    return system_prompt_translate, system_prompt_review, glossary, word_list


def load_improved_prompt(seed_number, lang="jp"):
    with open(f"./prompt/prompt_glossary_{lang}.txt", "r", encoding="utf-8") as file:
        glossary_lines = file.readlines()

    glossary = ''.join('   ' + line for line in glossary_lines)

    with open(f"./prompt/prompt_word_{lang}.txt", "r", encoding="utf-8") as file:
        word_list_lines = file.readlines()

    word_list = ''.join('   ' + line for line in word_list_lines)

    with open(f"./prompt/prompt_translate_improvement_system_{lang}.txt", "r", encoding="utf-8") as file:
        system_prompt_translate = file.read()

    system_prompt_translate = system_prompt_translate.replace("{glossary}", glossary)
    system_prompt_translate = system_prompt_translate.replace("{word}", word_list)
    system_prompt_translate = system_prompt_translate.replace("{seed}", seed_number)

    with open(f"./prompt/prompt_translate_improvement_user_{lang}.txt", "r", encoding="utf-8") as file:
        system_prompt_translate_improvement = file.read()

    system_prompt_translate_improvement = system_prompt_translate_improvement.replace("{glossary}", glossary)
    system_prompt_translate_improvement = system_prompt_translate_improvement.replace("{word}", word_list)
    system_prompt_translate_improvement = system_prompt_translate_improvement.replace("{seed}", seed_number)

    return system_prompt_translate, system_prompt_translate_improvement, glossary, word_list