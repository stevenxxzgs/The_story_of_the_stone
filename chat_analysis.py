#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   extract_dialogue.py
@Time    :   2025/01/10 10:40:09
@Author  :   stevenxxzg
@Version :   1.2
@Desc    :   从长文本中提取对话并保存为 JSONL 格式
'''

import json
import logging
from openai import OpenAI
from tqdm import tqdm
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize the Deepseek client
client = OpenAI(api_key="sk-ec5f21eb82bc4151b695be928cce25b3", base_url="https://api.deepseek.com")

# Schema for novel dialogue extraction
novel_schema = dict(
    task_description='Adapted from the novel into script',  # 小说改编成剧本
    attributes=[
        dict(
            name='role',
            description='The character who is speaking',
            type='String'
        ),  # 角色
        dict(
            name='dialogue',
            description='The dialogue spoken by the characters in the sentence',
            type='String'
        ),  # 对话
    ],
    example=[
        dict(
            text="""
                回来只见地下还有许多，宝玉正踟蹰间，只听背后有人说道：“你在这里作什么？”
                宝玉一回头，却是林黛玉来了，肩上担着花锄，锄上挂着花囊，手内拿着花帚。
                宝玉笑道：“好，好，来把这个花扫起来，撂在那水里。我才撂了好些在那里呢。”
                林黛玉道：“撂在水里不好。你看这里的水干净，只一流出去，有人家的地方脏的臭的混倒，仍旧把花遭塌了。那畸角上我有一个花冢，如今把他扫了，装在这绢袋里，拿土埋上，日久不过随土化了，岂不干净。”
                宝玉听了喜不自禁，笑道：“待我放下书，帮你来收拾。”
                黛玉道：“什么书？”
                宝玉见问，慌的藏之不迭，便说道：“不过是《中庸》《大学》。”
                黛玉笑道：“你又在我跟前弄鬼。趁早儿给我瞧，好多着呢。”
                宝玉道：“好妹妹，若论你，我是不怕的。你看了，好歹别告诉别人去。真真这是好书！你要看了，连饭也不想吃呢。”
                一面说，一面递了过去。
                林黛玉把花具且都放下，接书来瞧，从头看去，越看越爱看，不到一顿饭工夫，将十六出俱已看完，自觉词藻警人，余香满口。
                虽看完了书，却只管出神，心内还默默记诵。
            """,
            script=[
                {"role": "林黛玉", "dialogue": "你在这里作什么？"},
                {"role": "宝玉", "dialogue": "好，好，来把这个花扫起来，撂在那水里。我才撂了好些在那里呢。"},
                {"role": "林黛玉", "dialogue": "撂在水里不好。你看这里的水干净，只一流出去，有人家的地方脏的臭的混倒，仍旧把花遭塌了。那畸角上我有一个花冢，如今把他扫了，装在这绢袋里，拿土埋上，日久不过随土化了，岂不干净。"},
                {"role": "宝玉", "dialogue": "待我放下书，帮你来收拾。"},
                {"role": "林黛玉", "dialogue": "什么书？"},
                {"role": "宝玉", "dialogue": "不过是《中庸》《大学》。"},
                {"role": "林黛玉", "dialogue": "你又在我跟前弄鬼。趁早儿给我瞧，好多着呢。"},
                {"role": "宝玉", "dialogue": "好妹妹，若论你，我是不怕的。你看了，好歹别告诉别人去。真真这是好书！你要看了，连饭也不想吃呢。"}
            ]
        )
    ]
)

# Generate system prompt from schema
def generate_system_prompt(schema):
    """
    根据 schema 生成系统提示。
    :param schema: 包含任务描述、属性和示例的字典
    :return: 系统提示字符串
    """
    task_description = schema['task_description']
    attributes = schema['attributes']
    example = schema['example'][0]  # 使用第一个示例

    # 构建属性描述
    attributes_desc = "\n".join([
        f"- {attr['name']}: {attr['description']} (类型: {attr['type']})"
        for attr in attributes
    ])

    # 构建示例描述
    example_text = example['text'].strip()
    example_script = json.dumps(example['script'], ensure_ascii=False, indent=2)

    # 生成系统提示
    system_prompt = f"""
    {task_description}

    请从以下文本中提取对话，并将其格式化为 JSONL 格式。每个 JSON 对象应包含以下属性：
    {attributes_desc}

    示例：
    输入文本：
    {example_text}

    输出 JSONL：
    {example_script}
    """
    return system_prompt.strip()

# Function to read the text file in chunks
def read_file_in_chunks(file_path, chunk_size=5000):
    """
    读取大文件的分块生成器。
    :param file_path: 文件路径
    :param chunk_size: 每个分块的大小（字符数）
    :return: 生成器，每次返回一个分块
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        buffer = ""
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                if buffer:
                    yield buffer
                break
            buffer += chunk
            if len(buffer) >= chunk_size:
                last_period = buffer.rfind('。')
                if last_period == -1:
                    last_period = buffer.rfind('\n')
                if last_period != -1:
                    yield buffer[:last_period + 1]
                    buffer = buffer[last_period + 1:]

# Retry mechanism for API calls
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_dialogue(text, system_prompt):
    """
    调用 Deepseek API 分析对话。
    :param text: 输入的文本
    :param system_prompt: 系统提示
    :return: API 返回的 JSONL 格式结果
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract dialogue from the following text:\n\n{text}"},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"API request failed: {e}")
        raise

# Function to process the text file and save results
def process_file(file_path, output_file):
    """
    处理大文件并保存结果。
    :param file_path: 输入文件路径
    :param output_file: 输出文件路径
    """
    system_prompt = generate_system_prompt(novel_schema)  # 生成系统提示
    for chunk in tqdm(read_file_in_chunks(file_path), desc="Processing chunks"):
        try:
            result = analyze_dialogue(chunk, system_prompt)
            for line in result.splitlines():
                if line.strip():
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(line.strip() + '\n')
        except Exception as e:
            logging.error(f"Error processing chunk: {e}")
            time.sleep(5)

# Main script
if __name__ == "__main__":
    input_file = "data/red_nonum_clip.txt"  # 替换为你的输入文件路径
    output_file = "red_nonum_clip.jsonl"  # 输出文件路径

    # Process the file
    process_file(input_file, output_file)
    print(f"对话提取完成。结果已保存到 {output_file}")