#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   extract_dialogue.py
@Time    :   2025/01/10 10:40:09
@Author  :   stevenxxzg
@Version :   1.2
@Desc    :   从长文本中提取对话并保存为 JSONL 格式
'''
import re

# 读取文件内容
with open('data/red.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# 使用正则表达式删除所有数字
content_without_numbers = re.sub(r'\d+', '', content)

# 将处理后的内容写回文件
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(content_without_numbers)

print("数字已删除并保存到output.txt")