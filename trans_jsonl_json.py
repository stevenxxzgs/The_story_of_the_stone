import json

# 读取 JSONL 文件
input_file = 'red_nonum_clip.jsonl'  # 输入文件路径
output_file = 'daiyu.jsonl'  # 输出文件路径

# 初始化结果列表
result = []

# 读取文件内容
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 遍历每一行
for i in range(1, len(lines)):  # 从第二行开始，因为需要前一句
    try:
        # 解析当前行和前一行
        prev_line = json.loads(lines[i - 1])
        current_line = json.loads(lines[i])

        # 检查当前行的角色是否是黛玉或林黛玉
        if current_line['role'] in ['黛玉', '林黛玉']:
            # 构造新的 JSON 对象
            new_entry = {
                "instruction": prev_line['dialogue'],  # 前一句作为 instruction
                "input": "",  # 输入为空
                "output": current_line['dialogue']  # 黛玉的对话作为 output
            }
            result.append(new_entry)
    except json.JSONDecodeError as e:
        print(f"解析错误：{e}，跳过该行：{lines[i]}")

# 将结果写入新的 JSONL 文件
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in result:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"处理完成，结果已写入 {output_file}")