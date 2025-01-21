from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

# 模型路径
model_path = '../model/LLM-Research/Meta-Llama-3___1-8B-Instruct'
lora_path = './output/llama3_1_instruct_lora/checkpoint-72'  # 这里改成你的 LoRA 输出对应 checkpoint 地址

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
).eval()

# 加载 LoRA 权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

def generate_response(prompt):
    """
    根据用户输入生成林黛玉风格的回复
    """
    messages = [
        {"role": "system", "content": "假设你是红楼梦的角色--林黛玉。"},
        {"role": "user", "content": prompt}
    ]

    # 将对话模板应用到输入
    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 将输入转换为模型输入格式
    model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')

    # 生成回复
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512  # 控制生成的最大长度
    )

    # 解码生成的回复
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def main():
    """
    主函数，支持用户输入并调用模型生成回答
    """
    print("欢迎使用林黛玉对话模型！输入 '退出' 结束对话。")
    while True:
        # 获取用户输入
        prompt = input("你：")
        if prompt.lower() in ['退出', 'exit', 'quit']:
            print("对话结束，再见！")
            break

        # 调用模型生成回复
        response = generate_response(prompt)

        # 打印林黛玉的回复
        print("林黛玉：", response)

if __name__ == "__main__":
    main()