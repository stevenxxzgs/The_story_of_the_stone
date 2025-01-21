from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

# 模型路径
model1_path = '../model/LLM-Research/Meta-Llama-3___1-8B-Instruct'
lora1_path = './output/llama3_1_instruct_lora/checkpoint-720'  # 第一个模型的 LoRA 权重路径

model2_path = '../model/LLM-Research/Meta-Llama-3___1-8B-Instruct'
lora2_path = './output/llama3_1_instruct_lora_baoyu/checkpoint-174'  # 第二个模型的 LoRA 权重路径

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model1_path, trust_remote_code=True)

# 加载第一个模型到 GPU 0
model1 = AutoModelForCausalLM.from_pretrained(
    model1_path, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True
).eval()
model1 = PeftModel.from_pretrained(model1, model_id=lora1_path, device_map="cuda:0")

# 加载第二个模型到 GPU 1
model2 = AutoModelForCausalLM.from_pretrained(
    model2_path, device_map="cuda:1", torch_dtype=torch.bfloat16, trust_remote_code=True
).eval()
model2 = PeftModel.from_pretrained(model2, model_id=lora2_path, device_map="cuda:1")

def generate_response(model, tokenizer, prompt, role, device):
    """
    根据用户输入生成回复
    """
    messages = [
        {"role": "system", "content": f"假设你是{role}。"},
        {"role": "user", "content": prompt}
    ]

    # 将对话模板应用到输入
    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 将输入转换为模型输入格式，并移动到指定设备
    model_inputs = tokenizer([input_ids], return_tensors="pt").to(device)

    # 显式设置 attention_mask 和 pad_token_id
    attention_mask = model_inputs['attention_mask']
    pad_token_id = tokenizer.eos_token_id  # 使用 eos_token_id 作为 pad_token_id

    # 生成回复
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=attention_mask,  # 设置 attention_mask
        pad_token_id=pad_token_id,      # 设置 pad_token_id
        max_new_tokens=512              # 控制生成的最大长度
    )

    # 解码生成的回复
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def main():
    """
    主函数，让两个模型互相聊天
    """
    print("欢迎使用双模型对话系统！输入 '退出' 结束对话。")
    
    # 初始提示
    initial_prompt = "好久不见"
    print(f"初始提示：{initial_prompt}")

    # 设置对话轮次
    max_turns = 10
    current_turn = 0

    # 设置角色
    role1 = "红楼梦的角色--林黛玉"
    role2 = "红楼梦的角色--贾宝玉"

    # 初始化对话
    prompt = initial_prompt

    while current_turn < max_turns:
        # 模型1生成回复（使用 GPU 0）
        response1 = generate_response(model1, tokenizer, prompt, role1, device="cuda:0")
        print(f"{role1}：{response1}")

        # 检查是否达到停止条件
        if "退出" in response1.lower() or "exit" in response1.lower() or "quit" in response1.lower():
            print("对话结束，再见！")
            break

        # 模型2生成回复（使用 GPU 1）
        response2 = generate_response(model2, tokenizer, response1, role2, device="cuda:1")
        print(f"{role2}：{response2}")

        # 检查是否达到停止条件
        if "退出" in response2.lower() or "exit" in response2.lower() or "quit" in response2.lower():
            print("对话结束，再见！")
            break

        # 更新 prompt 为模型2的回复，以便下一轮对话
        prompt = response2

        # 增加对话轮次
        current_turn += 1

    if current_turn >= max_turns:
        print(f"已达到最大对话轮次 {max_turns}，对话结束。")

if __name__ == "__main__":
    main()