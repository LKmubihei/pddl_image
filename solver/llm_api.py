from openai import OpenAI
import json

# 强烈建议: 不要将 API Key 硬编码在代码中。
# 更好的做法是使用环境变量。
# 例如: client = OpenAI(api_key=os.environ.get("BIANXIE_API_KEY"), ...)
client = OpenAI(api_key="sk-Akxfqns3yu893SxhzeAM4nzM528IcJoI107ZiroMYiaGKb1W", base_url="https://toapis.com/v1")

# 模型名称映射表
model_name_map = {
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gpt-5.4": "gpt-5.4",
}

def get_gpt_response(messages, model_choice="gpt-5.4", tokenizer=None, local_model=None):
    """
    根据 model_choice 调用外部接口或本地模型生成回复。
    """
    # 检查 model_choice 是否在支持的 API 模型列表中
    if model_choice in ['gpt-5.4', "gemini-2.5-pro"]:
        
        # 修正1：从映射表中获取正确的模型名称，而不是硬编码
        if model_choice not in model_name_map:
            print(f"错误：模型 '{model_choice}' 在 model_name_map 中未定义。")
            return "模型选择错误。"
        
        mapped_model = model_name_map[model_choice]
        print(f"--- 正在调用 API，使用模型: {mapped_model} ---")
        
        try:
            response = client.chat.completions.create(
                model=mapped_model,
                messages=messages,
            )
            
            # 修正2：在访问 .choices 之前，先检查响应类型
            # 官方 openai 库返回的是一个对象，我们可以直接访问属性
            # 但如果 API 返回的是其他内容（如 HTML 字符串），这里就会出错
            
            # 调试：打印原始响应的类型和内容
            print("--- API 原始响应类型 ---")
            print(type(response))
            print("--- API 原始响应内容 ---")
            # 如果是对象，打印其字典表示，方便查看
            if hasattr(response, 'model_dump_json'):
                 print(response.model_dump_json(indent=2))
            else:
                 print(response)

            return response.choices[0].message.content

        except Exception as e:
            # 修正3：捕获所有可能的异常，并打印错误信息和原始响应
            print("\n--- 发生错误！ ---")
            print(f"错误类型: {type(e)}")
            print(f"错误信息: {e}")
            print("这很可能是因为 API Key 无效或模型名称错误，导致返回了非预期的内容（如HTML网页）。")
            return f"API 调用失败: {e}"

    else:
        # 本地模型逻辑保持不变
        print("--- 正在使用本地模型 ---")
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(local_model.device)
        generated_ids = local_model.generate(
            **model_inputs,
            max_new_tokens=4096
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return response

# --- 开始执行 ---
# 使用一个在 API 列表中的模型进行测试
# ans = get_gpt_response([{"role": "user", "content": "你模型的名称叫什么,版本是多少，发布时间是什么时候"}], model_choice="gpt-4o")
ans = get_gpt_response([{"role": "user", "content": "用Python写一个快速排序算法"}], model_choice="gpt-5.4")
print("\n--- 最终答案 ---")
print(ans)
