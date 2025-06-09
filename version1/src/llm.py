import os
from langchain.chat_models import ChatOpenAI

# 设置 DashScope API 密钥
os.environ["DASHSCOPE_API_KEY"] = "sk-cb534124d0c44e7fb12dcb0271715482"

# 初始化模型
def create_llm():
    return ChatOpenAI(
        model="qwen-plus",
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_tokens=8192,
        extra_body={"enable_thinking": False}
    )

# 处理用户输入并生成响应
def process_input(user_input):
    llm = create_llm()
    response = llm.invoke([{"role": "system", "content": default_prompt},
                           {"role": "user", "content": user_input}])
    return response.content

# 提取评分和建议
def extract_scores(response):
    # 解析响应内容，提取评分和建议
    # 这里需要根据实际的响应格式进行解析
    scores = {}
    # 示例解析逻辑
    # scores['题目贴合性'] = ...
    # scores['内容表达'] = ...
    # ...
    return scores

# 生成可视化数据
def generate_visualization_data(scores):
    # 将提取的评分数据转换为可视化所需的格式
    # 例如，可以将分数转换为列表或字典
    visualization_data = {
        "labels": list(scores.keys()),
        "values": list(scores.values())
    }
    return visualization_data