import os
import streamlit as st
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

# Streamlit 应用
st.set_page_config(page_title="汉语作文批改助手", layout="wide")
st.title("💯汉语作文批改助手")
st.markdown("""
欢迎使用“汉语作文批改助手”
您可在文本框直接输入你的作文内容，进行批改打分。
我们会返回直接对应的评分和修改建议。
建议每次输入不超过2000字。
""")

# 默认提示词
default_prompt = """
若我给的内容不是一篇作文，请你只回复“请刷新页面，输入正确的作文内容。”
你是一个汉语作文批改助手，你需要根据用户的作文内容进行评分和修改建议。
请使用中文回答。
请使用表格形式给出评分和建议，严格按照如下格式进行回答.
当我输入任何内容，针对我的内容，只回复形式如以下的表格。内容根据作文内容：
示例
| 作文题目         | 我的中国之旅 |
| 评分维度        | 分值（满分10分） | 评语说明 |
|-----------------|------------------|-----------|
| 内容表达        | 9                | 内容真实、生动，围绕旅行经历展开，有情感表达 |
| 语言准确性      | 7                | 大部分语法正确，但存在个别错误 |
| 句式多样性      | 6                | 主要使用简单句，复合句较少 |
| 篇章结构        | 8                | 结构清晰，段落过渡自然 |
| 文化与逻辑性    | 8                | 对中国文化有一定理解，逻辑通顺 |
| **总分**        | **38/50**        | 表现良好，具备继续提升空间 |

| 优秀句子展示 | 评语 |
|--------------|------|
| 原句：我第一次坐高铁，感觉非常快而且舒服。<br>点评：表达自然流畅，体现了真实感受。 | 运用了“非常……而且……”结构，表达清楚。 |
| 原句：北京有很多历史遗迹，比如故宫和天坛，它们让我感受到中国的古老文化。<br>点评：句型完整，体现文化理解。 | 使用举例法表达观点，结构合理。 |

| 需改进句子及问题 | 修改建议 |
|------------------|----------|
| 原句：我去北京的时候，天气很冷，我穿不多衣服。<br>问题：“穿不多衣服”表达不当。 | 建议修改为：“我穿得不够暖和。” |
| 原句：我吃了火锅和烤鸭，两个都很好吃。<br>问题：“两个都”用词不地道。 | 建议修改为：“两种都很好吃。”或“都很好吃。” |
| 原句：我觉得中国人很热情，他们对我笑。<br>问题：句子之间缺乏连接，略显突兀。 | 建议修改为：“我觉得中国人很热情，他们经常对我微笑。” |

| 综合评语 |  |
|----------|------|
| 本篇作文内容真实有趣，能够围绕“中国之旅”这一主题展开叙述，表达了作者对中国文化和生活的感受。语言整体较为通顺，但在句式多样性和语法准确性方面还有提升空间。建议多阅读中文文章，积累常用表达，并尝试使用更多复杂句型进行写作。总体表现不错，继续保持写作热情！ |
"""

# 用户输入
st.subheader("✍️ 作文")
user_input = st.text_area("请输入作文内容：", placeholder="《我的一天》 今天是快乐的一天，我去往了……", height=200)

# 响应区域
if st.button("生成回复") and user_input:
    llm = create_llm()
    with st.spinner("正在批改作文..."):
        try:
            response = llm.invoke([{"role": "system", "content": default_prompt},
                                   {"role": "user", "content": user_input}])
            st.subheader("✒️批改结果")
            st.success(response.content)
        except Exception as e:
            st.error(f"批改失败：{e}")
