import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from extract_model import extract_metrics
from visualizer import visualize_metrics
import json


# 设置 DashScope API 密钥
os.environ["DASHSCOPE_API_KEY"] = "sk-d2a633b43aa448f4bc2f19fb092500a5"

# 初始化模型
def create_llm():
    return ChatOpenAI(
        model="qwen-max",
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
您可在文本框直接输入你的作文标题和内容，进行批改打分。
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
| 作文题目        | 我的中国之旅        |
|-----------------|---------------------|

| 评分维度      | 分值（满分10分） | 评语说明                                     |
|---------------|-----------------|--------------------------------------------|
| 题目贴合性    | 7               | 题目和文章内容比较匹配，可以表达主题             |
| 语言准确性    | 7               | 大部分句子正确，但有几个错误                   |
| 内容完整性      | 9               | 内容真实、生动，讲述了旅行经历，表达了感情         |
| 句式多样性    | 6               | 主要用简单句，复杂句比较少                     |
| 其他  | 8               | 理解中国文化，文章逻辑清楚                    |
| **总分**      | **37/50**       | 表现不错，但还可以更好                        |

| 优秀句子展示 | 评语                                   |
|-------------|--------------------------------------|
| 原句：我第一次坐高铁，觉得又快又舒服。<br>点评：句子流畅，感情真实。 | 用“又……又……”这样的句型，表达得很清楚。 |
| 原句：北京有很多历史遗迹，比如故宫和天坛，让我感受到了中国的古老文化。<br>点评：句子完整，表达了对文化的理解。 | 用例子说明观点，写得清楚明白。           |

| 需改进句子及问题 | 修改建议                                       |
|------------------|--------------------------------------------|
| 原句：我去北京的时候，天气很冷，我穿不多衣服。<br>问题：“穿不多衣服”这个说法不好。 | 建议改为：“我穿得不够暖和。”                      |
| 原句：我吃了火锅和烤鸭，两个都很好吃。<br>问题：“两个都”这个说法不太合适。 | 建议改为：“两种都很好吃。” 或 “都很好吃。”         |
| 原句：我觉得中国人很热情，他们对我笑。<br>问题：句子之间缺乏连接，不够流畅。 | 建议改为：“我觉得中国人很热情，他们经常对我微笑。”        |

| 综合评语 |
|----------|
| 这篇作文真实有趣，内容围绕“中国之旅”展开，表达了作者对中国文化和生活的感受。语言整体较通顺，但句子的丰富性和语法准确性还可以提高。建议多读中文文章，学习常用表达，并尝试使用更多不同的句型。总体来说，表现不错，要继续努力！ |
"""

# 用户输入
st.subheader("✍️ 作文")
user_input = st.text_area("请输入作文内容：", 
placeholder="""《我的梦想》
每个人都有自己的梦想，我也有一个属于自己的梦想。我的梦想是成为一名老师。
老师是一个非常神圣的职业，他们像一盏明灯，为我们照亮前进的道路；像一位园丁，细心地培育我们这些小花小草；更像我们的朋友，陪伴我们一起成长。每当我看到老师站在讲台上认真讲课的样子，我就特别羡慕，也想像他们一样，把知识传授给更多的人。
我知道，要实现这个梦想并不容易。首先，我要努力学习，尤其是语文、数学和英语这三门主科，因为它们是我未来学习的基础。其次，我要多读书，开阔自己的眼界，增长见识。最后，我要锻炼自己的表达能力，这样将来才能更好地与同学们交流。
虽然我现在还是一名小学生，离梦想还有很远的距离，但我相信只要我坚持不懈地努力，总有一天我会实现自己的梦想，成为一名优秀的老师！
                          """, 
                          height=200)

# 响应区域
if st.button("生成回复") and user_input:
    llm = create_llm()
    with st.spinner("正在批改作文..."):
        try:
            # 调用大模型生成作文批改结果（包含评分表格等内容）
            response = llm.invoke([
                {"role": "system", "content": default_prompt},
                {"role": "user", "content": user_input}
            ])
            st.subheader("✒️ 批改结果")
            st.success(response.content)
            
            # 使用大模型提取评分信息
            extraction_prompt = """
                            请从以下作文批改结果中提取出各评分维度的得分，并以纯JSON格式返回。键为评分维度，值为对应得分（数值范围均为0~10）。请只返回JSON，不要其他任何解释。包括以下健['题目贴合性', '语言准确性','内容完整性',  '句式多样性', '其他']
                            作文批改结果：
                            {response_content}
                            """
            extraction_response = llm.invoke([
                {"role": "system", "content": extraction_prompt.format(response_content=response.content)}
            ])

            try:
                scores = json.loads(extraction_response.content.strip())
            except Exception as e:
                st.error("评分数据提取失败，请检查模型输出格式。")
                scores = {}
            
            if scores:
                # 调用已有的 visualize_metrics 函数生成柱状图
                fig = visualize_metrics({k: {"score": v} for k, v in scores.items()})
                st.subheader("📊 评分可视化")
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"批改失败：{e}")