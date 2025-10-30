import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import uuid
from datetime import datetime
from langchain_openai import ChatOpenAI

# 导入本地模块
from extract_model import extract_metrics
from visualizer import visualize_metrics
from utils import (
    load_essays, save_essay,
    load_reviews, save_review,
    load_discussions, add_discussion_message
)

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

# 默认提示词 - 作文批改
default_prompt = """
你是一个汉语作文批改助手，专门为中级水平的外国汉语学习者批改作文。请根据用户提供的作文题目、作文要求和作文内容进行评分和提供修改建议。
若输入内容不是一篇完整的作文（例如仅包含零散句子或无关内容），请仅回复：“请刷新页面，输入正确的作文内容。”
请使用中文回答，并严格按照以下表格格式提供评分和建议。评分维度和分值需根据作文内容合理评估。
示例：
| 作文题目        | 我的中国之旅        |
|-----------------|---------------------|
| 评分维度      | 分值（满分10分） | 评语说明                                     |
|---------------|-----------------|--------------------------------------------|
| 题目贴合性    | 7               | 题目和文章内容比较匹配，可以表达主题             |
| 语言准确性    | 7               | 大部分句子正确，但有几个错误                   |
| 内容完整性    | 9               | 内容真实、生动，讲述了旅行经历，表达了感情         |
| 篇章逻辑性    | 6               | 整体篇章内容连贯，但有些逻辑词使用单调           |
| 其他          | 9               | 理解中国文化，文章逻辑清楚                    |
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

# 聊天机器人提示词 - 作文修改（英文回应）
chat_revision_prompt = """
Please make suggestions in English
You are a helpful essay revision assistant for intermediate-level Chinese language learners. Your task is to provide specific, actionable revision suggestions in English based on the student's request regarding their Chinese essay. The essay title, requirements, content, and the student's specific revision request are provided below. Focus on addressing the student's request (e.g., improving grammar, vocabulary, structure, or clarity). If the request is vague, provide general suggestions to enhance the essay's quality. Use the following format for your response:
**Revision Suggestions**  
- **Issue 1**: [Describe the specific issue based on the student's request or identified problem]  
  **Suggestion**: [Provide a clear, actionable suggestion to address the issue]  
- **Issue 2**: [Describe another specific issue, if applicable]  
  **Suggestion**: [Provide another actionable suggestion]  
- **General Advice**: [Provide overall advice for improving the essay, considering the title, requirements, and content]
**Essay Title**: {title}  
**Essay Requirements**: {requirements}  
**Essay Content**: {content}  
**Student's Revision Request**: {request}
Ensure your response is concise, encouraging, and tailored to the student's needs as an intermediate Chinese learner.
please answer in English
"""

# --- 用户身份初始化 ---
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "user_name" not in st.session_state:
    st.session_state.user_name = st.text_input("请输入你的名字", "匿名用户")
    if st.session_state.user_name != "匿名用户":
        st.rerun()

# --- 页面设置 ---
st.set_page_config(page_title="汉语作文批改与修改助手", layout="wide")
st.title("💯 Assistant for correcting and modifying Chinese compositions 汉语作文批改与修改助手")

# --- 创建选项卡 ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 作文批改", 
    "🤖 作文修改聊天机器人",
    "📚 作文集",
    "💬 讨论与互评"
])

# ============ Tab 1: 作文批改 ============
with tab1:
    st.markdown("""
    欢迎使用“汉语作文批改助手”！  
    请在下方输入作文题目、作文要求和作文内容，我们将为您提供评分和修改建议。  
    建议每次输入的作文内容不超过2000字。
    Welcome to use "Chinese Composition Correction Assistant"!
    Please enter the title, requirements and content of your composition below. We will provide you with a score and suggestions for revision.
    It is recommended that the content of each input should not exceed 2,000 words.
    """)
    
    st.subheader("✍️ 作文输入")
    essay_title = st.text_input("请输入作文题目：", placeholder="我的梦想")
    essay_requirements = st.text_area("请输入作文要求：", placeholder="例如：写一篇关于梦想的文章，300-500字，需包含个人目标和实现计划。", height=100)
    essay_content = st.text_area("请输入作文内容：", 
        placeholder="""《我的梦想》
每个人都有自己的梦想，我也有一个属于自己的梦想。我的梦想是成为一名老师。
老师是一个非常神圣的职业，他们像一盏明灯，为我们照亮前进的道路；像一位园丁，细心地培育我们这些小花小草；更像我们的朋友，陪伴我们一起成长。每当我看到老师站在讲台上认真讲课的样子，我就特别羡慕，也想像他们一样，把知识传授给更多的人。
我知道，要实现这个梦想并不容易。首先，我要努力学习，尤其是语文、数学和英语这三门主科，因为它们是我未来学习的基础。其次，我要多读书，开阔自己的眼界，增长见识。最后，我要锻炼自己的表达能力，这样将来才能更好地与同学们交流。
虽然我现在还是一名小学生，离梦想还有很远的距离，但我相信只要我坚持不懈地努力，总有一天我会实现自己的梦想，成为一名优秀的老师！
        """, 
        height=200)

    if st.button("生成批改结果"):
        if essay_title and essay_content:
            llm = create_llm()
            with st.spinner("正在批改作文..."):
                try:
                    user_input = f"作文题目：{essay_title}\n作文要求：{essay_requirements}\n作文内容：{essay_content}"
                    response = llm.invoke([
                        {"role": "system", "content": default_prompt},
                        {"role": "user", "content": user_input}
                    ])
                    st.subheader("✒️ 批改结果")
                    st.success(response.content)
                    
                    # 保存作文到作文集
                    essay_data = {
                        "title": essay_title,
                        "content": essay_content,
                        "requirements": essay_requirements,
                        "author_id": st.session_state.user_id,
                        "author_name": st.session_state.user_name
                    }
                    essay_id = save_essay(essay_data)
                    st.info(f"✅ 作文已保存至作文集，ID: {essay_id}")

                    # 提取评分并可视化
                    extraction_prompt = """
                    请从以下作文批改结果中提取出各评分维度的得分，并以纯JSON格式返回。键为评分维度，值为对应得分（数值范围均为0~10）。请只返回JSON，不要其他任何解释。键使用英文：['题目贴合性', '语言准确性', '内容完整性', '篇章逻辑性', '其他']
                    作文批改结果：
                    {response_content}
                    """
                    extraction_response = llm.invoke([
                        {"role": "system", "content": extraction_prompt.format(response_content=response.content)}
                    ])
                    try:
                        raw_scores = eval(extraction_response.content.strip())
                        scores = {k: {"score": v} for k, v in raw_scores.items()}
                    except Exception as e:
                        st.error("评分数据提取失败。")
                        scores = {}
                    
                    if scores:
                        st.subheader("📊 评分可视化")
                        fig = visualize_metrics(scores)
                        st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"批改失败：{e}")
        else:
            st.error("请输入作文题目和内容！")

# ============ Tab 2: 作文修改聊天机器人 ============
with tab2:
    st.markdown("""
    欢迎使用“作文修改聊天机器人”！  
    请先输入作文题目、要求和内容，然后通过聊天框提出您的修改要求（例如：改进语法、丰富词汇、优化结构等）。  
    机器人将以英文提供针对性的修改建议。
    Welcome to use the "Essay Revision Chatbot"!
    Please first enter the title, requirements and content of your composition, and then put forward your revision requests (such as: improving grammar, enriching vocabulary, optimizing structure, etc.) through the chat box.
    The robot will provide targeted modification suggestions in English.
    """)
    
    st.subheader("✍️ 提交作文信息")
    revision_title = st.text_input("请输入作文题目：", placeholder="我的梦想", key="revision_title")
    revision_requirements = st.text_area("请输入作文要求：", placeholder="例如：写一篇关于梦想的文章，300-500字，需包含个人目标和实现计划。", height=100, key="revision_requirements")
    revision_content = st.text_area("请输入作文内容：", 
        placeholder="请输入需要修改的作文内容...", 
        height=200, key="revision_content")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("💬 聊天")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    revision_request = st.chat_input("请输入您的修改要求（例如：请改进语法的准确性）")

    if revision_request and revision_title and revision_content:
        st.session_state.chat_history.append({"role": "user", "content": revision_request})
        
        llm = create_llm()
        with st.spinner("正在生成修改建议..."):
            try:
                user_input = chat_revision_prompt.format(
                    title=revision_title,
                    requirements=revision_requirements,
                    content=revision_content,
                    request=revision_request
                )
                response = llm.invoke([
                    {"role": "system", "content": user_input},
                    {"role": "user", "content": revision_request}
                ])
                st.session_state.chat_history.append({"role": "assistant", "content": response.content})
                st.rerun()
            except Exception as e:
                st.error(f"生成修改建议失败：{e}")
    elif revision_request and (not revision_title or not revision_content):
        st.error("请输入作文题目和内容后再提出修改要求！")

# ============ Tab 3: 作文集 ============
with tab3:
    st.subheader("📖 所有作文集")
    essays = load_essays()
    if not essays:
        st.info("暂无作文提交。")
    else:
        filter_type = st.radio("浏览方式", ["按题目", "按作者"])
        
        if filter_type == "按题目":
            titles = sorted(list(set(e["title"] for e in essays)))
            selected_title = st.selectbox("选择题目", titles)
            filtered = [e for e in essays if e["title"] == selected_title]
        else:
            authors = sorted(list(set(e["author_name"] for e in essays)))
            selected_author = st.selectbox("选择作者", authors)
            filtered = [e for e in essays if e["author_name"] == selected_author]

        for essay in filtered:
            with st.expander(f"📘 {essay['title']} - {essay['author_name']}"):
                st.write(f"**要求**：{essay.get('requirements', '无')}")
                st.write(f"**内容**：\n\n{essay['content']}")
                st.caption(f"提交时间：{essay['created_at']}")

# ============ Tab 4: 讨论与互评 ============
with tab4:
    st.subheader("👥 合作评价与讨论区")
    essays = load_essays()
    if not essays:
        st.info("暂无作文可评价。")
    else:
        essay_options = {e["id"]: f"{e['title']} ({e['author_name']})" for e in essays}
        selected_id = st.selectbox("选择一篇作文", options=list(essay_options.keys()), format_func=lambda x: essay_options[x])

        essay = next(e for e in essays if e["id"] == selected_id)
        st.markdown(f"### {essay['title']} - {essay['author_name']}")
        st.write(essay['content'])

        # 显示同伴评价
        st.markdown("#### 📊 同伴评价")
        reviews = load_reviews(selected_id)
        if reviews:
            df = pd.DataFrame(reviews)
            st.dataframe(df[["reviewer_name", "title_relevance", "language_accuracy", 
                           "content_completeness", "structure_logic", "other", "total_score", "comment"]])
            
            avg = df[["title_relevance", "language_accuracy", "content_completeness", 
                     "structure_logic", "other"]].mean().to_dict()
            fig = visualize_metrics({k.replace("_", "").replace("accuracy","准确性").replace("completeness","完整性").replace("logic","逻辑性").title(): v for k, v in avg.items()})
            st.pyplot(fig)
        else:
            st.info("暂无同伴评价。")

        # 提交评价
        with st.form("submit_review"):
            st.markdown("#### ✍️ 提交你的评价")
            c1, c2 = st.columns(2)
            with c1:
                title_rel = st.slider("题目贴合性", 0, 10, 5)
                lang_acc = st.slider("语言准确性", 0, 10, 5)
                cont_comp = st.slider("内容完整性", 0, 10, 5)
            with c2:
                struct_log = st.slider("篇章逻辑性", 0, 10, 5)
                other = st.slider("其他", 0, 10, 5)
            comment = st.text_area("评语")
            submit_review = st.form_submit_button("提交评价")

            if submit_review:
                total = title_rel + lang_acc + cont_comp + struct_log + other
                save_review({
                    "essay_id": selected_id,
                    "reviewer_id": st.session_state.user_id,
                    "reviewer_name": st.session_state.user_name,
                    "title_relevance": title_rel,
                    "language_accuracy": lang_acc,
                    "content_completeness": cont_comp,
                    "structure_logic": struct_log,
                    "other": other,
                    "total_score": total,
                    "comment": comment
                })
                st.success("评价提交成功！")
                st.rerun()

        # 讨论区
        st.markdown("#### 💬 讨论区")
        messages = load_discussions(selected_id)
        for msg in messages:
            st.markdown(f"**{msg['user_name']}**: {msg['message']} *({msg['created_at']})*")

        new_msg = st.text_input("发表你的看法", key=f"msg_{selected_id}")
        if st.button("发送", key=f"send_{selected_id}") and new_msg:
            add_discussion_message({
                "essay_id": selected_id,
                "user_id": st.session_state.user_id,
                "user_name": st.session_state.user_name,
                "message": new_msg
            })
            st.rerun()
