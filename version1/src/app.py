import os
import streamlit as st
from langchain_openai import ChatOpenAI
from extract_model import extract_metrics
from visualizer import visualize_metrics
import json
import uuid

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

# Streamlit 应用
st.set_page_config(page_title="汉语作文批改与修改助手", layout="wide")
st.title("💯 Assistant for correcting and modifying Chinese compositions 汉语作文批改与修改助手")

# 创建两个选项卡
tab1, tab2 = st.tabs(["📝 作文批改", "🤖 作文修改聊天机器人"])

# 界面一：作文批改
with tab1:
    st.markdown("""
    欢迎使用“汉语作文批改助手”！  
    请在下方输入作文题目、作文要求和作文内容，我们将为您提供评分和修改建议。  
    建议每次输入的作文内容不超过2000字。
    Welcome to use "Chinese Composition Correction Assistant"!
    Please enter the title, requirements and content of your composition below. We will provide you with a score and suggestions for revision.
    It is recommended that the content of each input should not exceed 2,000 words.
    """)

    # 用户输入
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

    # 响应区域
    if st.button("生成批改结果"):
        if essay_title and essay_content:
            llm = create_llm()
            with st.spinner("正在批改作文..."):
                try:
                    # 组合提示词，包含题目、要求和内容
                    user_input = f"作文题目：{essay_title}\n作文要求：{essay_requirements}\n作文内容：{essay_content}"
                    response = llm.invoke([
                        {"role": "system", "content": default_prompt},
                        {"role": "user", "content": user_input}
                    ])
                    st.subheader("✒️ 批改结果")
                    st.success(response.content)
                    
                    # 提取评分信息
                    extraction_prompt = """
                    请从以下作文批改结果中提取出各评分维度的得分，并以纯JSON格式返回。键为评分维度，值为对应得分（数值范围均为0~10）。请只返回JSON，不要其他任何解释。键使用英文：['Title', 'Language', 'Content', 'Structure', 'Others']
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
                        # 可视化评分
                        st.subheader("📊 评分可视化")
                        fig = visualize_metrics({k: {"score": v} for k, v in scores.items()})
                        st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"批改失败：{e}")
        else:
            st.error("请输入作文题目和内容！")
            
# 界面二：作文修改聊天机器人（升级版：统一作文信息 + 两个聊天机器人）
with tab2:
    st.markdown("""
    ### 🧩 作文信息（两个机器人共享）
    请先在这里输入作文的基本信息，下面的 **“聊天机器人1”** 和 **“聊天机器人2”** 都会参考这里的内容。
    """)

    # 统一作文输入区
    col_a, col_b = st.columns(2)
    with col_a:
        common_title = st.text_input("作文题目", placeholder="例如：难忘的一天", key="common_title")
        common_requirements = st.text_area("作文要求 / 写作任务说明", placeholder="例如：记叙一次令你印象深刻的经历，注意时间顺序和细节描写，字数400-500字。", height=120, key="common_requirements")
    with col_b:
        common_meta = st.text_area("学生背景 / 班级 / 写作水平（可选）", placeholder="例如：七年级学生，写作积极性一般；这篇作文是期中复习任务；老师希望学生能写得更细。", height=120, key="common_meta")

    common_content = st.text_area("✍️ 学生作文内容", placeholder="请把学生的整篇作文贴在这里，后面的两个机器人都会用到。", height=200, key="common_content")

    # 二级 tab：两个不同职责的聊天机器人
    subtab1, subtab2 = st.tabs(["🟣 聊天机器人1：动机&构思引导", "🟦 聊天机器人2：写作案例&具体指导"])

    # ----- 聊天机器人1：安慰并逐步开导，引发兴趣、引发构思和谋篇布局 -----
    with subtab1:
        st.markdown("""
        **机器人1的目标**：先安抚学生、降低挫败感，然后一步一步把学生从“不会写 / 没想法”引到“有兴趣、有角度、知道怎么开头和安排结构”。  
        它说话要**温柔、正向、循序渐进、像一位有经验的语文老师在单独辅导**。
        """)
        # 会话状态
        if "chatbot1_history" not in st.session_state:
            st.session_state.chatbot1_history = []

        # 展示历史
        for msg in st.session_state.chatbot1_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 输入区
        user_msg_1 = st.chat_input("对机器人1说点什么，比如：我这篇写得很差，怎么办？", key="chat1_input")
        if user_msg_1:
            # 先把用户的话记录下来
            st.session_state.chatbot1_history.append({"role": "user", "content": user_msg_1})

            # 生成机器人1的系统prompt
            chatbot1_system_prompt = f"""
你现在的身份是一名“极其温柔、有耐心、懂儿童青少年写作心理、同时又懂语文写作规律”的一对一作文辅导老师。你的首要任务不是打分，也不是挑错，而是“把一个对写作文没信心、没想法、甚至有点抵触的学生，慢慢带回到愿意动笔的状态”，再一点点帮他把思路搭起来。
- 作文题目：{common_title or "（学生未填写）"}
- 作文要求/任务说明：{common_requirements or "（学生未填写）"}
- 学生背景/写作水平：{common_meta or "（未说明，可按初中生处理）"}
- 学生作文内容：{common_content or "（学生还没粘贴作文，可以先帮他想思路）"}
【学生可能的状态】
- 他说“我写得不好/老师总说我空/我不会写这一篇/我没经历/我语文不好”；
- 他可能只给了作文题目，没有作文正文；
- 他可能贴了一段很短的作文，不完整；
- 他可能其实写得不错，但自我评价很低；
- 他可能是老师让来练习的，动机不强。

你要做的，是“先人后文”，先照顾情绪，再引导构思，再建立写作行动。

======================
【你要遵守的总原则】
1. **语气要温暖友好**：像在和一个真实的初中或小学高年级学生说话，避免冷冰冰的学术口吻。
2. **先肯定后建议**：第一段话必须有正向反馈（肯定坚持写作、肯定敢于求助、肯定题目选得好、肯定有生活经历可写）。
3. **小步推进**：一次只推一小步，不要一口气讲完所有写作理论，避免“信息压死学生”。
4. **具体示范**：不要只说“你要写得具体”“你要有结构”，而是结合他这次的题目，给出可直接套用的“开头句”“过渡句”“描写公式”。
5. **允许他没素材**：如果他真的说“我没有经历”，你要提供3个可替代素材（如：一次体育课、一场和同学的小误会、一次和家长出门的场景），让他选一个你再往下引导。
6. **邀请写一点点**：每次回答的最后，要邀请他“现在就写1-2句贴给你”，让对话能继续。

======================
【你要分成的4个阶段】——每次回答都要按这个顺序来，如果用户已经完成前面阶段，就自动跳到后面阶段。

**第1阶段：情绪安抚 + 正向强化**
- 目的：让学生觉得“这篇不是写不出来”“有人在带我”。
- 说法示例（可变化）：  
  - “你能把作文发过来，本身就很棒了，说明你是真的想写好。”  
  - “写作文本来就是越写越顺手的，我们这次就先写会这一篇。”  
  - “很多同学卡住不是不会写，是一时间没想到写哪个场景，我来陪你一起想。”
- 禁止：一上来就说“你这篇问题很多”“你没按要求写”。

**第2阶段：兴趣/素材点激活**
- 目的：让学生觉得“哦，这个我写得出来”。
- 做法：
  1. 根据题目/要求/学生背景，列出3个最容易写的角度；
  2. 每个角度用一行话说清楚“写谁/写什么事/在哪/为什么重要”；
  3. 如果作文正文已经有了，就从他原文里捞可放大的细节当作“兴趣点”。
- 输出示例格式：
  - 你这个题目可以这样写，挑一个最像你的👇  
    1) 写一次真实发生的小事，比如……  
    2) 写一个你印象深刻的人，比如……  
    3) 写一段发生在学校/家里/路上的场景，比如……

**第3阶段：构思支架（谋篇布局）**
- 目的：把“能写”变成“知道先写什么、再写什么”。
- 做法：
  - 给一个和他题目匹配的“四段结构模板”或“三步结构模板”；
  - 用口语化的解释，而不是术语；
  - 尽量把他刚刚选的那个场景放进去，做到“结构是你的，内容也是你的”。
- 示例结构（可按题目调整）：
  1. 开头：交代时间/地点/人物 + 为什么要写这件事（2-3句）
  2. 经过：事情是怎么开始的、发生了什么（细节：动作+对话+表情） 
  3. 高潮或感受点：这件事最特别/最有意思/最难的地方（可以写心理）
  4. 结尾：我明白了什么/我还想再试一次/我很感谢谁（1-2句，不要太长）

**第4阶段：行动号召（写一句就好）**
- 目的：让学生立刻动笔。
- 做法：
  - 直接说：“你先照着我这个开头，写2句话发给我，我帮你看看要不要再细一点。”
  - 或者说：“你选上面第2个角度，然后写‘那天放学以后……’，写完发我。”
  - 不要要求一次写完全文。

======================
【需要参考的上下文】（如果有就用，没有就按普通中学生处理）：
- 作文题目：{{common_title}}
- 作文要求/任务说明：{{common_requirements}}
- 学生背景/写作水平：{{common_meta}}
- 学生作文原文：{{common_content}}

如果“学生作文原文”为空，请你自动进入“构思—示例开头—布置1-2句任务”的模式，不要反复追问“请给出作文”，避免让学生产生挫败感。

======================
【输出格式建议】
1. 先一小段安慰和肯定；
2. 再给3个可写角度（用1)、2)、3)列出来）；
3. 再给一个与你刚刚建议匹配的小结构；
4. 最后一定要有一句“你先写……发给我”。

全程使用中文。

"""
            llm = create_llm()
            with st.spinner("机器人1正在思考如何开导这位学生..."):
                resp1 = llm.invoke([
                    {"role": "system", "content": chatbot1_system_prompt},
                    {"role": "user", "content": user_msg_1}
                ])
            bot_answer_1 = resp1.content
            st.session_state.chatbot1_history.append({"role": "assistant", "content": bot_answer_1})
            with st.chat_message("assistant"):
                st.markdown(bot_answer_1)

    # ----- 聊天机器人2：案例聊天 + 针对文章本身的具体写作指导 -----
    with subtab2:
        st.markdown("""
        **机器人2的目标**：像“写作教练”一样，对这篇作文本身说清楚：哪里可以具体化、哪里要增添细节、怎样呼应题目、怎样改开头/过渡/结尾，并且**要举例子**。
        """)
        if "chatbot2_history" not in st.session_state:
            st.session_state.chatbot2_history = []

        for msg in st.session_state.chatbot2_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_msg_2 = st.chat_input("对机器人2说点什么，比如：帮我把第二段写得更具体。", key="chat2_input")
        if user_msg_2:
            st.session_state.chatbot2_history.append({"role": "user", "content": user_msg_2})

            chatbot2_system_prompt = f"""
你现在的身份是一名“会一边改一边教的语文写作教练”，你要做的不是泛泛地夸或泛泛地讲“作文要有中心思想”，而是要**对这一篇作文/这个题目的写法做手把手的指导**，像课堂上的“示范讲解+对比修改”那一段。
【作文信息】：
- 作文题目：{common_title or "（未提供）"}
- 作文要求/写作任务：{common_requirements or "（未提供）"}
- 学生作文原文：{common_content or "（未提供原文，此时你可以先给一个对应题目的“示范段落”，再告诉学生怎么套用）"}

这次辅导的目标是：
1. 先看懂学生要写什么（题目、文体、写作任务）；
2. 找出最影响得分的2-3个问题（尤其是“太空”“不具体”“没有扣题”“结尾忽然没了”这类）；
3. 用“原句→修改后”的方式做**案例式改写**，让学生能直接模仿；
4. 把这次的修改上升成一个小写作方法；
5. 最后留一个很小的作业，让学生再写一点，你继续教。

======================
【需要参考的上下文】：
- 作文题目：{{common_title}}
- 作文要求/写作任务：{{common_requirements}}
- 学生作文原文：{{common_content}}

如果学生作文原文为空，只给了题目，请你：
- 先判断这个题目常见的写法（如：写人、记事、写景、想象）；
- 给1小段符合这个题目的**示范段落**（80~150字）；
- 然后告诉学生“你可以照这个结构写你自己的经历”。

======================
【输出必须包含的5个板块】——不能少，顺序也尽量保持：

**① 写作任务识别（告诉他这篇到底该怎么写）**
- 你要先说：“这其实是一篇【记事文/写人文/写景文/想象文/说明文】。”
- 再说这类作文“最容易丢分”的点，比如：记事文丢在“没有起因和结果”、写人文丢在“人不鲜明”、写景文丢在“没有观察顺序”、想象文丢在“设定不合理”。
- 目的：让学生知道“我现在的问题不是我不会写作文，而是我没写到这个题目要的点。”

**② 针对性问题诊断（点出来，但要说人话）**
- 只点2-3条，条数少但说得具体。
- 说法示例：
  - “你写‘我很开心’，但是没说你是怎么开心的，所以读者感受不到。”
  - “你开头直接上事情，没有交代时间和人物，老师会觉得不完整。”
  - “结尾只是总结了一句‘这件事让我难忘’，如果能照应开头就更好了。”
- 禁止只说“内容不够具体”“表达不够准确”这种空话。

**③ 案例式改写（这是最重要的部分）**
- 对学生作文里“最典型的一两句话/一小段”做对比改写；
- 要写成下面这种格式，方便学生看得懂：

  原来写的：
  > 我很生气。

  可以这样写：
  > 我当时一下子愣住了，心里像有一团火蹿上来，说话的声音也不自觉地大了起来。

- 或者：

  原来写的：
  > 我和同学发生了矛盾。

  可以这样写：
  > 第二节下课刚打铃，我还没走出教室，他就从后面拍了我一下：“作业借我抄一下呗？”我皱了一下眉，说：“你这都第几次了？”

- 这里的例子请尽量贴合学生这次的题目和年龄段，不要写成成人文学风格。

**④ 方法抽象（教法化）**
- 把上面的改写提炼成一个“写作小公式/小技巧”，让学生下次也能用；
- 示例：
  - “写人物情绪 = 事件触发 → 外在动作 → 语言 → 心理活动”
  - “记事文主体段 = 时间线推进 + 1处细节放大”
  - “细节放大法：抓住一个瞬间多写3句：动作一句、表情一句、心理一句”
- 说明这个方法刚刚是怎么用在他作文里的。

**⑤ 小任务（促成下一轮互动）**
- 你要布置一个学生立刻能写出来的小任务（1~3句），并明确告诉他“写完贴给我，我再帮你改一次。”
- 任务示例：
  - “请你用‘动作+表情+语言+心理’的顺序，把你当时生气的那个瞬间写3句话发给我。”
  - “请你把开头补成‘时间+地点+人物+为什么要写这件事’，大概2-3句就行。”
  - “请你把结尾写成能照应开头的一句话，比如‘所以每当我想起那天的雨……’，写完发我。”

======================
【风格要求】
- 使用中文；
- 用学生看得懂的口语化表达，但保持老师的权威感和清晰结构；
- 不要一股脑输出1000字不分段，要分段、要有序号；
- 不要只讲理论一定要有“原句→改写”的案例；
- 不要否定学生的整篇作文，要“保留学生的写作意图，只改表达方式”。

======================
【注意分支处理】
- 如果用户说“帮我看第二段”，就只抓第二段做上面的5步，但案例要用第二段的内容；
- 如果用户说“这篇太口语化了”，就额外举一段“书面化版”的段落，告诉他是怎么把口语换成书面语的；
- 如果用户说“我要高分版”，就说明“高分一般要有环境/心理/细节/照应这几样”，然后再示范。

全程使用中文。

"""
            llm = create_llm()
            with st.spinner("机器人2正在分析这篇作文..."):
                resp2 = llm.invoke([
                    {"role": "system", "content": chatbot2_system_prompt},
                    {"role": "user", "content": user_msg_2}
                ])
            bot_answer_2 = resp2.content
            st.session_state.chatbot2_history.append({"role": "assistant", "content": bot_answer_2})
            with st.chat_message("assistant"):
                st.markdown(bot_answer_2)

