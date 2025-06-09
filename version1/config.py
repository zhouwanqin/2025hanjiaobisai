import os

# 配置设置
class Config:
    # API 密钥
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "your_default_api_key_here")
    
    # 模型参数
    MODEL_NAME = "qwen-plus"
    MAX_TOKENS = 8192
    OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # 可视化设置
    CHART_COLOR = "#1f77b4"  # 默认柱形图颜色
    CHART_TITLE = "作文评分可视化"
    CHART_X_LABEL = "评分维度"
    CHART_Y_LABEL = "分值"