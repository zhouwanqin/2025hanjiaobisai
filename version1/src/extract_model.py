import pandas as pd

class ExtractModel:
    def __init__(self, response):
        self.response = response
        self.scores = {}
        self.excellent_sentences = []
        self.improvement_suggestions = []

    def extract_scores(self):
        # 提取评分和评语
        lines = self.response.split('\n')
        for line in lines:
            if '评分维度' in line:
                continue
            if '总分' in line:
                continue
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 4:
                    dimension = parts[1].strip()
                    try:
                        score = int(parts[2].strip())
                    except ValueError:
                        score = 0
                    comment = parts[3].strip()
                    self.scores[dimension] = {'score': score, 'comment': comment}

    def extract_excellent_sentences(self):
        # 提取优秀句子
        lines = self.response.split('\n')
        for line in lines:
            if '优秀句子展示' in line:
                continue
            if '需改进句子及问题' in line:
                break
            if '原句：' in line:
                sentence = line.split('原句：')[1].split('<br>')[0].strip()
                self.excellent_sentences.append(sentence)

    def extract_improvement_suggestions(self):
        # 提取修改建议
        lines = self.response.split('\n')
        for line in lines:
            if '需改进句子及问题' in line:
                continue
            if '综合评语' in line:
                break
            if '问题：' in line:
                suggestion = line.split('问题：')[1].split('<br>')[0].strip()
                self.improvement_suggestions.append(suggestion)

    def process(self):
        self.extract_scores()
        self.extract_excellent_sentences()
        self.extract_improvement_suggestions()
        return {
            'scores': self.scores,
            'excellent_sentences': self.excellent_sentences,
            'improvement_suggestions': self.improvement_suggestions
        }

# 添加新函数供外部调用
def extract_metrics(response):
    model = ExtractModel(response)
    result = model.process()
    return result['scores']