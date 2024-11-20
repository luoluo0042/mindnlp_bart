import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# 定义预测函数
def predict_qnli(question, sentence):
    # 分词并生成 PyTorch 张量
    inputs = tokenizer(question, sentence, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 模型前向传播
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # 获取分类分数

    # 获取预测类别
    return logits.argmax(dim=1).item()

if __name__ == '__main__':
    # 加载测试集
    test_file = './QNLI/dev.tsv'
    df = pd.read_csv(test_file, sep='\t', header=0, names=['idx', 'question', 'sentence', 'label'])

    # 删除包含 NaN 的行
    df = df.dropna(subset=['label'])

    # 检查并过滤无效标签
    label_map = {'entailment': 0, 'not_entailment': 1}
    valid_data = df[df['label'].isin(label_map.keys())]

    # 提取有效数据
    questions = valid_data['question'].tolist()
    sentences = valid_data['sentence'].tolist()
    labels = [label_map[label] for label in valid_data['label']]

    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    model = AutoModelForSequenceClassification.from_pretrained("ModelTC/bart-base-qnli", num_labels=2)
    model.eval()  # 设置为评估模式，避免计算梯度

    # 对测试集进行预测并计算准确率
    predict_true = 0
    for question, sentence, true_label in zip(questions, sentences, labels):
        pred_label = predict_qnli(question, sentence)
        if pred_label == true_label:
            predict_true += 1

    # 输出预测结果和准确率
    accuracy = float(predict_true / len(questions) * 100)
    print(f"测试集总样本数: {len(questions)}")
    print(f"预测正确的数量: {predict_true}")
    print(f"准确率为: {accuracy:.2f}%")
