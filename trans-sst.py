import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 定义情感预测函数
def predict_sentiment(text):
    """
    对单个文本进行情感预测，返回预测类别。
    """
    # 分词并生成 PyTorch 张量
    inputs = bart_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 模型前向传播
    with torch.no_grad():  # 禁用梯度计算
        outputs = bart_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # 获取分类分数

    # 返回预测类别
    return logits.argmax(dim=1).item()

if __name__ == '__main__':
    # 1. 读取测试数据
    test_file = './SST/test.tsv'
    df = pd.read_csv(test_file, sep='\t', header=None, names=['label', 'text'])

    # 提取评论文本和标签
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # 2. 加载 BART 分词器和模型
    bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    bart_model = AutoModelForSequenceClassification.from_pretrained("valhalla/bart-large-sst2", num_labels=2)
    bart_model.eval()  # 设置模型为评估模式

    # 3. 对测试集进行预测并计算准确率
    predict_true = 0
    for text, true_label in zip(texts, labels):
        pred_label = predict_sentiment(text)  # 调用预测函数
        if pred_label == true_label:         # 比较预测值与真实值
            predict_true += 1

    # 4. 输出预测结果和准确率
    accuracy = float(predict_true / len(texts) * 100)
    print(f"测试集总样本数: {len(texts)}")
    print(f"预测正确的数量: {predict_true}")
    print(f"正确率为: {accuracy:.2f}%")
