import pandas as pd
from mindnlp.transformers import AutoTokenizer, AutoModelForSequenceClassification
import mindspore
import mindspore.ops as ops

# 定义情感预测函数
def predict_sentiment(text):
    # 分词并生成 MindSpore 张量
    inputs = bart_tokenizer(text, return_tensors="ms", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 模型前向传播
    outputs = bart_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # 获取分类分数

    # 获取预测类别
    return logits.argmax(axis=1).asnumpy()[0]

if __name__ == '__main__':
    # 1. 读取测试数据
    test_file = './SST/test.tsv'
    df = pd.read_csv(test_file, sep='\t', header=None, names=['label', 'text'])

    # 提取评论文本和标签
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # 2. 加载 BART 分词器和模型
    bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    bart_model = AutoModelForSequenceClassification.from_pretrained("valhalla/bart-large-sst2", num_labels=2)  # 可换为微调的 SST2 模型



    # # 2. 加载 BART 分词器和模型
    # bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    # bart_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")  # 可换为微调的 SST2 模型

    # 3. 对测试集进行预测并计算准确率
    predict_true = 0
    for text, true_label in zip(texts, labels):
        pred_label = predict_sentiment(text)  # 调用预测函数
        if pred_label == true_label:         # 比较预测值与真实值
            predict_true += 1

    # 4. 输出预测结果和准确率
    accuracy = float(predict_true / len(texts) * 100)
    print(f"预测正确的数量: {predict_true}")
    print(f"正确率为: {accuracy:.2f}%")
