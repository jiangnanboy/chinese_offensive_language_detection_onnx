import torch
from transformers.models.bert import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('../weights')
model = BertForSequenceClassification.from_pretrained('../weights')
model.eval()

def model2onnx():
    pass


def detect(text):
    model_input = tokenizer(text, return_tensors="pt", padding=True)
    model_output = model(**model_input, return_dict=False)

    prediction = torch.argmax(model_output[0].cpu(), dim=-1)
    prediction = [p.item() for p in prediction]
    return prediction

if __name__ == '__main__':
    text = ['你就是个傻逼！', '黑人很多都好吃懒做，偷奸耍滑！', '男女平等，黑人也很优秀。']
    prediction = detect(text)
    print(prediction) # --> [1, 1, 0] (0 for Non-Offensive, 1 for Offenisve)

