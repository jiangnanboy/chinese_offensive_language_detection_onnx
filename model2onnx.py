from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from pathlib import Path
from transformers.convert_graph_to_onnx import convert
import onnx
import onnxruntime as ort

def load_model():
    tokenizer = BertTokenizer.from_pretrained("../weights")
    model = BertForSequenceClassification.from_pretrained("../weights")
    return model, tokenizer

'''
    Found input input_ids with shape: {0: 'batch', 1: 'sequence'}
    Found input token_type_ids with shape: {0: 'batch', 1: 'sequence'}
    Found input attention_mask with shape: {0: 'batch', 1: 'sequence'}
    Found output output_0 with shape: {0: 'batch'}
    Ensuring inputs are in correct order
    position_ids is not present in the generated input list.
    Generated inputs order: ['input_ids', 'attention_mask', 'token_type_ids']
'''
def convert2onnx(model, tokenizer, save_path):
    convert('pt', model, Path(save_path), 11, tokenizer)

def onnx_test(onnx_path):
    tokenizer = BertTokenizer.from_pretrained("../weights")
    sent1 = '你就是个傻逼！'
    sent2 = '黑人很多都好吃懒做，偷奸耍滑！'
    sent3 = '男女平等，黑人也很优秀。'

    tokenized_tokens = tokenizer(sent3, padding=True, max_length=512)
    input_ids = np.array([tokenized_tokens['input_ids']], dtype=np.int64)
    attention_mask = np.array([tokenized_tokens['attention_mask']], dtype=np.int64)
    token_type_ids = np.array([tokenized_tokens['token_type_ids']], dtype=np.int64)

    print('input_ids:{}'.format(input_ids))
    print('attention_mask:{}'.format(attention_mask))
    print('token_type_ids:{}'.format(token_type_ids))

    model = onnx.load(onnx_path)
    sess = ort.InferenceSession(bytes(model.SerializeToString()))
    result = sess.run(
        output_names=None,
        input_feed={"input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids}
    )[0][0]
    pred = softmax(result)
    label = np.argmax(result)
    print('----', pred, label)
    '''
    input_ids:[[ 101 4511 1957 2398 5023 8024 7946  782  738 2523  831 4899  511  102]]
    attention_mask:[[1 1 1 1 1 1 1 1 1 1 1 1 1 1]]
    token_type_ids:[[0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
    ---- [0.9962118  0.00378819] 0
    '''

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

if __name__ == '__main__':
    # model, tokenizer = load_model()
    # convert2onnx(model, tokenizer, '../onnx/model.onnx')
    onnx_test('../onnx/model.onnx')


