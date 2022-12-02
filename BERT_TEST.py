from transformers import BertForSequenceClassification, BertTokenizer
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import tensorflow as tf
import torch
import numpy as np


def main():
    # 모델명
    model = 'bert_model.pt'

    # Test
    sentence = """ i love this movie """  # Test 문장
    logits = test_sentence([sentence], model)  # 함수
    print(logits)

    if np.argmax(logits) == 1:
        print("\nPositive sentence")
    elif np.argmax(logits) == 0:
        print("\nNegatvie sentence")


def convert_sentence(sentence):
    # 토큰화
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)  # bert base
    tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased', do_lower_case=True)  # mobile bert
    tokenized = [tokenizer.tokenize(s) for s in sentence]

    # 정수화 및 패딩
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized]
    input_ids = tf.keras.utils.pad_sequences(input_ids,
                                             maxlen=128,
                                             dtype="long",
                                             truncating="post",
                                             padding="post")

    # 어텐션 마스크 생성
    attention_mask = [[float(i > 0) for i in ids] for ids in input_ids]

    # 배치
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_mask)

    return inputs, masks


def test_sentence(sentence, model):
    # 모델 불러오기
    # model = BertForSequenceClassification.from_pretrained(model)  # bert base
    model = MobileBertForSequenceClassification.from_pretrained(model)  # mobile bert
    model.eval()

    # 문장 불러오기 및 배치
    inputs, masks = convert_sentence(sentence)
    batch_input_ids = inputs
    batch_input_mask = masks

    # gradient 무시
    with torch.no_grad():
        # Forward
        outputs = model(batch_input_ids, token_type_ids=None, attention_mask=batch_input_mask)

    # loss
    logits = outputs[0]

    # numpy화
    logits = logits.numpy()

    return logits


if __name__ == "__main__":
    main()