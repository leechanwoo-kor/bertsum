import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
import datetime

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(USE_CUDA, device)

def main():
    # Warining 방지
    logging.set_verbosity_error()

    # 파일 불러오기
    path = "reviews_sample2.csv"
    df = pd.read_csv(path, encoding="cp949")
    data_X = df['Text'].values   # 문장 컬럼
    labels = df['Sentiment'].values     # 라벨 컬럼
    print("### 데이터 ###")
    print("문장")
    print(data_X[:5])
    print("라벨")
    print(labels[:5])

    # 토큰화 (do_lower_case = 대소문자 변환여부)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)  # bert base
    tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased', do_lower_case=True)  # mobile bert
    input_ids = [tokenizer.encode(x, add_special_tokens=True) for x in data_X]  # [CLS], [SEP] 추가 및 정수화
    print("\n\n### 토큰화 ###")
    print(input_ids[0])

    # 패딩 (maxlen = 길이 최대값, truncating = 길이 초과시 삭제, padding = 뒤를 0으로 패딩)
    input_ids = tf.keras.utils.pad_sequences(input_ids,
                                             maxlen=128,
                                             dtype="long",
                                             truncating="post",
                                             padding="post")
    print("\n\n### 패딩 ###")
    print(input_ids[0])

    # 어텐션 마스크 생성
    attention_mask = [[float(i > 0) for i in ids] for ids in input_ids]
    print("\n\n### 어텐션 마스크 ###")
    print(attention_mask[0])

    # 데이터 분리
    train, validation, train_y, validation_y = train_test_split(input_ids, labels, test_size=0.1, random_state=2022)
    train_masks, validation_masks, _, _ = train_test_split(attention_mask, labels, test_size=0.1, random_state=2022)

    # 배치
    # batch_size = 16  # bert base
    batch_size = 8  # mobile bert
    train_inputs = torch.tensor(train).to(device)
    train_labels = torch.tensor(train_y).to(device)
    train_masks = torch.tensor(train_masks).to(device)
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_inputs = torch.tensor(validation).to(device)
    validation_labels = torch.tensor(validation_y).to(device)
    validation_masks = torch.tensor(validation_masks).to(device)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # 모델 생성
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # bert base
    model = MobileBertForSequenceClassification.from_pretrained('mobilebert-uncased', num_labels=2).to(device)  # mobile bert

    # 최적화 알고리즘 설정 (AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # epoch 및 scheduler 설정
    epoch = 4
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * epoch)

    for e in range(0, epoch):
        # 훈련
        print('\n\nEpoch {:} / {:}'.format(e + 1, epoch))
        print('Training')
        t0 = time.time()    # 시간 초기화
        total_loss = 0      # 총 loss 초기화
        model.train()       # 모델 - 훈련모드
        for step, batch in enumerate(train_dataloader):
            # step 50 단위 정보
            if step % 50 == 0 and not step == 0:
                elapsed_rounded = int(round((time.time() - t0)))
                elapsed = str(datetime.timedelta(seconds=elapsed_rounded))
                print('- Batch {:>5,} of {:>5,}, Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            # batch 데이터 추출
            batch_ids, batch_mask, batch_labels = tuple(t for t in batch)
            # gradient 초기화
            model.zero_grad()
            # Forward
            outputs = model(batch_ids, token_type_ids=None, attention_mask=batch_mask, labels=batch_labels)
            # loss
            loss = outputs.loss
            total_loss += loss.item()
            if step % 10 == 0 and not step == 0:
                print("step: {:}, loss: {:.2f}".format(step, loss.item()))
            # Backward
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 최적화(AdamW)
            optimizer.step()
            # 학습률 조정
            scheduler.step()

        # 총 loss 계산
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average training loss: {0:.2f}".format(avg_train_loss))
        print("Training epoch took: {:}".format(str(datetime.timedelta(seconds=(int(round(time.time() - t0)))))))

        # 검증
        print('\nValidation')
        t0 = time.time()    # 시간 초기화
        model.eval()        # 모델 - 검증모드
        eval_loss, eval_accuracy, eval_steps, eval_examples = 0, 0, 0, 0    # 검증값 초기화
        for batch in validation_dataloader:
            # batch 데이터 추출
            batch_ids, batch_mask, batch_labels = tuple(t for t in batch)
            # gradient 무시
            with torch.no_grad():
                # Forward
                outputs = model(batch_ids, token_type_ids=None, attention_mask=batch_mask)
            # logit
            logits = outputs[0]
            logits = logits.cpu().numpy()
            label_ids = batch_labels.cpu().numpy()
            # accuracy
            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            eval_accuracy_temp = np.sum(pred_flat == labels_flat) / len(labels_flat)
            eval_accuracy += eval_accuracy_temp
            eval_steps += 1
        print("Accuracy: {0:.2f}".format(eval_accuracy / eval_steps))
        print("Validation took: {:}".format(str(datetime.timedelta(seconds=(int(round(time.time() - t0)))))))

    # 모델 저장
    print('\nSave Model')
    save_path = 'bert_model'
    model.save_pretrained(save_path + '.pt')
    # torch.save(model, 'model.pt')
    print("\nfinish")


if __name__ == "__main__":
    main()