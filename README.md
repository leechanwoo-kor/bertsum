# Bertsum + Hugging Face🤗

**This code is for paper `Fine-tune BERT for Extractive Summarization`**(https://arxiv.org/pdf/1903.10318.pdf)

**!New: Please see our [full paper](https://arxiv.org/abs/1908.08345) with trained models**

**Hugging Face🤗의 Transformers BERT-Multilingual 모델을 이용하여 한글 문서 추출요약 Task를 수행할 수 있습니다.**

**데이터는 DACON의 뉴스 추출요약 데이터셋을 활용했습니다.**

**데이터의 전처리는 Konlpy의 Mecab(은전한닢)을 활용하여 형태소 분리를 진행하였습니다.**

**같이 업로드한 Jupyter Notebook을 이용하여 Train, Test로  됩니다.**

## How to Test ?

모델을 만들고 테스트하는 방법은 BertSum_test Jupyter Notebook을 활용하시면 됩니다.

아래의 기사를 입력시 

```
[이데일리 박지혜 기자] 조상호 더불어민주당 전 부대변인이 결국 “천안함 함장이 부하들을 수장시켰다”는 발언에 대해 사과했다.
조 전 부대변인은 9일 페이스북을 통해 “제 주변 분들의 애정 어린 권고가 있었다”고 운을 뗐다.
그러면서 “제 표현 중 혹여 순국한 46 용사의 유가족, 특히 아직도 시신조차 거두지 못한 6인의 유가족과 피해 장병들에게 고통스런 기억을 떠올리게 한 부분이 있다는 지적, 깊게 받아드린다”며 “상처로 떠올리신 유가족과 피해 장병께는 진심으로 사죄드린다”고 전했다.
이어 “다시 한 번 46 용사의 명복을 빈다”고 덧붙였다.
조상호 더불어민주당 전 부대변인 (사진=채널A 방송 캡처)
앞서 조 전 부대변인은 지난 7일 오후 종합편성채널 채널A ‘뉴스톱10’에서 “최원일 전 함장이라는 예비역 대령, 그분도 승진했다. 그런데 그분은 그(처우 관련) 말을 할 자격이 없다”며 “최 전 함장이 그때 당시 생때같은 자기 부하들을 다 수장시켜 놓고 이후에 제대로 된 책임이 없었다”고 말했다.
당시 방송하던 진행자와 다른 출연자들이 최 함장이 수장시킨 건 아니라며 발언을 제지했지만, 조 전 부대변인은 주장을 굽히지 않았다. 이후 자신의 페이스북에도 “도대체 뭐가 막말이냐”는 글을 올렸다.
그는 또 “작전에 실패한 군인은 몰라도 경계에 실패한 군인은 용서할 수 없다는 군사 격언이 있다”며 “심지어 당시는 한미연합훈련 중이었다. 하지만 함장 지휘관이 폭침으로 침몰 되는데도 뭐에 당했는지도 알지 못 했다”고 했다.
그러면서 “결국 46명의 젊은 목숨을 잃었다. 근데 함장이 책임이 없나”고 반문했다.
한편, 송영길 민주당 대표도 이날 조 전 부대변인의 발언에 항의한 최 전 함장과 유가족에게 “죄송하다”고 사과했다.
최 전 함장과 천안함 유가족은 이날 여의도 국회를 찾아 송 대표를 면담하고 공식 사과를 요구했다.
송 대표는 이 자리에서 “당 대표로서 죄송하다”며 “조 전 부대변인의 잘못된 언어 사용에 대해서 유감을 표명한다”고 밝힌 것으로 전해졌다.
고용진 수석대변인은 이날 면담 후 “조 전 대변인은 아무 당직 없이 당적만 보유한 분이며, 그분의 의견은 당과는 전혀 관련없는 의견”이라고 설명했다.
그는 “함장이 수장시켰다는 식으로 발언한 것은 사과해야 한다고 (조 전 대변인에게) 요구하고 있다”며 “김병주 의원도 참석했는데, 국방위에서 천안함 폭침이 분명히 북한 소행이라는 점을 말할 것”이라고 전했다.
박지혜 (noname@edaily.co.kr)
```
```
['[이데일리 박지혜 기자] 조상호 더불어민주당 전 부대변인이 결국 “천안함 함장이 부하들을 수장시켰다”는 발언에 대해 사과했다.',
 '그러면서 “제 표현 중 혹여 순국한 46 용사의 유가족, 특히 아직도 시신조차 거두지 못한 6인의 유가족과 피해 장병들에게 고통스런 기억을 떠올리게 한 부분이 있다는 지적, 깊게 받아드린다”며 “상처로 떠올리신 유가족과 피해 장병께는 진심으로 사죄드린다”고 전했다.',
 '당시 방송하던 진행자와 다른 출연자들이 최 함장이 수장시킨 건 아니라며 발언을 제지했지만, 조 전 부대변인은 주장을 굽히지 않았다. 이후 자신의 페이스북에도 “도대체 뭐가 막말이냐”는 글을 올렸다.']
```
위와 같은 결과물을 얻을 수 있습니다.

결과물의 길이는
```
[list(filter(None, text.split('\n')))[i] for i in sum_list[0][:SENTENCE_LENGTH]]
```
위 코드에서 SENTENCE_LENGTH를 조정하면 됩니다.

```
input_data = txt2input(text)
sum_list = test(args, input_data, -1, '', None)
sum_list[0]
```
인풋데이터의 요약 순위를 확인할 수 있습니다.

## Refference
* https://velog.io/@dev_halo/BertSum
* https://github.com/raqoon886/KorBertSum
* https://github.com/huggingface/transformers
* https://huggingface.co/docs/transformers/model_doc/bert
* https://github.com/SOMJANG/Mecab-ko-for-Google-Colab
* https://dacon.io/competitions/official/235671/overview/description
* https://teddylee777.github.io/colab/colab-mecab
