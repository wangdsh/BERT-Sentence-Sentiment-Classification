import torch
import numpy as np
import pandas as pd
import transformers as ppb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

# 导入数据
df = pd.read_csv('train.tsv', delimiter='\t', header=None)
batch_1 = df[:2000] # 使用一部分数据
print(batch_1[1].value_counts())

# 加载预训练BERT模型
model_class = ppb.DistilBertModel
tokenizer_class = ppb.DistilBertTokenizer
pretrained_weights = '/data/wangdsh/bert_pre-trained_models/distilbert-base-uncased'

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
model = model.cuda()

# 准备数据
tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
print('padded:', np.array(padded).shape)

attention_mask = np.where(padded != 0, 1, 0)
print('attention_mask:', attention_mask.shape)

# 模型
input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)
input_ids = input_ids.cuda()
attention_mask = attention_mask.cuda()

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask)

features = last_hidden_states[0][:,0,:].cpu().numpy()
labels = batch_1[1]

# 分训练集/测试集
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

# 分类
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

# 评估
print(lr_clf.score(test_features, test_labels))

from sklearn.dummy import DummyClassifier
clf = DummyClassifier()

scores = cross_val_score(clf, train_features, train_labels)
print("Dummy classifier score: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

