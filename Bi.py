# 下面我们用 Bi_LSTM 模型实现一个命名实体识别任务：

from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.tasks.labeling import BiLSTM_Model, BiLSTM_CRF_Model


# 加载内置数据集，此处可以替换成自己的数据集，保证格式一致即可
train_x, train_y = ChineseDailyNerCorpus.load_data('train')
test_x, test_y = ChineseDailyNerCorpus.load_data('test')
valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

model = BiLSTM_CRF_Model()
model.fit(train_x, train_y, valid_x, valid_y, epochs=1)

model.save("BiLSTM_CRF_Model")