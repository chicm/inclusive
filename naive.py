import pandas as pd

import settings

top_n = 3

tuning_labels = pd.read_csv(settings.VAL_LABEL_FILE, names=['id', 'labels'], index_col=['id'])

# calculate top_n most popular labels
predicted = ' '.join(
    tuning_labels['labels']
    .str
    .split()
    .apply(pd.Series)
    .stack()
    .value_counts()
    .head(top_n)
    .index
    .tolist()
)

print('{} most popular labels are: {}'.format(top_n, predicted))

submission = pd.read_csv(settings.STAGE1_SAMPLE_SUB, index_col='image_id')

# tuning table is part of submission.csv
submission.index.isin(tuning_labels.index).sum()

# use most popular labels as a prediction unless the correct labels are provided
submission['labels'] = predicted
submission.update(tuning_labels)

submission.to_csv('sub1_naive.csv')