import utils as u
import time
import pandas as pd


def save_to_csv(YY, columns, sample_file='input/sample_submission.csv'):
    sample = pd.read_csv(sample_file)
    u.log('Sample', sample.shape)

    submission = pd.DataFrame(columns=columns)
    for i in len(columns):
        name = columns[i]
        if i == 0:
            submission[name] = sample[name]
        else:
            submission[name] = YY[:, i]
    submission.info()

    filename = 'submission-' + \
        time.strftime("%Y%m%d%H%M", time.gmtime()) + '.csv'
    submission.to_csv(filename, index=False)
