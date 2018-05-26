# import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from sklearn import feature_extraction as fe
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams
import seaborn as sns
from textacy import preprocess as tap
# from stemming.porter2 import stem
import sys
import os
from utils import log
import processing as p
import extraction as e

sys.path.append(os.path.abspath("../"))


def load_data(raw=True):
    return load_train_data(raw), load_test_data(raw)


def load_train_data(raw=False):
    log('Loading train data...')
    data = pd.read_csv('input/train.csv')
    data = inputing_data(data)
    if raw is True:
        return data

    # data['shared_words'] = load_words_data(data)
    # data['shared_chars'] = load_chars_data(data)
    # data['count'] = load_count_data(data)
    # data['similar'] = load_similar_data(data)

    # qs = questions(data)
    # data['freqs1'] = load_freqs1_data(data, qs)
    # data['freqs2'] = load_freqs2_data(data, qs)

    # encoder = questions_encoder(train_data, test_data)
    # x_q1 = e.extract(data, 'question1', encoder)
    # data['tfidf'] = load_freqs_data(data, freqs)

    # load_words(data)

    log('DONE train data.')
    return data


def load_test_data(raw=False):
    log('Loading test data...')
    data = pd.read_csv('input/test.csv')
    data = inputing_data(data)
    if raw is True:
        return data

    # data['shared_words'] = load_words_data(data)
    # data['shared_chars'] = load_chars_data(data)
    # data['count'] = load_count_data(data)
    # data['similar'] = load_similar_data(data)

    # qs = questions(data)
    # data['freqs1'] = load_freqs1_data(data, qs)
    # data['freqs2'] = load_freqs2_data(data, qs)

    log('DONE test data.')
    return data


def inputing_data(data):
    log('Inputing data...')
    return data.fillna('')


def normalized_count(row):
    w1 = len(row.question1.split())
    w2 = len(row.question2.split())
    return min(w1, w2) / max(w1, w2)


def load_count_data(data):
    log('Loading count...')
    return data.apply(normalized_count, axis=1, raw=True)


#
# TF-IDF
#
def normalized_tfidf(row, x_q1, x_q2):
    tf1 = x_q1.iloc[row.id].sum()
    tf2 = x_q2.iloc[row.id].sum()
    return float(tf1) / (tf1 + tf2)


def load_tfidf_data(data, encoder):
    x_q1 = e.extract(data, 'question1', encoder)
    x_q2 = e.extract(data, 'question2', encoder)
    data['tfidf_ratio'] = data.apply(
        normalized_tfidf, x_q1=x_q1, x_q2=x_q2, axis=1, raw=True)


def questions_corpus(train_data, test_data):
    data = train_data.question1.append(train_data.question2)
    return data.append(test_data.question1).append(test_data.question2).values


def questions_encoder(train_data, test_data, max_features):
    corpus = questions_corpus(train_data, test_data)
    question_we = p.count_encoder(corpus, max_features=max_features)
    return question_we, question_we.get_feature_names(), question_we.get_stop_words()


stops = fe.text.ENGLISH_STOP_WORDS


def normalized_word_share(row):
    w1 = set(map(lambda word: word.lower().strip(),
                 row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(),
                 row['question2'].split(" ")))
    return len(w1 & w2) / (len(w1) + len(w2))


def shared_words(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are
        # nothing but stopwords
        return 0
    shared_words = [w for w in q1words.keys() if w in q2words]
    R = (len(shared_words)) / (len(q1words) + len(q2words))
    return R


def load_shared_words_data(data):
    log('Loading shared words...')
    return data.apply(shared_words, axis=1, raw=True)


#
# n-grams
#
stops = set(stopwords.words('english'))
color = sns.color_palette()


def _words(que):
    # TODO: try stemming and lemming
    return set(word_tokenize(str(que).lower()))


def get_stops(que):
    return [word for word in _words(que) if word in stops]


#
# unigrams
#
def get_unigrams(que):
    return [word for word in _words(que) if word not in stops]


def get_common_unigrams(row):
    return len(set(row["unigrams_ques1"]).intersection(set(row["unigrams_ques2"])))


def get_common_unigram_ratio(row):
    return float(row["unigrams_common_count"]) / max(len(set(row["unigrams_ques1"]).union(set(row["unigrams_ques2"]))), 1)


def load_unigrams(data):
    log('Loading unigrams...')
    data["unigrams_ques1"] = data['question1'].apply(get_unigrams)
    data["unigrams_ques2"] = data['question2'].apply(get_unigrams)
    data["unigrams_common_count"] = data.apply(get_common_unigrams, axis=1)
    data["unigrams_common_ratio"] = data.apply(
        get_common_unigram_ratio, axis=1)


#
# bigrams
#

def get_bigrams(que):
    return [i for i in ngrams(que, 2)]


def get_common_bigrams(row):
    return len(set(row["bigrams_ques1"]).intersection(set(row["bigrams_ques2"])))


def get_common_bigram_ratio(row):
    return float(row["bigrams_common_count"]) / max(len(set(row["bigrams_ques1"]).union(set(row["bigrams_ques2"]))), 1)


def load_bigrams(data):
    log('Loading bigrams...')
    data["bigrams_ques1"] = data["q1"].apply(get_bigrams)
    data["bigrams_ques2"] = data["q2"].apply(get_bigrams)
    data["bigrams_common_count"] = data.apply(get_common_bigrams, axis=1)
    data["bigrams_common_ratio"] = data.apply(get_common_bigram_ratio, axis=1)

#
# trigrams
#


def get_trigrams(que):
    return [i for i in ngrams(que, 3)]


def get_common_trigrams(row):
    return len(set(row["trigrams_ques1"]).intersection(set(row["trigrams_ques2"])))


def get_common_trigram_ratio(row):
    return float(row["trigrams_common_count"]) / max(len(set(row["trigrams_ques1"]).union(set(row["trigrams_ques2"]))), 1)


def load_trigrams(data):
    log('Loading trigrams...')
    data["trigrams_ques1"] = data["q1"].apply(get_trigrams)
    data["trigrams_ques2"] = data["q2"].apply(get_trigrams)
    data["trigrams_common_count"] = data.apply(get_common_trigrams, axis=1)
    data["trigrams_common_ratio"] = data.apply(get_common_bigram_ratio, axis=1)

#
# words / stops / ratios
#


def process_questions(t):
    text = tap.preprocess_text(t, fix_unicode=True, lowercase=True,
                               transliterate=True, no_urls=True,
                               no_emails=True, no_phone_numbers=True,
                               no_numbers=True,
                               no_currency_symbols=True, no_punct=True,
                               no_contractions=True, no_accents=True)
    return set(word_tokenize(text))


def load_questions(data):
    log('Loading q1...')
    data['q1'] = data['question1'].apply(process_questions)
    log('Loading q2...')
    data['q2'] = data['question2'].apply(process_questions)


def process_words(words):
    return words.difference(stops)


def process_stops(words):
    return words.intersection(stops)


def load_words(data):
    log('Loading words and counts...')
    data['words1'] = data['q1'].apply(process_words)
    data['wc1'] = data['words1'].apply(lambda ws: len(ws))
    data['words2'] = data['q2'].apply(process_words)
    data['wc2'] = data['words2'].apply(lambda ws: len(ws))


def load_stops(data):
    log('Loading stops and counts...')
    data['stops1'] = data['q1'].apply(process_stops)
    data['sc1'] = data['stops1'].apply(lambda ws: len(ws))
    data['stops2'] = data['q2'].apply(process_stops)
    data['sc2'] = data['stops2'].apply(lambda ws: len(ws))


def process_shared_words_ratio(row):
    count = row['wc1'] + row['wc2']
    if count == 0:
        return 0
    else:
        return row['shared_words_count'] / count


def load_ratios(data):
    log('Loading stops/words ratio...')
    data['ws_ratio1'] = data.apply(
        lambda row: 0 if row['wc1'] == 0 else row['sc1'] / row['wc1'], axis=1)
    data['ws_ratio2'] = data.apply(
        lambda row: 0 if row['wc2'] == 0 else row['sc2'] / row['wc2'], axis=1)


def load_shared_words(data):
    log('Loading shared words and counts...')
    data['shared_words'] = data.apply(
        lambda row: row['words1'].intersection(row['words2']), axis=1)
    data['shared_words_count'] = data.apply(
        lambda row: len(row['shared_words']), axis=1)
    data['shared_words_ratio'] = data.apply(process_shared_words_ratio, axis=1)


#
# similarity
#
def process_similar(row):
    return SequenceMatcher(None, row.question1, row.question2).ratio()


def load_similar(data):
    log('Loading similar...')
    data['similar'] = data.apply(process_similar, axis=1)


#
# shared chars
#
def process_chars(row):
    c1 = set(list(row.question1))
    c2 = set(list(row.question2))
    return len(c1.intersection(c2))


def load_chars(data):
    log('Loading chars...')
    data['cc1'] = data['question1'].apply(lambda q: len(set(list(q))))
    data['cc2'] = data['question2'].apply(lambda q: len(set(list(q))))
    data['shared_chars_count'] = data.apply(process_chars, axis=1)
    data['shared_chars_ratio'] = data.apply(
        lambda row: row['shared_chars_count'] / (row['cc1'] + row['cc2']), axis=1)

#
# frequencies
#


def questions(data):
    q = data.question1.append(data.question2)
    return pd.DataFrame(q.value_counts())


def freq1(row, questions):
    return questions.loc[row['question1']][0]


def freq2(row, questions):
    return questions.loc[row['question2']][0]


def load_freqs(data, questions):
    log('Loading freqs...')
    data['freqs1'] = data.apply(freq1, questions=questions, axis=1, raw=True)
    data['freqs2'] = data.apply(freq2, questions=questions, axis=1, raw=True)
