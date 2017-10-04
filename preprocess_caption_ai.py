#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This program preprocesses the caption into picke file.
Main purpose is to tokenize, make lower case, and filter out low frequent vocaburaries.
Note tokenize and  make lower case is done by a function (read_MSCOCO_json) in another file MSCOCO.py.
"""
from gensim import corpora
import pickle
import json
import jieba
from tqdm import tqdm


def read_from_json(dir_input):
    """
    :param dir_input:
    :return:
        captions: {sentence_length: caption_id}
        caption_id2tokens: {caption_id: caption tokens}  1050000 items
        caption_id2image_id : {caption_id: image_id}
    """
    with open(dir_input, 'r') as f_json:
        data = json.load(f_json)

    len2caption_id = {}
    caption_id2tokens = {}
    caption_id2image_id = {}

    captions_raw = [data[i]['caption'] for i in range(len(data))]
    image_ids = [data[i]['image_id'] for i in range(len(data))]
    count = 0
    for captions_list, image_id in zip(captions_raw, image_ids):
        for caption in captions_list:
            caption = caption.replace('\n', '').strip()
            if caption[-1] == '.':  # delete the last period.
                caption = caption[0:-1]

            caption_tokens = ['<SOS>']
            caption_tokens += jieba.cut(caption)
            # caption_tokens += nltk.word_tokenize(caption)
            caption_tokens.append("<EOS>")
            caption_length = len(caption_tokens)
            if caption_length in len2caption_id:
                len2caption_id[caption_length].add(count)
            else:
                len2caption_id[caption_length] = set([count])

            caption_id2tokens[count] = caption_tokens
            caption_id2image_id[count] = image_id
            count += 1

    return len2caption_id, caption_id2tokens, caption_id2image_id


def pre_process_caption_train():
    dir0 = '/media/csc105/Data/dataset/AI-challenger/ai_challenger_caption_train_20170902/'
    len2caption_id_t, caption_id2tokens_t, caption_id2image_id_t = read_from_json(dir0 + 'caption_train_annotations_20170902.json')
    texts = caption_id2tokens_t.values()  # sentence list
    dictionary = corpora.Dictionary(texts)  # 17625
    dictionary.filter_extremes(no_below=5, no_above=1.0)  # 7757
    dictionary.compactify()  # remove gaps in id sequence after words that were removed
    index2token = dict((v, k) for k, v in dictionary.token2id.iteritems())
    ukn_id = len(dictionary.token2id)
    index2token[ukn_id] = '<UKN>'  # add UKN in the last of dict

    with open('data/ai/index2token.pkl', 'w') as f_dict:
        pickle.dump(index2token, f_dict)
        print 'saved index2token'

    caption_id2sentence_t = dict()
    for (caption_id, tokens) in tqdm(caption_id2tokens_t.iteritems()):
        sentence = []
        for token in tokens:
            if token in dictionary.token2id:
                sentence.append(dictionary.token2id[token])
            else:
                sentence.append(ukn_id)

        caption_id2sentence_t[caption_id] = sentence

    with open('data/ai/preprocessed_captions_train.pkl', 'w') as f:
        pickle.dump((len2caption_id_t, caption_id2sentence_t, caption_id2image_id_t), f)
        print 'saved captions'


if __name__ == '__main__':
    pre_process_caption_train()
