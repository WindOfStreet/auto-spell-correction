import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pinyin

from twogram import TwoGrams
from edit_distance import EditDistance


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR + '\SentenceGeneration')
sys.path.append((BASE_DIR + '\edit_distance'))
print('base dir is:{}'.format(BASE_DIR))

'''
这样写就是不行...
pycharm都重装了
sys.path.append(BASE_DIR)
from SentenceGeneration.twogram import TwoGrams
from dit_distance.dit_distance import EditDistance
'''


class Correct:
    def __init__(self):
        self.lm = TwoGrams()
        self.dist = EditDistance(1, 1, 1)

    def preprocess(self, token_path, save_dict_path):
        # 读取文本转换为拼音
        file = open(token_path, encoding='utf-8').readlines()
        tokens = []  # 存放所有拼音语料['yun','cai'...]
        paragraphs = []  # 分段落存储拼音语料for 2gram model
        for line in file[:2]:
            text = pinyin.get(line, format='strip', delimiter=' ')
            token_list = re.findall('[a-z]+', text.lower())
            token_list = [token for token in token_list if len(token) > 1]
            tokens.extend(token_list)
            paragraphs.append(token_list)

        # 词频字典for jieba
        cnt = Counter(tokens)
        dict_file = open(save_dict_path, 'w')
        for key, val in cnt.items():
            dict_file.write('{} {}\n'.format(key, val))
        dict_file.close()
        # 建立语言模型
        data = pd.Series(paragraphs)
        self.lm.add_dict(save_dict_path)
        self.lm.train(data)
        # print(self.lm.prob_2('xiao', 'mi'))
        # print(self.lm.prob_2('wang', 'he'))
        # print(self.lm.calc_perplexity(['xiao', 'mi']))
        # print(self.lm.calc_perplexity(['bao', 'mi']))
        # print(self.lm.calc_perplexity(['tiong', 'shuo']))
        # print(self.lm.calc_perplexity(['ting', 'shuo']))

        return

    def find_related(self, string, level=1):
        """
        从语料库中，为字符串string寻找编辑距离最近的候选字符串
        :param string: 需要处理的字符串
        :param level: 编辑距离范围，建议1或者2
        :return: 候选字符串的list
        """
        candidate = []
        for token in self.lm.wfreq:
            if abs(len(string) - len(token)) <= level:
                if self.dist.calc_distance(token, string) <= level:
                    candidate.append(token)
        return candidate

    def correct(self, string):
        seg = self.lm.jieba._lcut_for_search(string)
        suggest = []
        seg.insert(0, 'BOS')
        seg.append('EOS')
        for i in range(1, len(seg) - 1, 1):
            if seg[i] in self.lm.wfreq:
                suggest.append(seg[i])
            else:
                # 处理需要纠正的字符串：
                candidate = self.find_related(seg[i], 1)
                score = []
                for token in candidate:
                    # candidate[j]部分以再换成已更正后的string
                    prob_last = self.lm.prob_2(seg[i - 1], token)  # 与上一词的联合概率
                    prob_front = self.lm.prob_2(token, seg[i + 1])  # 与下一词的联合概率
                    score.append((token, prob_last + prob_front))  # 计算所有候选词的概率
                best_str, max_idx = max(score, key=lambda x: x[1])  # 按概率排序
                suggest.append(best_str)
        return ''.join(suggest)


if __name__ == '__main__':
    model = Correct()
    model.preprocess('input/article_9k.txt', 'input/pinyin.dict')
    string = 'xaomi'
    print('{}==>{}'.format(string, model.correct(string)))
