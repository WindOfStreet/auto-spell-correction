import os
import re
import sys
from collections import Counter

import pandas as pd
import pinyin
from twogram import TwoGrams
from edit_distance import EditDistance


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR + '\SentenceGeneration')
sys.path.append((BASE_DIR + '\edit_distance'))
# print('base dir is:{}'.format(BASE_DIR))

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

    def __process_single_char(self, string):
        """
        对列表中的单独的字母进行处理：与前面合并/与后面合并/删除
        :param seg: list of tokens
        :return: new list of tokens without single character
        """
        seg = self.lm.jieba.lcut(string)
        seg.insert(0, 'BOS')
        seg.append('EOS')
        for i in range(1, len(seg) - 1, 1):
            if len(seg[i]) == 1:
                if seg[i - 1] not in model.lm.wfreq:
                    # 前一个词不在字典中，则与前一个词合并
                    tmp = seg[i - 1] + seg[i]
                    seg.remove(seg[i - 1])
                    seg.remove(seg[i])
                    seg.insert(i - 1, tmp)
                elif seg[i + 1] not in model.lm.wfreq:
                    # 后一个词不在字典中，则与后一个词合并
                    tmp = seg[i] + seg[i + 1]
                    seg.remove(seg[i])
                    seg.remove(seg[i])
                    print('after remove:{}'.format(seg))
                    seg.insert(i, tmp)
                else:
                    # 前后词都在字典中，删除此字符
                    print('{}删除'.format(seg[i]))
                    seg.remove(seg[i])
        return seg

    def correct(self, string):
        """
        对输入的字符串拼音进行校正
        :param string: 待校正的字符串
        :return: 校正后的字符串
        """
        # 字符串预处理，获得分词列表
        seg = self.__process_single_char(string)
        # 处理各个分词
        suggest = []
        for i in range(1, len(seg)-1, 1):
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
    string = 'zhngguo'
    print('{}==>{}'.format(string, model.correct(string)))
