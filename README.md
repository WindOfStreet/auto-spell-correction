## 简介
    对于输入的一串拼音字符进行自动纠错，给出最佳改正
## spell_corrections.py
    基于我的另外两个项目（2元语法模型和编辑距离计算模型），使用给定语料的拼音，训练一个拼音的2元语法模型。
    此模型可以对拼音字符串进行切词，然后对切分的单词进行纠错处理。利用编辑距离模型，给出候选的替换字符串列表。
    再利用2元语法模型，对候选字符串列表进一步筛选，获得最优改正结果。
    2元语法模型：https://github.com/WindOfStreet/2gram_model
    编辑距离计算模型：https://github.com/WindOfStreet/edit-distance
## example_spell_corrections.ipynb
    测试用例
  
