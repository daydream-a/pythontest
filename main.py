import pandas as pd
import numpy as np
import warnings
import jieba
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from scipy.linalg import norm

tqdm.pandas()
warnings.filterwarnings('ignore')


def clearTxt(line):
    line = str(line)
    if line != '':
        line = line.strip()
        # 去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]", "", line)
        # 去除文本中的中文符号和英文符号

        # 分词
        segList = jieba.cut(line, cut_all=False)
        segSentence = ''
        for word in segList:
            if word != '\t':
                segSentence += word + " "
    return segSentence.strip()


def get_stop_key():
    stop_list = []
    f = open('hit_stopwords.txt', encoding='utf-8')
    for k in f.readlines():
        if k != '':
            stop_list.append(k.strip())
    return stop_list


stop_list = get_stop_key()


def stopWord(line):
    line = str(line)
    sentence = delstopword(line, stop_list)
    return sentence


def delstopword(line, stopkey):
    wordList = line.split(' ')
    sentence = ''
    for word in wordList:
        word = word.strip()
        if word not in stopkey:
            if word != '\t':
                sentence += word + " "
    return sentence.strip()


def space_process(line):
    return re.sub(" {1,}", " ", line)


def get_Relevance():
    tagetlist = []
    for i in sin_num1:
        if i == 0:
            tagetlist.append("不相关")
        elif i <= 0.05:
            tagetlist.append("弱相关")
        else:
            tagetlist.append("强相关")
    taget_num = pd.DataFrame(tagetlist)
    return taget_num


if __name__ == "__main__":
    # 读取数据
    datatxt = pd.read_excel("2020-2021茂名（含自媒体）.xlsx", sheet_name=4)
    # 数据整合
    dataset = datatxt[:]
    dataset["text"] = datatxt["公众号标题"] + '\n' + dataset["正文"]
    dataset = pd.concat([dataset["文章ID"], dataset["text"]], axis=1)
    # 清除文章特殊字符
    text = dataset['text'].progress_apply(clearTxt)
    # 清楚文章停止词
    text1 = text.progress_apply(stopWord)
    # 清楚文章空格
    text2 = text1.progress_apply(space_process)
    # 打开匹配词表
    s1 = open("s1.txt", encoding='utf-8').readline()
    s1 = re.sub(',', ' ', s1)
    # 计算TF-DIF值
    tf_sim_num = []
    for i in range(len(text2)):
        s2 = text2[i]
        cv = CountVectorizer(tokenizer=lambda s: s.split())  # 转化为TF矩阵
        corpus = [s1, s2]
        vectors = cv.fit_transform(corpus).toarray()  # 计算TF系数
        sim = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
        tf_sim_num.append(sim)

    sin_num1 = np.array(tf_sim_num)
    # 数据输出
    sin_num = pd.DataFrame(sin_num1)
    sin_num.columns = ["相关系数"]
    taget_num = get_Relevance()
    taget_num.columns = ["相关性"]
    all = pd.concat([dataset["文章ID"], sin_num["相关系数"], taget_num["相关性"]], axis=1)
    all.to_excel("result.xlsx", sheet_name="1", encoding='utf-8')
