import re
import jieba

def isnumber(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


def remove(word):
    # remove everything other than number, letter and 中文
    regex = re.compile('[^a-zA-Z0-9\u4e00-\u9fa5-%.]')
    word = regex.sub('', word)

    # regex = re.compile('[^\x21-\x7E^\u4e00-\u9fa5]')
    # text = regex.sub('', text)

    # 去除数字
    if isnumber(word):
        return True

    # 去除单个ASCII字符
    if len(word) == 1 and word.isascii():
        return True

    # 去除ID
    if word[:2].lower() == 'id':
        return True

    # 去除单位
    # if word[-2:].lower() in ('ml', 'mg', 'ug', 'cm', 'iu', 'aa'):
    # return isnumber(word[:-2])

    # if word[-1:].lower() in ('g', 'm'):
    #     return isnumber(word[:-1])

    if word[:-1] in ('一', '二', '两', '三', '四', '五', '六', '七', '八', '九', '十') and word[-1:] in ('日', '次', '粒', '天'):
        return True

    return False

def clean(texts):
    res = []
    for text in texts:

        # 去除非中文非ascii字符
        regex = re.compile('[^a-zA-Z0-9\u4e00-\u9fa5-%.α]')
        text = regex.sub(' ', text)

        # 结巴分词
        text = jieba.lcut(text)
        text_ = []
        for word in text:
            if not remove(word):
                text_.append(word)

        # res.append(' '.join(''.join(text_).split()))
        res.append(''.join(''.join(text_).split()))

    return res

# Test Clean
texts = [
    '  0.9%氯化钠注(袋）_下午执行  ',
    '(国)维生素B2片（粤三才）1*120',
    '(国)5%葡萄糖注射液(50ml塑瓶)(粤大冢)1*60 ',
    '肿瘤坏死因子α(TNF-α)抑制剂',
    '罂粟碱(ID=2002002909)',
    '右旋糖酐40氯化钠注射液/30g:500ml/瓶',
    '鱼腥草滴眼液(乙基）△',
    '重组人白介素-2(125Ser)/50WIU/支',
    '#乙酰谷酰胺氯化钠注射液(医乙)         '
]

text_clean = clean(texts)
print(text_clean)