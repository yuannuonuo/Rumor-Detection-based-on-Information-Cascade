import re

emo_repl = {
    # positive emotions
    "&lt;3": " good ",
    ":d": " good ",  # :D in lower case
    ":dd": " good ",  # :DD in lower case
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",

    # negative emotions:
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":S": " bad ",
    ":-S": " bad ",
}

emo_repl_order = [k for (k_len, k) in reversed(
    sorted([(len(k), k) for k in list(emo_repl.keys())]))]

re_repl = {
    r"\br\b": " are ",
    r"\bu\b": " you ",
    r"\bhaha\b": " ha ",
    r"\bhahaha\b": " ha ",
    r"\bdon't\b": " do not ",
    r"\bdoesn't\b": " does not ",
    r"\bdidn't\b": " did not ",
    r"\bhasn't\b": " has not ",
    r"\bhaven't\b": " have not ",
    r"\bhadn't\b": " had not ",
    r"\bwon't\b": " will not ",
    r"\bwouldn't\b": " would not ",
    r"\bcan't\b": " can not ",
    r"\bcannot\b": " can not ",
    r"\bhe's\b": " he is ",
    r"\bshe's\b": " she is ",
    r"\bit's\b": " it is ",
    r"\byou're\b": " you are ",
    r"\bwasn't\b": " was not ",
    r"\bthat's\b": " was not ",
    r"\bi'm\b": " i am ",
    r"\bisn't\b": " is not ",
    r"\bi'll\b": " i will ",
    r"\byou'll\b": " you will ",
    r"\bshe'll\b": " she will ",
    r"\bhe'll\b": " he will ",
    r"\bit'll\b": " it will ",
    r"\bgod's\b": " god is ",
    r"\bi'd\b": " i would ",
    r"\b'd\b": " would ",
    r"\bwhat's\b": " what is ",
    r"\bwomen's\b": " women ",
    r"\bwe're\b": " we are ",
    r"\bthis's\b": " this is ",
    r"\byr\b": " year ",
    r"\baren't\b": " are not ",
    r"\bcouldn't\b": " could not ",
    r"\bshouldn't\b": " should not ",
    r"\bthere's\b": " there is ",
    r"\bhow's\b": " how is ",
    r"\b&\b": " and ",
    r"\bweren't\b": " were not ",
    r"\b's\b": " 's ",
    r"\b'll\b": " will ",
    r"- ": "  ",
    r"'s ": " ",
    r" -": "  "
}

def clean_zh_text(text):
    # keep English, digital and Chinese
    s = re.sub(r'[^\w\s]', '', text)
    return s

##英文文本预处理：包括去除非文本字符，词干化，去除停用词。
def text_preprocess(text):
    ##定义表情
    try:
            # Wide UCS-4 build
        core = re.compile(u'['
            u'\U0001F300-\U0001F64F'
            u'\U0001F680-\U0001F6FF'
            u'\u2600-\u2B55]+',
            re.UNICODE)
    except re.error:
        # Narrow UCS-2 build
        core = re.compile(u'('
            u'\ud83c[\udf00-\udfff]|'
            u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
            u'[\u2600-\u2B55])+',
            re.UNICODE)
    # 定义url的正则表达式
    urlre = re.compile('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]')
    # 定义@用户的正则表达式
    atre = re.compile('@[A-Za-z0-9_]*')

    # 定义\t正则表达式
    tabre = re.compile('\t+')
    # 定义\n正则表达式
    enterre = re.compile('\n+')
    # 定义数字
    number = re.compile('[-+]?[.\d]*[\d]+[/:,.\d]*')
    # 定义hashtag
    hashtag = re.compile('[#]+')
    ##定义语气符号
    g_symbol = re.compile('[,\?\?!]+')

    ##定义去除非英文字符
    other = re.compile('[/(/)/%/^/*///+/"/:/;/-/”/`/`/./</~/>/\[/\]/\'/=!?“：.]+')


    text = urlre.sub('', text)
    text = hashtag.sub(' ', text)
    text = atre.sub(' ', text)
    text = preprocessor(text)
    text = core.sub(' ', text)
    text = number.sub('', text)
    text = tabre.sub(' ', text)
    text = enterre.sub(' ', text)
    text = other.sub(' ', text)

    for i in g_symbol.findall(text):
        if len(i) > 1:
            text = text.replace(i, ' ' + i[0] + ' ' + '<repeat>' + ' ')
        else:
            text = text.replace(i, ' ' + i[0] + ' ')

    return text

def preprocessor(tweet):
    global emoticons_replaced
    tweet = tweet.replace('RT ', '')
    tweet = tweet.lower()
    # tweet = tweet.replace('#semst', '')
    # tweet = tweet.replace('#tcot', '')
    # tweet = tweet.replace('#freethinkers', '')
    # tweet = tweet.replace('non-feminist', 'non feminist')
    # tweet = tweet.replace('?!', ' ? ! ')
    # tweet = tweet.replace('obama', ' obama ')
    # tweet = tweet.replace('hillary', ' hillary ')
    # tweet = tweet.replace('clinton', ' clinton ')
    # tweet = tweet.replace('hillaryclinton', ' hillary clinton ')
    # tweet = tweet.replace('donald', ' donald  ')
    # tweet = tweet.replace('trump', ' trump ')
    # tweet = tweet.replace('donaldtrump', ' donald trump ')
    # tweet = tweet.replace('misandreeeeeeeeeee', ' ')
    # tweet = tweet.replace('"Í¢‘„‘îChristopher Hitchens', ' ')
    # tweet = tweet.replace('#1a', ' ')
    # i=0
    # with open("./stopphrase.txt") as f:
    #     for line in f:
    #         # i=i+1
    #         # print(i)
    #         word = line.strip().split()[0].lower()
    #         tweet = tweet.replace('#' + word, ' ')
    #         tweet = tweet.replace(word, ' ')

    for k in emo_repl_order:
        tweet = tweet.replace(k, emo_repl[k])
    for r, repl in re_repl.items():
        tweet = re.sub(r, repl, tweet)

    return tweet