# -*- coding:utf-8 -*-
import logging
import re
import time
import traceback
from datetime import datetime, timedelta
from gensim import corpora
import numpy as np
import json
from dao import load_session
from dao.News import News
from ltp import LTP
import hanlp
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
logger = logging.getLogger(__name__)
DEFAULT = './tmp/default/'
NEW_PATH = './tmp/new/'
NEW = './tmp/new/new_topic'
UPDATE_PATH = './tmp/new/'
UPDATE = './tmp/new/new_topic'
TASK = ('tok/coarse',)


def test():
    # for days in range(1,0,-1):
    #     target_date = datetime.now().date() + timedelta(days=-days)
    #     with open("./predict/{}.json".format(target_date), "r", encoding="utf-8") as f:
    #         old_data = json.load(f)
    #         with open("./predict/top/{}.json".format(target_date), "r", encoding="utf-8") as f1:
    #             new_data = json.load(f1)
    #             for nd in new_data:
    #                 for od in old_data:
    #                     if nd['topic'] == od ['topic']:
    #                         nd['kw'] = od ['kw']
    #             with open("./predict/top/{}.json".format(target_date), "w", encoding="utf-8") as f2:
    #                 json.dump(new_data, f2)
    HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH, verbose=True)
    tasks = list(HanLP.tasks.keys())
    print(tasks)
    for task in tasks:
        if task not in TASK:
            del HanLP[task]
    tok = HanLP[TASK[0]]
    tok.dict_combine = {'新冠', '新冠病毒', '新冠肺炎'}
    print(HanLP("中新网3月31日电 据云南省委宣传部微博消息，3月31日晚，瑞丽市新冠肺炎疫情防控第二场新闻发布会召开，通报瑞丽市新冠肺炎疫防控情最新情况。")["tok/coarse"])
    pass


def load_word_segmentation_tool():
    """
    加载分词工具
    :return: HanLP: hanlp, ltp: LTP
    """
    logger.info("loading word segmentation tool")
    # HanLP = HanLPClient(url='https://www.hanlp.com/api', auth='MTE4QGJicy5oYW5scC5jb206MXFFOHhWUkJNQXBNdlh0NA==')
    HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH, verbose=True,devices=0)
    tasks = list(HanLP.tasks.keys())
    for task in tasks:
        if task not in TASK:
            del HanLP[task]
    tok = HanLP[TASK[0]]
    tok.dict_combine = {'新冠', '新冠病毒', '新冠肺炎'}
    ltp = LTP()
    logger.info("loaded word segmentation tool")
    return HanLP,ltp


def load_stop_words():
    """
    加载停用词
    :return: stop_words: List[str]
    """
    logger.info("loading stop words")
    with open('tmp/baidu_stopwords.txt', 'r') as f:
        stop_words = [line.strip('\n') for line in f.readlines()]
        logger.info("loaded stop words")
        return stop_words


def calculate_target_date(proportion):
    session = load_session()
    cursor = session.execute("select count(publish_date) "
                             "from (select distinct DATE_FORMAT(publish_time,'%Y-%m-%d') as publish_date "
                             "      from news) as a")
    all_days = cursor.fetchone()[0]
    target_date = datetime.now().date() + timedelta(days=-int((1 - proportion) * all_days))
    return target_date

def load_raw_documents(proportion):
    """
    从数据库中加载生文档
    :return: raw_corpus: List[Tuple[str]]
    """
    logger.info("loading {}% raw documents from database".format(int(proportion*100)))
    session = load_session()
    # cursor = session.execute("select count(publish_date) "
    #                          "from (select distinct DATE_FORMAT(publish_time,'%Y-%m-%d') as publish_date "
    #                          "      from news) as a")
    # all_days = cursor.fetchone()[0]
    # 目标日期以今天作为基准
    target_date = calculate_target_date(proportion)
    logger.info("loading data from ~ - {}".format(target_date + timedelta(days=-1)))
    raw_corpus = []
    for row in session.query(News.article_id,News.content).filter(News.publish_time < target_date):
        raw_corpus.append((row[0],row[1]))
    session.close()
    logger.info("loaded raw documents")
    return raw_corpus


def data_clean(raw_corpus=None):
    """
    数据清洗，包括分词与去停用词
    :param raw_corpus: List[Tuple(str,str)]
    :return: cleaned_documents: List[Tuple(str,str)]
    """
    logger.info("cleaning data")
    stop_words = load_stop_words()
    HanLP,ltp = load_word_segmentation_tool()
    cleaned_documents = []
    n = len(raw_corpus)
    i = 0
    for doc in raw_corpus:
        i += 1
        if i % 10 == 0 or i == n:
            logger.info("seg words from doc({}/{})".format(i,n))
            # time.sleep(60)
        doc_seg = []
        try:
            # sents = ltp.seg(ltp.sent_split(doc, flag='zh'))
            sents = HanLP(ltp.sent_split([doc[1]], flag='zh'))['tok/coarse']
            for sent in sents:
                for word in sent:
                    if len(word) > 1 and len(re.findall(r"\d+\.?\d*", word)) == 0 and word not in stop_words:
                        doc_seg.append(word)
            cleaned_documents.append((doc[0],doc_seg,len(doc[1])))
        except Exception as e:
            logger.error(e)
            logger.error(traceback.print_exc())
            continue
    logger.info("cleaned data")
    return cleaned_documents


def load_corpus_dictionary(proportion):
    """
    加载语料库和字典
    :return: corpus: corpora.MmCorpus, dictionary: Dictionary
    """
    try:
        logger.info("loading dictionary and corpus")
        dictionary = corpora.Dictionary().load(DEFAULT+'topic.dict')  # 加载字典
        corpus = corpora.MmCorpus(DEFAULT+'topic.mm')  # 加载语料库
    except:
        logger.info("not found dictionary and corpus saved")
        raw_documents = load_raw_documents(proportion)
        documents = data_clean(raw_documents)
        dictionary = corpora.Dictionary([doc[1] for doc in documents])
        os.makedirs(name=DEFAULT, exist_ok=True)
        dictionary.save(DEFAULT+'topic.dict')  # 存储字典
        corpus = [dictionary.doc2bow(doc[1]) for doc in documents]
        corpora.MmCorpus.serialize(DEFAULT+'topic.mm', corpus)  # 存储语料库
    logger.info("loaded dictionary and corpus")
    return corpus, dictionary


def load_new_raw_documents(days):
    """
    从数据库中加载最新一日的生文档
    :return: raw_documents: List[Tuple[str]]
    """
    logger.info("loading new raw documents")
    session = load_session()
    target_date = datetime.now().date() + timedelta(days=-days)
    tomorrow = target_date + timedelta(days=1)
    logger.info("loading data from {} - {}".format(target_date,tomorrow))
    raw_documents = []
    for row in session.query(News.article_id,News.content).filter(News.publish_time > target_date, News.publish_time < tomorrow):
         raw_documents.append((row[0],row[1]))
    session.close()
    logger.info("loaded new raw documents")
    return raw_documents


def load_new_corpus(days):
    """
    加载最新一日的语料库
    :return: new_corpus: corpora.MmCorpus, new_corpus_id: List[List[Str, Int]]
    """
    try:
        logger.info("loading new corpus")
        new_corpus = corpora.MmCorpus(NEW+'.mm')  # 加载新语料库
        new_corpus_id = np.load(NEW+'_id.npy').tolist()
    except:
        new_raw_documents = load_new_raw_documents(days)  # 新文档
        new_documents = data_clean(new_raw_documents)
        dictionary = corpora.Dictionary().load(DEFAULT+'topic.dict') # 加载词典
        new_corpus = [dictionary.doc2bow(doc[1]) for doc in new_documents]
        new_corpus_id = [[doc[0],doc[2]] for doc in new_documents]
        os.makedirs(name=NEW_PATH, exist_ok=True)
        corpora.MmCorpus.serialize(NEW+'.mm', new_corpus)
        np.save(NEW+'_id.npy', np.array(new_corpus_id))
        logger.info("loaded new corpus")
    return new_corpus, new_corpus_id


def load_update_raw_documents(days):
    """
        从数据库中加载昨日的生文档
        :return: raw_documents: List[Tuple[str]]
    """
    logger.info("loading update raw documents")
    session = load_session()
    target_date = datetime.now().date() + timedelta(days=-days)
    tomorrow = target_date + timedelta(days=1)
    logger.info("loading data from {} - {}".format(target_date, tomorrow))
    raw_documents = []
    for row in session.query(News.article_id, News.content).filter(News.publish_time > target_date, News.publish_time < tomorrow):
        raw_documents.append((row[0], row[1]))
    session.close()
    logger.info("loaded update raw documents")
    return raw_documents


def load_update_corpus(days):
    """
        加载昨日的语料库，用于更新
        :return: update_corpus: corpora.MmCorpus
    """
    try:
        logger.info("loading update corpus")
        update_corpus = corpora.MmCorpus(UPDATE+'.mm')  # 加载新语料库
        # dictionary.cfs={}
        # dictionary.add_documents([doc[1] for doc in update_documents])
        # dictionary.save('topic.dict')  # 存储字典
        # corpus = list(corpora.MmCorpus('topic.mm'))
        # for doc in update_documents:
        #     corpus.append(dictionary.doc2bow(doc[1]))
        # return corpus, dictionary
    except:
        logger.info("not found update corpus")
        update_raw_documents = load_update_raw_documents(days)  # 更新文档
        update_documents = data_clean(update_raw_documents)
        dictionary = corpora.Dictionary().load(DEFAULT+'topic.dict')  # 加载词典
        update_corpus = [dictionary.doc2bow(doc[1]) for doc in update_documents]
        os.makedirs(name=UPDATE_PATH, exist_ok=True)
        corpora.MmCorpus.serialize(UPDATE+'.mm', update_corpus)  # 存储语料库
    logger.info("loaded update corpus")
    return update_corpus


def save_json(dict_list,flag,days):
    target_date = datetime.now().date() + timedelta(days=-days)
    if flag == 'full':
        os.makedirs(name='./predict/', exist_ok=True)
        filename = './predict/{}.json'.format(target_date)
    elif flag == 'top':
        os.makedirs(name='./predict/top', exist_ok=True)
        filename = './predict/top/{}.json'.format(target_date)
    else:
        raise NameError
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(dict_list, f)

def save_ppl_coh(ppl, coh, days):
    diction = {
        "date": str(datetime.now().date() + timedelta(days=-days)),
        "ppl": ppl,
        "coh": coh
    }
    try:
        with open("./ppl_coh.json", "r", encoding="utf-8") as f:
            old_data = json.load(f)
            old_data.append(diction)
            with open("./ppl_coh.json", "w", encoding="utf-8") as f1:
                json.dump(old_data, f1)
    except:
        old_data = [diction]
        with open("./ppl_coh.json", "w", encoding="utf-8") as f:
            json.dump(old_data, f)




if __name__ == '__main__':
    test()