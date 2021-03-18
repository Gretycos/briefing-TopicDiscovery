# -*- coding: utf-8 -*-

"""
Usage:
    run.py train --num_topics=<int> [options]
    run.py predict --num_topics=<int> [options]
    run.py update --num_topics=<int> [options]

Options:
    -h --help                               show this screen.
    --days=<int>                            process data before days [default: 88]
    --proportion=<float>                    load how many data for training [default: 0.7]
    --num_topics=<int>                      num of topics [default: 10]
    --chunk_size=<int>                      num of documents to be used in each training chunk [default: 2000]
    --passes=<int>                          num of passes through the corpus during training [default: 40]
    --alpha                                 a-priori belief for the each topics' probability [default: auto]
    --eta                                   a-priori belief on word probability [default: auto]
    --decay=<float>                         weight what pct of the previous lambda value is forgotten [default: 0.85]
    --offset=<float>                        controls how much we slow down 1st steps or few iterations [default: 1.0]
    --eval_every=<int>                      estimate log perplexity every that many updates [default: 10]
    --iterations=<int>                      max number of iterations through the corpus [default: 400]
    --gamma_threshold=<float>               min change in gamma parameters to continue iterating [default: 0.001]
    --minimum_probability=<float>           prob of topics lower than this will be filtered out [default: 0.01]
    --random_state=<int>                    Useful for reproducibility [default: 1]
    --per_word_topics                       most likely topics for each word
    --all                                   train models of all topics
"""


from docopt import docopt
from gensim.models import LdaModel, CoherenceModel
from util import load_new_corpus, load_corpus_dictionary, load_update_corpus, save_json, save_ppl_coh
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
# sys.path.append(os.path.dirname(sys.path[0]))

# 进度记录
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def train(args):
    lda_model = None
    corpus = None
    perplexities = []
    coherence_values = []
    topics = range(2,int(args['--num_topics'])+1,2) if args['--all'] else range(int(args['--num_topics']),int(args['--num_topics'])+1)

    for num_topics in topics:
        os.makedirs(name='./model/{}/'.format(num_topics), exist_ok=True)
        try:
            logger.info("loading model")
            lda_model = LdaModel.load('./model/{}/topic_{}.model'.format(num_topics, num_topics))
            corpus, dictionary = load_corpus_dictionary(float(args['--proportion']))
        except:
            logger.info("not found model saved")
            corpus, dictionary = load_corpus_dictionary(float(args['--proportion']))
            logger.info("training model")
            lda_model = LdaModel(
                corpus=corpus,
                num_topics=num_topics,
                id2word=dictionary,  # Dictionary对象
                chunksize=int(args['--chunk_size']),
                passes=int(args['--passes']),
                alpha='symmetric' if args['--alpha'] else 'auto',
                eta=None if args['--eta'] else 'auto',
                decay=float(args['--decay']),
                offset=float(args['--offset']),
                eval_every=int(args['--eval_every']),
                iterations=int(args['--iterations']),
                gamma_threshold=float(args['--gamma_threshold']),
                minimum_probability=float(args['--minimum_probability']),
                random_state=int(args['--random_state']),
                per_word_topics=True if args['--per_word_topics'] else False
            )
            logger.info("saving trained model")
            lda_model.save('./model/{}/topic_{}.model'.format(num_topics, num_topics))
        finally:
            perplexities.append(np.exp2(-lda_model.log_perplexity(corpus)))
            coherence_values.append(CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass').get_coherence())
            lda_model.print_topics(5,5)


    if args['--all']:
        os.makedirs(name='./pic/{}/'.format(args['--days']), exist_ok=True)
        draw_graph_perplexity(args,perplexities,topics)
        draw_graph_coherence(args,coherence_values,topics)
    else:
        save_ppl_coh(perplexities[0],coherence_values[0],int(args['--days']))
        logger.info("perplexity: {}; coherence: {}.".format(perplexities[0],coherence_values[0]))


def draw_graph_perplexity(args,perplexities,topics):
    _, ax = plt.subplots()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # 整数坐标轴
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2)) # 坐标轴间隔
    x = topics
    y = perplexities
    plt.plot(x, y, color="red", linewidth=2, marker='o')
    plt.grid(True,linestyle='--')
    plt.xlabel("Num of Topics")
    plt.ylabel("Perplexity")
    plt.savefig("./pic/{}/topics_perplexity_{}.jpg".format(args['--days'],args["--num_topics"]))
    # plt.show()


def draw_graph_coherence(args,coherence_values,topics):
    _, ax = plt.subplots()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    x = topics
    y = coherence_values
    plt.plot(x, y, color="red", linewidth=2, marker='o')
    plt.grid(True,linestyle='--')
    plt.xlabel("Num of Topics")
    plt.ylabel("Coherence Value")
    plt.savefig("./pic/{}/topics_coherence_{}.jpg".format(args['--days'],args["--num_topics"]))
    # plt.show()


def update(args):
    """
    更新模型
    :param args: dict
    :return: None
    """
    topics = range(2, int(args['--num_topics']) + 1, 2) if args['--all'] else range(int(args['--num_topics']),
                                                                                    int(args['--num_topics']) + 1)
    for topic in topics:
        logger.info("loading model")
        model = LdaModel.load('./model/{}/topic_{}.model'.format(topic, topic))
        update_corpus = load_update_corpus(int(args['--days']))
        logger.info("updating model")
        model.update(update_corpus)
        logger.info("saving updated model")
        model.save('./model/{}/topic_{}.model'.format(topic, topic))
        # corpus, dictionary = load_update_corpus()
        # logger.info("training to update model")
        # lda_model = LdaModel(
        #     corpus=corpus,
        #     num_topics=int(args['--num_topics']),
        #     id2word=dictionary,  # Dictionary对象
        #     chunksize=int(args['--chunk_size']),
        #     passes=int(args['--passes']),
        #     alpha='symmetric' if args['--alpha'] else 'auto',
        #     eta=None if args['--eta'] else 'auto',
        #     decay=float(args['--decay']),
        #     offset=float(args['--offset']),
        #     eval_every=int(args['--eval_every']),
        #     iterations=int(args['--iterations']),
        #     gamma_threshold=float(args['--gamma_threshold']),
        #     minimum_probability=float(args['--minimum_probability']),
        #     random_state=int(args['--random_state']),
        #     per_word_topics=True if args['--per_word_topics'] else False
        # )
        # logger.info("saving trained model")
        # lda_model.save('./model/{}/topic_{}.model'.format(args['--num_topics'], args['--num_topics']))
        # lda_model.print_topics(5, 5)
        # logger.info("perplexity: {}; coherence: {}.".format(np.exp2(-lda_model.log_perplexity(corpus)),CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass').get_coherence()))



def predict(args):
    """
    预测今日文档的主题
    :param args: dict
    :return: None
    """
    lda_model = LdaModel.load('./model/{}/topic_{}.model'.format(args['--num_topics'],args['--num_topics']))
    topics = lda_model.show_topics(num_topics=int(args['--num_topics']), num_words=5,log=False,formatted=False)
    topics_dict = {} # 主题编号与关键词
    topics_count_dict = {} # 主题-计数
    topics_doc_dict = {} # 主题-文章
    topics_doc_pro_dict = {} # 主题-(文档，概率)
    # 初始化三个字典
    for t in topics:
        topics_dict[t[0]] = [word[0] for word in t[1]]
        topics_count_dict[t[0]] = 0
        topics_doc_dict[t[0]] = []
        topics_doc_pro_dict[t[0]] = (None,0) # id, prob
    # 加载今日文档
    # news_corpus_id: List[List[id, len(doc)]]
    new_corpus, new_corpus_id = load_new_corpus(int(args['--days']))
    # 预测每个文档的主题
    predicted_topics = lda_model.get_document_topics(new_corpus,minimum_probability=0.1)
    # 主题计数
    for idx,predicted_topic in enumerate(predicted_topics):
        # logger.info("topics of doc #{}: {}".format(new_corpus_id[idx],predicted_topic))
        # 处理单篇文章的主题
        for p_t in predicted_topic:
            topics_count_dict[p_t[0]] += 1
            topics_doc_dict[p_t[0]].append(new_corpus_id[idx][0])
            # 主题的概率更大，则替换
            if p_t[1] > topics_doc_pro_dict[p_t[0]][1]:
                # 过滤掉字数少的文章
                if new_corpus_id[idx][1] < 150:
                    continue
                topics_doc_pro_dict[p_t[0]] = (new_corpus_id[idx][0],p_t[1])
                logger.info("Topic #{}; doc_prob: {}".format(p_t[0],topics_doc_pro_dict[p_t[0]]))
    logger.info("Today's Top5 Topics from {} documents: ".format(len(new_corpus)))
    dict_list = []
    top_topic_doc_list=[]
    for item in sorted(topics_count_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
        d2j = {
            'topic': item[0],
            'count': item[1],
            'kw': topics_dict[item[0]],
            'article_id': topics_doc_dict[item[0]]
        }
        d2j2 = {
            'topic': item[0],
            'kw': topics_dict[item[0]],
            'article_id': topics_doc_pro_dict[item[0]][0]
        }
        dict_list.append(d2j)
        top_topic_doc_list.append(d2j2)
        logger.info("topic #{} (count={}): {}, top_kw: '{}'; doc={}, top_doc='{}'".format(
            item[0],
            item[1],
            topics_dict[item[0]],
            topics_dict[item[0]][0],
            topics_doc_dict[item[0]],
            topics_doc_pro_dict[item[0]][0]
        ))

    save_json(dict_list,'full',int(args['--days']))
    save_json(top_topic_doc_list,'top',int(args['--days']))

def main():
    args = docopt(__doc__)

    if args['train']:
        train(args)
    elif args['predict']:
        predict(args)
    elif args['update']:
        update(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()