from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer, scoring
from nltk.tokenize import word_tokenize
import json
from google_bleu import compute_bleu
import numpy as np
import string
import collections
from nltk.util import ngrams
from meteor import meteor_score
import os
from transformers import T5Tokenizer

SPECIAL_TOKEN = ["<s>", "<cls>", "</s>", "<pad>", "<mask>"]
punct = string.punctuation # string


def read_data(path):
    data = [json.loads(ln) for ln in open(path).readlines()]
    gen_list = []
    ref_list = []

    syss = []
    refs = []

    for sample in data:

        cur_gen = sample["gen"].replace("<s>", "").replace("<pad>", "").replace("<cls>", "").replace("<mask>", "").replace("</s>", "").strip()
        cur_ref = sample["output"].replace("<s>", "").replace("<pad>", "").replace("<cls>", "").replace("<mask>", "").replace("</s>", "").strip() 

        cur_gen = cur_gen.lower()
        cur_ref = cur_ref.lower()

        gen_list.append(word_tokenize(cur_gen))
        ref_list.append(word_tokenize(cur_ref))

    return gen_list, ref_list


def eval_bleu(gen_list, ref_list):
    reference_corpus = [[elem] for elem in ref_list]
    translation_corpus = gen_list

    bleu_score_dict = {}
    for max_order in [1, 2, 3, 4]:
        bleu_score = compute_bleu(
            reference_corpus, 
            translation_corpus, 
            max_order=max_order,
            smooth=True
            )
        bleu_score_dict[max_order] = bleu_score[0]
    
    chencherry = SmoothingFunction()
    nltk_bleu = corpus_bleu(
        reference_corpus, 
        translation_corpus, 
        smoothing_function=chencherry.method1
        )
    bleu_score_dict["nltk"] = nltk_bleu
    
    # return bleu_score_dict
    for score_type in bleu_score_dict:
        print("bleu-", score_type, ": ", bleu_score_dict[score_type])
    return bleu_score_dict


def eval_meteor(gen_list, ref_list):
    score_list = []
    for gen, ref in zip(gen_list, ref_list):
        score = round(meteor_score([" ".join(ref)], " ".join(gen)),4)
        score_list.append(score)
    # return np.mean(score_list)
    print("meteor score: ", np.mean(score_list))



def eval_rouge(gen_list, ref_list):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()
    for gen, ref in zip(gen_list, ref_list):
        gen_str = " ".join(gen)
        ref_str = " ".join(ref)
        cur_score = scorer.score(ref_str, gen_str)
        aggregator.add_scores(cur_score)

    aggregates = aggregator.aggregate()
    for score_type, aggregate in sorted(aggregates.items()):
        print("%s-R,%f,%f,%f\n" %
             (score_type, aggregate.low.recall, aggregate.mid.recall,
             aggregate.high.recall))
        print("%s-P,%f,%f,%f\n" %
             (score_type, aggregate.low.precision,
             aggregate.mid.precision, aggregate.high.precision))
        print("%s-F,%f,%f,%f\n" %
             (score_type, aggregate.low.fmeasure,
             aggregate.mid.fmeasure, aggregate.high.fmeasure))


def eval_one(path):
    gen_list, ref_list = read_data(path)

    print("num: ", len(gen_list), len(ref_list))
    print("avg length: ", np.mean([len(elem) for elem in gen_list]))
    print(gen_list[0])
    print(ref_list[0])
    print("=======evaluate bleu=======")
    eval_bleu(gen_list, ref_list)

    print("\n=======evaluate rouge=======")
    eval_rouge(gen_list, ref_list)

    print("\n=======evaluate meteor=======")
    eval_meteor(gen_list, ref_list)



if __name__ == "__main__":
    folder = "./outputs_all/"
    all_files = [elem for elem in os.listdir(folder) if ".jsonl" in elem]
    print(all_files)


    for cur_file in all_files:
        path = folder + "/" + cur_file
        print("evaluate file: ", path)
        eval_one(path)
        print("\n\n")



