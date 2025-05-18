
import argparse
import csv
import os.path as osp
from transformers import BertTokenizerFast
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import torch
import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Union
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, accuracy_score

def evaluate(text_model, input_file, text_trunc_length):
    outputs = []

    with open(osp.join(input_file), encoding='utf8') as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for n, line in enumerate(reader):
            out_tmp = line['output'][6:] if line['output'].startswith('[CLS] ') else line['output']
            outputs.append((line['SMILES'], line['ground truth'], out_tmp))

    text_tokenizer = BertTokenizerFast.from_pretrained(text_model)

    bleu_scores = []
    meteor_scores = []

    references = []
    hypotheses = []

    for i, (smi, gt, out) in enumerate(outputs):
        if i % 100 == 0: print(i, 'processed.')


        gt_tokens = text_tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = text_tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))

    print('BLEU-2 score:', bleu2)
    print('BLEU-4 score:', bleu4)
    _meteor_score = np.mean(meteor_scores)
    print('Average Meteor score:', _meteor_score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for i, (smi, gt, out) in enumerate(outputs):

        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    print('ROUGE score:')
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])
    print('rouge1:', rouge_1)
    print('rouge2:', rouge_2)
    print('rougeL:', rouge_l)
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score


def get_roc_score(model,
                  features,
                  adj: torch.sparse.FloatTensor,
                  adj_tensor,
                  drug_nums,
                  edges_pos: np.ndarray, edges_neg: Union[np.ndarray, List[list]], test=None) -> Tuple[float, float]:

    model.eval()
    rec, emb = model(features, adj, adj_tensor, drug_nums, return_embeddings=True)
    emb = emb.detach().cpu().numpy()
    rec = rec.detach().cpu().numpy()
    adj_rec = rec

    preds, preds_neg = gen_preds(edges_pos, edges_neg, adj_rec)
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    preds_all, preds_all_ = eval_threshold(labels_all, preds_all, preds, edges_pos, edges_neg, adj_rec, test)

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    f1_score_ = f1_score(labels_all, preds_all_)
    acc_score = accuracy_score(labels_all, preds_all_)
    return roc_score, ap_score, f1_score_, acc_score


def gen_preds(edges_pos, edges_neg, adj_rec):
    preds = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])

    return preds, preds_neg


def eval_threshold(labels_all, preds_all, preds, edges_pos, edges_neg, adj_rec, test):
    for i in range(int(0.5 * len(labels_all))):
        if preds_all[2*i] > 0.95 and preds_all[2*i+1] > 0.95:
            preds_all[2*i] = max(preds_all[2*i], preds_all[2*i+1])
            preds_all[2*i+1] = preds_all[2*i]
        else:
            preds_all[2*i] = min(preds_all[2*i], preds_all[2*i+1])
            preds_all[2*i+1] = preds_all[2*i]
    fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    preds_all_ = []
    for p in preds_all:
        if p >=optimal_threshold:
            preds_all_.append(1)
        else:
            preds_all_.append(0)
    return preds_all, preds_all_

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_model', type=str, default='allenai/scibert_scivocab_uncased', help='Desired language model cs_model.')
    parser.add_argument('--input_file', type=str, default='smiles2caption_example.txt', help='path where test generations are saved')
    parser.add_argument('--text_trunc_length', type=str, default=512, help='cs_model maximum length')
    args = parser.parse_args()
    evaluate(args.text_model, args.input_file, args.text_trunc_length)
