import os
import json
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import collections




dataset_selection = 0
model_selection = 0  # only need for inference with ICL

dataset_name = ['TGQA', 'TimeQA_easy', 'TimeQA_hard', 'TempReason_l2', 'TempReason_l3'][dataset_selection]
model_name = ['gpt-3.5-turbo', 'gpt-4-1106-preview', 'Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf'][model_selection]


f_SFT_TGLLM = 1
f_inference_ICL = 0



if f_SFT_TGLLM:
    folder_path = f'../results/{dataset_name}_TGSR'

if f_inference_ICL:
    folder_path = f'../results/{dataset_name}_ICL_{model_name}'


num_question_cat = 1
if dataset_name == 'TGQA':
   num_question_cat = 9



EM_dict = {}
f1_score_dict = {}


for i in range(num_question_cat):
    EM_dict[i] = [0, 0]
    f1_score_dict[i] = []



def calculate_EM(a_gold, a_pred):
    return a_gold.replace(' ', '').lower() == a_pred.replace(' ', '').lower()


def calculate_F1(a_gold, a_pred):
    gold_toks = word_tokenize(a_gold)
    pred_toks = word_tokenize(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1



def parse_generation(pred):
    for start_identifier in ['Answer:', 'answer is']:
        if start_identifier in pred:
            pred = pred.split(start_identifier)[-1].strip()
            break

    for end_identifier in ['Test:']:
        if end_identifier in pred:
            pred = pred.split(end_identifier)[0].strip()
        break

    if '\n' in pred:
        pred = pred.split('\n')[-1].strip()

    if '(' in pred:
        pred = pred[len(pred.split('(')[0]) + 1:]

    if ')' in pred:
        pred = pred[:- (len(pred.split(')')[-1]) + 1)]

    if len(pred)>0 and pred[-1] in [')', '.']:
        pred = pred[:-1]

    pred = pred.strip()
    return pred





num_test_samples = 10000
for i in range(num_test_samples):
    file_path = folder_path + f'/{str(i)}.json'
    if not os.path.exists(file_path):
        continue

    with open(file_path) as json_file:
        data = json.load(json_file)

    pred = data['prediction'].strip()
    pred = parse_generation(pred)

    gts = data['A']
    gts = [gt[1:-1].strip() if gt[0] == '(' and gt[-1] == ')' else gt for gt in gts]

    if data['Q-Type'] is None:
        data['Q-Type'] = 0

    cur_f1_score = [calculate_F1(pred, gt) for gt in gts]
    f1_score_dict[data['Q-Type']].append(max(cur_f1_score))

    cur_EM = [calculate_EM(pred, gt) for gt in gts]
    EM_dict[data['Q-Type']][0] += max(cur_EM)
    EM_dict[data['Q-Type']][1] += max(cur_EM)




print('EM:')

EM_dict = []
for i in range(num_question_cat):
    if EM_dict[i][1] > 0:
        EM_dict[i][0] = EM_dict[i][0]/EM_dict[i][1]

    print(i, cnt_correct[i], EM_dict[i])

print(np.mean([EM_dict[i][0] for i in range(num_question_cat)]))



print('\nF1 score:')
print(np.mean([np.mean(f1_score_dict[i]) for i in range(num_question_cat)]))