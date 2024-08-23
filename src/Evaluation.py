import os
import json
import numpy as np
from nltk.tokenize import word_tokenize
import collections
from utlis import *
import argparse




parser = argparse.ArgumentParser()


parser.add_argument('--dataset', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--SFT', action='store_true')
parser.add_argument('--ICL_only', action='store_true')
parser.add_argument('--CoT', action='store_true')
parser.add_argument('--ppl', action='store_true')
parser.add_argument('--no_TG', action='store_true')
parser.add_argument('--prompt_format', type=str, default='plain')

args = parser.parse_args()



######### Config #########

dataset_selection = ['TGQA', 'TimeQA_easy', 'TimeQA_hard', 'TempReason_l2', 'TempReason_l3'].index(args.dataset)
model_selection = ['gpt-3.5', 'gpt-4', 'Llama2-7b', 'Llama2-13b', 'Llama2-70b'].index(args.model)

f_SFT = args.SFT   # whether use SFT
f_inference_ICL = args.ICL_only   # whether use inference with ICL only
f_using_CoT = args.CoT   # whether use CoT
f_ppl = args.ppl   # whether use perplexity
f_no_TG = args.no_TG  # whether to use the temporal graph or original story as context
prompt_format = args.prompt_format  # whether use plain (text) or json as prompt format

###########################


dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]
split_name = ['', '_easy', '_hard', '_l2', '_l3'][dataset_selection]
model_name = ['gpt-3.5-turbo', 'gpt-4-1106-preview', 'Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf'][model_selection]
learning_setting = 'SP' if not f_using_CoT else 'CoT'





if f_SFT:
    strategy = 'TGR' if not f_no_TG else 'storyR'
    folder_path = f'../results/{dataset_name}_{strategy}{split_name}'
else:
    folder_path = f'../results/{dataset_name}_ICL_{learning_setting}{split_name}_{model_name}' if f_inference_ICL else f'../results/{dataset_name}{split_name}_{model_name}'


if f_ppl:
    folder_path += '_ppl'

print(folder_path)




def calculate_EM(a_gold, a_pred):
    # remove spaces and convert to lower case
    return a_gold.replace(' ', '').lower() == a_pred.replace(' ', '').lower()


def calculate_F1(a_gold, a_pred):
    # token-level F1
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
    '''
    Parse the generated answer based on rules
    '''
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




EM_dict = {}
f1_score_dict = {}


num_question_cat = 1
if dataset_name == 'TGQA':
   num_question_cat = 9  # for TGQA, there are 9 question categories, and we use avarage of them to avoid imbalance problem

for i in range(num_question_cat):
    EM_dict[i] = [0, 0]
    f1_score_dict[i] = []


num_test_samples = 10000
for i in range(num_test_samples):
    file_path = folder_path + f'/{str(i)}.json'
    if not os.path.exists(file_path):
        continue

    with open(file_path) as json_file:
        data = json.load(json_file)

    pred = data['prediction'].strip()
    
    # For our framework, we use plain/json format.
    if f_SFT:
        if f_ppl:
            pred = data['prediction']
        else:
            _, pred = parse_TGR_pred(pred, prompt_format)
            pred = '' if pred is None else pred[0]
    else:
        pred = parse_generation(pred)

    gts = data['answer']
    gts = [gt[1:-1].strip() if gt[0] == '(' and gt[-1] == ')' else gt for gt in gts]

    if data['Q-Type'] is None:
        data['Q-Type'] = 0

    cur_f1_score = [calculate_F1(pred, gt) for gt in gts]
    f1_score_dict[data['Q-Type']].append(max(cur_f1_score))

    cur_EM = [calculate_EM(pred, gt) for gt in gts]
    EM_dict[data['Q-Type']][0] += max(cur_EM)
    EM_dict[data['Q-Type']][1] += 1


for i in range(num_question_cat):
    if EM_dict[i][1] > 0:
        EM_dict[i][0] = EM_dict[i][0]/EM_dict[i][1]




print('\nEM:')
print(np.mean([EM_dict[i][0] for i in range(num_question_cat) if EM_dict[i][1] > 0]), sum(EM_dict[i][1] for i in range(num_question_cat)))


# for results based on perplexity, we only need EM since we select the answer from the candidates
if not f_ppl:
    print('\nF1 score:')
    print(np.mean([np.mean(f1_score_dict[i]) for i in range(num_question_cat) if len(f1_score_dict[i]) > 0]), sum(len(f1_score_dict[i]) for i in range(num_question_cat)))