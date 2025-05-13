import os
import json
import random
import pandas as pd
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from bleurt import score

# the evaluation is done per model
model_name = "gpt-4o"  # deepseek-ai/DeepSeek-V3
model_name = model_name.split("/")[1].lower() if "/" in model_name else model_name
print(model_name)
prompts = ['constraint-aware', 'sheng et al']

# 1. Organising the data (saving as CSV file for inspection) for manual evaluation of constraints violation:
# save_folder = os.path.join('./data/manual evaluation', model_name)

# for prompt in prompts:
#     number_of_utterances = 1
#     number_of_endpoints = 1
    
#     utterance_folder = os.path.join('./data/manual evaluation', model_name, prompt, 'utterances' if prompt == 'constraint-aware' else '') 
#     print(utterance_folder)

#     rows = []
#     for root, _, files in os.walk(utterance_folder):
#         for filename in sorted(files):
#             file_path = os.path.join(root, filename)
#             with open(file_path, 'r') as f:
#                 data = json.load(f)
#                 api_name = data["name"]
                
#                 for endpoint in data['api_methods']:
#                     api_method_name = endpoint['name']

#                     for utterance in endpoint['utterances']:
#                         utt = utterance['utterance']
#                         par = utterance['parameters']

#                         rows.append({
#                             'utterance index': number_of_utterances,
#                             'endpoint index': number_of_endpoints,
#                             'API name': api_name,
#                             'API method name': api_method_name,
#                             'utterance': utterance['utterance'],
#                             'parameters': utterance['parameters'],
#                             'REQUIRED PARAMETERS': '',
#                             'VALUES': '',
#                             'FORMAT': '',
#                             'ID': '',
#                             'TECHNICAL': '',
#                             'INTER-DEPENDENCY': '',
#                             'SEMANTIC RELEVANT': ''
#                         })
#                         number_of_utterances+=1
#                     number_of_endpoints+=1

#     df = pd.DataFrame(rows)
#     file_name = prompt + '.csv'
#     file_path = os.path.join(save_folder, file_name)
#     df.to_csv(file_path, index=False)



# 2.1 Cosine similarity using embeddings sentence models: the problem is that it highlights similar things in the reference (which highlights the api name) rather than fluency and naturalness
cosine_similarity = [[], []] # one for each prompt
model = SentenceTransformer('all-mpnet-base-v2')

for prompt_index, prompt in enumerate(prompts):
    utterance_folder = os.path.join('./data/manual evaluation', model_name, prompt, 'utterances' if prompt == 'constraint-aware' else '')

    for root, _, files in os.walk(utterance_folder):
        for filename in sorted(files):
            file_path = os.path.join(root, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                api_description = data['description']

                for endpoint in data['api_methods']:
                    #print(data['name'], endpoint['name'])
                    # parameter_list = []
                    # for parameter in endpoint['parameters']:
                    #     parameter_list.append(parameter['name'])
                    # reference = [str(api_description + '. ' +  endpoint['description']) + '. Information that can be used: ' + str(parameter_list)]
                
                    reference = [str(api_description + '. ' +  endpoint['description'])]
                    utterances = [x['utterance'] for x in endpoint['utterances']]

                    # computing embeddings and similarity
                    reference_embedding = model.encode(reference, convert_to_tensor=True)
                    utterance_embedding = model.encode(utterances, convert_to_tensor=True)
                    cosine_scores = util.cos_sim(reference_embedding, utterance_embedding)[0]

                    #print(cosine_scores)
                    cosine_similarity[prompt_index].append(round(cosine_scores.mean().item(), 4))

                    # for idx, (candidate, score) in enumerate(zip(utterances, cosine_scores)):
                    #     print(f"{idx+1:2d}. Score: {score:.4f} | {candidate}")

print()
print('average cosine similarity:')
for cs, prompt in zip(cosine_similarity, prompts):
    print(prompt + ':', round(sum(cs)/len(cs), 4), '   -   ', cs)

# 2.2 BLEURT similarity: results similar with the one shown in the utterance.
bleurt_similarity = [[], []] # one for each prompt
checkpoint = "/home/vitor/Documents/phd/bleurt/BLEURT-20"
scorer = score.BleurtScorer(checkpoint)

for prompt_index, prompt in enumerate(prompts):
    utterance_folder = os.path.join('./data/manual evaluation', model_name, prompt, 'utterances' if prompt == 'constraint-aware' else '')

    for root, _, files in os.walk(utterance_folder):
        for filename in sorted(files):
            file_path = os.path.join(root, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                api_description = data['description']

                for endpoint in data['api_methods']:                
                    reference = [str(api_description + '. ' +  endpoint['description'])]
                    utterances = [x['utterance'] for x in endpoint['utterances']]
                    scores = scorer.score(references=len(utterances)*reference, candidates=utterances)
                    bleurt_similarity[prompt_index].append(round(sum(scores)/len(scores),4))

                    # for idx, (candidate, score) in enumerate(zip(utterances, scores)):
                    #     print(f"{idx+1:2d}. Score: {score:.4f} | {candidate}")


print()
print('BLEURT:')                
for prompt, bs in zip(prompts, bleurt_similarity):
    print(prompt + ':', round(sum(bs)/len(bs), 4), '   -   ', bs)


# 3. parameter coverage and parameter combination coverage
# I will have to remove the ones that are technical and compare for a fair evaluation. 
parameter_coverage = [[], []]
parameter_combination_coverage = [[], []]

# 3.1. storing parameters to compare with
parameters = []
utterance_folder = os.path.join('./data/manual evaluation', model_name, 'constraint-aware/utterances')

for root, _, files in os.walk(utterance_folder):
    for filename in sorted(files):
        file_path = os.path.join(root, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
            api_description = data['description']

            for endpoint in data['api_methods']:
                parameters_not_technical = [param["name"] for param in endpoint['parameters'] if not param.get('constraints') or not param['constraints'].get('technical')]
                parameters.append(parameters_not_technical) 


# 3.2. computing parameter coverage and parameter_combination_coverage
for prompt_index, prompt in enumerate(prompts):
    endpoint_index = 0
    utterance_folder = os.path.join('./data/manual evaluation', model_name, prompt, 'utterances' if prompt == 'constraint-aware' else '')

    for root, _, files in os.walk(utterance_folder):
        for filename in sorted(files):
            file_path = os.path.join(root, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                for endpoint in data['api_methods']:
                    parameters_used = [list(utterances['parameters'].keys()) for utterances in endpoint['utterances']]              # a list where each item is a list with all the parameters used within the given utterance 
                    
                    # computing parameter coverage per method
                    all_parameters_used = list(set([item for sublist in parameters_used for item in sublist]))                      # a list with all the parameter used
                    all_parameters_used = [param for param in all_parameters_used if param in parameters[endpoint_index]]           # removing technical parameters for fairer comparison between methods
                    parameter_coverage[prompt_index].append(round(len(all_parameters_used) / len(parameters[endpoint_index]), 4))   # computing coverage (number of parameters used/total parameters)

                    # computing number of unique parameter combination
                    combination_parameters = [[param for param in utterance if param in parameters[endpoint_index]] for utterance in parameters_used]   # removing technical parameters for fairer comparison
                    unique_sets = {frozenset(lst) for lst in combination_parameters if lst != []}
                    parameter_combination_coverage[prompt_index].append(len(unique_sets))

                    endpoint_index+=1


print()
# this is the average parameter coverage!
print('average parameter coverage:')
for prompt, pc in zip(prompts, parameter_coverage):
    print(prompt + ':', round(sum(pc)/len(pc), 4), '   -   ', pc)

print()
print('number of unique parameter combination:')
for prompt, pcc in zip(prompts, parameter_combination_coverage):
    print(prompt + ':', sum(pcc), '   -   ', pcc)