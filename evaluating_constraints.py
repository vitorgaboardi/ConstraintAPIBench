import os
import json
import random

# auxiliar code to check which OAS actually have constraints
model="gpt-4.1-mini"

if model == "gpt-4.1-mini" or model == "gpt-4.1":
  constraint_folder = './dataset/Constraint-based prompt/'+model+'/constraints/'
else:
  model_name = model.split('/')[1]
  constraint_folder = './dataset/Constraint-based prompt/'+model_name+'/constraints/'

values_constraint = []
specific_constraint = []
format_constraint = []
id_constraint = []
api_related_constraint = []
inter_dependency_constraint = []

# iterating over the categories
categories = sorted(os.listdir(constraint_folder))
for category_index, category in enumerate(categories):
    category_path = os.path.join(constraint_folder, category)
    #print(category_path)
    
    for root, _, files in os.walk(category_path):
        for filename in files:
            file_path = os.path.join(category_path, filename)

            with open(file_path, 'r') as f:
                data = json.load(f)

                for api_method_index, api_method in enumerate(data['api_list']):
                    api_method_name = api_method['name']
                    for parameter in api_method['parameters']:
                        if 'constraints' in parameter:
                            if parameter['constraints'] != {}:
                                constraint = parameter['constraints']

                                if 'values' in constraint:
                                    if ('min' in constraint['values'] or 'max' in constraint['values']) and filename+'-'+api_method_name not in values_constraint:
                                        values_constraint.append(filename+'-'+api_method_name)

                                    if 'specific' in constraint['values'] and filename+'-'+api_method_name not in specific_constraint:
                                        specific_constraint.append(filename+'-'+api_method_name)
                                
                                if 'format' in constraint and filename+'-'+api_method_name not in format_constraint:
                                    format_constraint.append(filename+'-'+api_method_name)

                                if 'api_related' in constraint and filename+'-'+api_method_name not in api_related_constraint:
                                    api_related_constraint.append(filename+'-'+api_method_name)

                                if 'id' in constraint and filename+'-'+api_method_name not in id_constraint:
                                    id_constraint.append(filename+'-'+api_method_name)

                                if 'inter-dependency' in constraint and filename+'-'+api_method_name not in inter_dependency_constraint:
                                    inter_dependency_constraint.append(filename+'-'+api_method_name)


print()
print('Number of endpoints with the following constraints:')
print('values constraints:', len(values_constraint))
print('specific constraints:',  len(specific_constraint))
print('format constraints:', len(format_constraint))
print('id constraints:', len(id_constraint))
print('technical constraints:', len(api_related_constraint))
print('inter-dependency constraints:', len(inter_dependency_constraint))
print()

random.seed(29) #18,20, 29 it is!
selected_endpoints = []

for list_of_endpoints in [values_constraint, specific_constraint, format_constraint, id_constraint, api_related_constraint, inter_dependency_constraint]:
    filtered_list_of_endpoints = [l for l in list_of_endpoints if l not in selected_endpoints]
    endpoint_samples = random.sample(filtered_list_of_endpoints, k=5)

    selected_endpoints.extend(endpoint_samples)

a = [0, 0, 0, 0, 0, 0]
for endpoint in selected_endpoints:
    if endpoint in values_constraint:
        a[0]+=1
    if endpoint in specific_constraint:
        a[1]+=1
    if endpoint in format_constraint:
        a[2]+=1
    if endpoint in id_constraint:
        a[3]+=1
    if endpoint in api_related_constraint:
        a[4]+=1
    if endpoint in inter_dependency_constraint:
        a[5]+=1
print(a)

print(selected_endpoints)