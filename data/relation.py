# import pandas as pd
import json 



def read_csv(filename):
    df = pd.read_csv(filename)
    return df


def read_json(filename):
    with open(filename, 'r') as file:
        obj = json.load(file)
    return obj
    
    
def read_relation_data(dataset='water', data_type='train', filename='data/train.csv'):
    if dataset == 'water':
        sentence_filename = f'data/water/{data_type}/{data_type}_sentence.json'
        sentences = relation_id_dict = read_json(sentence_filename)
        sentence_label_filename = f'data/water/{data_type}/{data_type}_label_id.json'
        sentences_labels = read_json(sentence_label_filename)
        relation2id_filename = f'data/water/{data_type}/relation2id.json'
        sentence_relation2id = read_json(relation2id_filename)
        id_relation_dict = {int(value): key for key,value in sentence_relation2id.items()}
        dataset_list = []
        
        for i, (text, label_id) in enumerate(zip(sentences, sentences_labels)):
            data = {
                "id": i,
                "text": text,
                "label": label_id,
                "relation_name": id_relation_dict[label_id]
            }
            dataset_list.append(data)
        return dataset_list, id_relation_dict

    elif dataset == 'tacred':
        return read_relation_data_from_final_file(dataset, data_type)
        
        
        
    train_df = read_csv(filename)
    relation_id_dict = read_json('data/relation2id.json')
    print(train_df)
    print(relation_id_dict)
    
    id_realtion_dict = {value: key for key,value in relation_id_dict.items()}
    train_df['relation_name'] = train_df['label'].map(id_realtion_dict)
    return json.loads(train_df.to_json(orient='records'))

def write_json_lists_to_file(filename, relations) -> None:
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(relations, f, ensure_ascii=False, indent=4)
    
def write_list_to_file(filename, lst):
    with open(filename, 'w') as json_file:
        json.dump(lst, json_file)

def read_relation_data_from_final_file(dataset='semeval', datatype='train'):
    if dataset == 'tacred':
        filepath = f'data/{dataset}/{datatype}_sentence.json'
        labelpath = f'data/{dataset}/{datatype}_label_id.json'
        relation2id_filename = f'data/{dataset}/relation2id.json'
    elif dataset == 'semeval':
        filepath = "/home/n/ngautam/researchscripts/#CustomAUMST/data/test_sentence.json"
        labelpath = "/home/n/ngautam/researchscripts/#CustomAUMST/data/test_label_id.json"
        relation2id_filename = f'data/relation2id.json'

    
    
    sentences = read_json(filepath)
    sentences_labels = read_json(labelpath)
    sentence_relation2id = read_json(relation2id_filename)
    id_relation_dict = {int(value): key for key,value in sentence_relation2id.items()}
    
    
    
    dataset_list = []
    for i, (text, label_id) in enumerate(zip(sentences, sentences_labels)):
        data = {
            "id": i,
            "text": text,
            "label": label_id,
            "relation_name": id_relation_dict[label_id]
        }
        dataset_list.append(data)
    return dataset_list, id_relation_dict