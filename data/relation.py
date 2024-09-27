import pandas as pd
import json 



def read_csv(filename):
    df = pd.read_csv(filename)
    return df


def read_json(filename):
    with open(filename, 'r') as file:
        obj = json.load(file)
    return obj
    
    
def read_relation_data(filename='data/train.csv'):
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