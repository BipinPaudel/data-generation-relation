import os
from openai import OpenAI
import pandas as pd 
client = OpenAI(api_key = "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")
import time
import re
model_id = 'gpt-3.5-turbo-0125'
# model_id = 'gpt-4-1106-preview'

# from generate_llama import LLAMA2

class Generate_Psuedo_Labels:
    def __init__(self, dataset= None, labelpath=None,  BATCH_SIZE=None, sent_model=None, sent_train=None, label_idx=None, mod_sent_train=None, mod_pmpt_train=None, b_labels=None, temp=0.0):
        self.dataset = dataset
        self.labelpath = labelpath
        self.batch = BATCH_SIZE
        self.model = sent_model
        self.train = sent_train
        self.label_idx = label_idx
        self.mod_sent_train = mod_sent_train
        self.mod_pmpt_train = mod_pmpt_train
        self.b_labels = b_labels
        self.relation_dict_whole = {
            1: 'Component-Whole(e2,e1)',
            2: 'Component-Whole(e1,e2)',
            3: 'Instrument-Agency(e2,e1)',
            4: 'Instrument-Agency(e1,e2)',
            5: 'Member-Collection(e2,e1)',
            6: 'Member-Collection(e1,e2)',
            7: 'Cause-Effect(e2,e1)',
            8: 'Cause-Effect(e1,e2)',
            9: 'Entity-Destination(e2,e1)',
            10: 'Entity-Destination(e1,e2)' ,
            11: 'Content-Container(e2,e1)' ,
            12: 'Content-Container(e1,e2)' ,
            13: 'Message-Topic(e2,e1)' ,
            14: 'Message-Topic(e1,e2)' ,
            15: 'Product-Producer(e2,e1)' ,
            16: 'Product-Producer(e1,e2)' ,
            17: 'Entity-Origin(e2,e1)' ,
            18: 'Entity-Origin(e1,e2)' ,
            0: 'Other'
        }

        self.relation_dict = {
            "Component-Whole(e2, e1)": 1,
            "Component-Whole(e1, e2)": 2,
            "Instrument-Agency(e2, e1)": 3,
            "Instrument-Agency(e1, e2)": 4,
            "Member-Collection(e1, e2)": 5,
            "Member-Collection(e2, e1)": 6,
            "Cause-Effect(e2, e1)": 7,
            "Cause-Effect(e1, e2)": 8,
            "Entity-Destination(e1, e2)": 9,
            "Entity-Destination(e2, e1)": 10,
            "Content-Container(e1, e2)": 11,
            "Content-Container(e2, e1)": 12,
            "Message-Topic(e1, e2)": 13,
            "Message-Topic(e2, e1)": 14,
            "Product-Producer(e2, e1)": 15,
            "Product-Producer(e1, e2)": 16,
            "Entity-Origin(e1, e2)": 17,
            "Entity-Origin(e2, e1)": 18,
            "No-Relation": 0
        }

        self.temperature = temp

    def ChatGPT_conversation(self, conversation):
        retries = 5
        while retries > 0: 
            try:
                response =  client.chat.completions.create(
                    model=model_id,
                    temperature=self.temperature,
                    messages=conversation,
                    max_tokens=128,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                conversation.append({'role' : response.choices[0].message.role, 'content':  response.choices[0].message.content})
                return conversation

            except Exception as e:  
                print(e)   
                print('Timeout error, retrying...')    
                retries -= 1    
                time.sleep(5)
        
        print("API Not responding after 5 tries")
        conversation.append({"role": "user", "content": "No Response"})
        return conversation 

    def sent_emb_entity(self, text, label):
        pattern = r'< e1 >(.*?)< \/ e1 >.*?< e2 >(.*?)< \/ e2 >'
        matches = re.search(pattern, text)
        if matches:
            e1_entity = matches.group(1).lower()
            e2_entity = matches.group(2).lower()
            # Remove e1 and e2 parts from the text
            modified_text = text.replace("< e1 >","").replace("< / e1 >","").replace("< e2 >","").replace("< / e2 >","").lower()
            if label % 2 != 0:
                new_text = f"Context: {modified_text}. Given, the context, the relation between e2={e2_entity} and e1={e1_entity}"
                ret_text = f"The relation between {e2_entity} and {e1_entity} for the context: {modified_text}"
                return ret_text, new_text
            else:
                new_text = f"Context: {modified_text}. Given, the context, the relation between e1={e1_entity} and e2={e2_entity}"
                ret_text = f"The relation between {e1_entity} and {e2_entity} for the context: {modified_text}"
                return ret_text, new_text

        return text, text


    def get_sim_examples(self,text):
        shot = 16
        results = self.model.search(text, threshold=0.0, top_k=shot+5)

        if results != []:
            ex = []
            try:
                res = results[1:shot]
                for r in res:
                    idx = self.mod_sent_train.index(f'{r[0]}')
                    label = self.label_idx[idx]
                    # for key, value in self.relation_dict.items():
                    #     if key == label:
                    #         our_key = value
                    #         break
                    match = self.mod_pmpt_train[idx]
                    ex.append([match, label])
                    # ex.append([r[0], our_key])
            except Exception as e:
                return ex

            return ex

        return None

    def generate_relation_description(self, text, rel):
        DEFAULT_SYSTEM_PROMPT = f"What are the clues that lead to '{rel}' to be {text}?"
        prompt = DEFAULT_SYSTEM_PROMPT 

        conversation = []
        conversation.append({'role': 'user', 'content': prompt})
        conversation = self.ChatGPT_conversation(conversation) 

        return conversation[-1]['content'].strip()
    
    def gpt_3_predict(self, text, label):
        ret_text, new_text = self.sent_emb_entity(text, label)
        ex = self.get_sim_examples(ret_text)

        # new_text = text.replace("[CLS]", "").replace("[SEP]", "")
        
        system_content = f'''You are an expert in natural language processing, specializing in understanding 
        relationships between entities in text. Your task is to identify and output the relationship 
        between two entities enclosed within the tags <e1> and <e2>'''

        header = f'''Analyze the given test sentence carefully and determine how the entities inside the tags e1 and e2 are related.'''

        footer = "Please output the relationship between entities in the format below:\n <relation> </relation>\n "\
            f"Select only one option from the predefined relations list given in the list below: \n{self.relation_dict}\n" \
            "The relationship in the list denote the flow of relation from entity e1 to e2 or e2 to e1."\
            "Also, output the confidence score (0-1) indicating your certainty about the relationship inside the <confidence> tag."\
            "Use the following demonstrations as examples:"

        #  "Example response format: <relation>causes</relation>\n<confidence>0.9</confidence>"\
        DEFAULT_SYSTEM_PROMPT =  header + footer

        # DEFAULT_SYSTEM_PROMPT += f'''
        # The relation between entity e1 and entity e2 in the given context : {text} is 
        # '''

        if ex != None or ex != []:
            for example in ex:
                rel = self.relation_dict_whole.get(example[1])

                DEFAULT_SYSTEM_PROMPT += f'''
                Example Sentence: {example[0]} is 
                <relation>{rel}</relation>\n<confidence>0.8</confidence>
                '''
        DEFAULT_SYSTEM_PROMPT += f"Test Sentence:\n{new_text}\n is" 

        prompt = DEFAULT_SYSTEM_PROMPT 
        conversation = []
        conversation.append({'role': 'system', 'content': system_content})
        conversation.append({'role': 'user', 'content': prompt})
        conversation = self.ChatGPT_conversation(conversation) 

        return conversation[-1]['content'].strip()


    def data_generate(self, outfile, type): 
        import json 
        texts = json.load(open(self.dataset, "r"))
        labels = json.load(open(self.labelpath,"r"))

        data = [
        {
            "index": i,
            "text": item,
            "label": labels[i]
        }
            for i, item in enumerate(texts)
        ]

        with open(outfile, 'w') as file:
            json.dump(data, file, indent=4)

        with open(outfile, 'r') as file:
            data = json.load(file)

 
        for entry in data:
            if "predicted_label" not in entry:
                text = entry["text"]
                label = entry["label"]

                response = self.gpt_3_predict(text, label)
                print(response)

                if '<relation>' in response:
                    part1, part2 = response.split("</relation>")
                    part1 = part1.replace("<relation>", "")
                    part2 = part2.replace("<confidence>", "").replace("</confidence>", "").strip()
                    part2 = float(part2)*100

                    entry["label_text"] = self.relation_dict_whole[label]
                    entry["predicted_label"] = part1
                    entry["predicted_confidence"] = part2

                    with open(outfile, 'w') as file:
                        json.dump(data, file, indent=4)
        
from simcse import SimCSE
import json

def sent_emb_entity(text, rel):
    pattern = r'<e1>(.*?)<\/e1>.*?<e2>(.*?)<\/e2>'
    matches = re.search(pattern, text)
    if matches:
        e1_entity = matches.group(1).lower()
        e2_entity = matches.group(2).lower()
        # Remove e1 and e2 parts from the text
        modified_text = text.replace("<e1>","").replace("</e1>","").replace("<e2>","").replace("</e2>","").lower()
        if rel % 2 == 0:
            new_text = f"Context: {modified_text}. Given, the context, the relation between e1={e1_entity} and e2={e2_entity}"
            ret_text = f"The relation between {e1_entity} and {e2_entity} for the context: {modified_text}"
            return ret_text, new_text
        else:
            new_text = f"Context: {modified_text}. Given, the context, the relation between e2={e2_entity} and e1={e1_entity}"
            ret_text = f"The relation between {e2_entity} and {e1_entity} for the context: {modified_text}"
            return ret_text, new_text

    return text, text


if __name__ == "__main__":
    sent_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    f = open('/home/n/ngautam/researchscripts/MetaSRE/data/SemEval/train_sentence.json')
    data = json.load(f)

    f2 = open('/home/n/ngautam/researchscripts/MetaSRE/data/SemEval/train_label_id.json')
    data2 = json.load(f2)
    
    mod_sent_train = []
    mod_pmpt_train = []
    sent_train = []

    label_idx = []
    for d in data2:
        label_idx.append(d)

    for index, d in enumerate(data):
        mod = d.split("[CLS]")[1].split("[SEP]")[0]
        sent_train.append(mod)
        ret_text, tex = sent_emb_entity(mod, label_idx[index])
        mod_sent_train.append(ret_text)
        mod_pmpt_train.append(tex)

    
    sent_model.build_index(mod_sent_train)

    filepath = "/home/n/ngautam/researchscripts/#CustomAUMST/data/test_sentence.json"
    labelpath = "/home/n/ngautam/researchscripts/#CustomAUMST/data/test_label_id.json"
    outfile = "/home/n/ngautam/researchscripts/#CustomAUMST/AUM-ST/semeval/test_15shot_label.json"

    gen = Generate_Psuedo_Labels(dataset=filepath, labelpath= labelpath, temp=0.5, sent_model=sent_model,
     mod_sent_train=mod_sent_train, mod_pmpt_train=mod_pmpt_train, sent_train=sent_train, label_idx=label_idx)
    gen.data_generate(outfile, 3)