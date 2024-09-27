import os
from openai import OpenAI
import pandas as pd 

import time
import re

# model_id = 'gpt-4-1106-preview'

# from generate_llama import LLAMA2

class Generate_Psuedo_Labels:
    def __init__(self, dataset= None, BATCH_SIZE=None, sent_model=None, sent_train=None, label_idx=None, mod_sent_train=None, mod_pmpt_train=None, b_labels=None, temp=0.0):
        self.dataset = dataset
        self.batch = BATCH_SIZE
        self.model = sent_model
        self.train = sent_train
        self.label_idx = label_idx
        self.mod_sent_train = mod_sent_train
        self.mod_pmpt_train = mod_pmpt_train
        self.b_labels = b_labels
        # self.relation_dict = {
        #     1: 'Component-Whole(e2,e1)',
        #     2: 'Component-Whole(e1,e2)',
        #     3: 'Instrument-Agency(e2,e1)',
        #     4: 'Instrument-Agency(e1,e2)',
        #     5: 'Member-Collection(e2,e1)',
        #     6: 'Member-Collection(e1,e2)',
        #     7: 'Cause-Effect(e2,e1)',
        #     8: 'Cause-Effect(e1,e2)',
        #     9: 'Entity-Destination(e2,e1)',
        #     10: 'Entity-Destination(e1,e2)' ,
        #     11: 'Content-Container(e2,e1)' ,
        #     12: 'Content-Container(e1,e2)' ,
        #     13: 'Message-Topic(e2,e1)' ,
        #     14: 'Message-Topic(e1,e2)' ,
        #     15: 'Product-Producer(e2,e1)' ,
        #     16: 'Product-Producer(e1,e2)' ,
        #     17: 'Entity-Origin(e2,e1)' ,
        #     18: 'Entity-Origin(e1,e2)' ,
        #     0: 'Other'
        # }

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
        shot = 2
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

     
        DEFAULT_SYSTEM_PROMPT = f'''
        Task: I will predict the relationship between two entities, e1 and e2, given the context. The set of pre-defined relationships is as follows:
        Component-Whole(e2, e1) = 1
        Component-Whole(e1, e2) = 2
        Instrument-Agency(e2, e1) = 3
        Instrument-Agency(e1, e2) = 4
        Member-Collection(e1, e2) = 5
        Member-Collection(e2, e1) = 6
        Cause-Effect(e2, e1) = 7
        Cause-Effect(e1, e2) = 8
        Entity-Destination(e1, e2) = 9
        Entity-Destination(e2, e1) = 10
        Content-Container(e1, e2) = 11
        Content-Container(e2, e1) = 12
        Message-Topic(e1, e2) = 13
        Message-Topic(e2, e1) = 14
        Product-Producer(e2, e1) = 15
        Product-Producer(e1, e2) = 16
        Entity-Origin(e1, e2) = 17
        Entity-Origin(e2, e1) = 18
        No-Relation = 0
        If the relation between e1 and e2 doesn't match any of the pre-defined relations, output 0.

        Your response should include:
        Classified Relation: One of the relation numbers listed above based on the given context.
        Confidence Level (0-100%): Your confidence about the prediction as a percentage, represented as Confidence: <percentage>%.
        Example response format:
        (Relation: Entity-Destination(e1, e2) = 9, Confidence: 80%)

        Learn from the following examples and follow the response format:
        '''

        # DEFAULT_SYSTEM_PROMPT += f'''
        # The relation between entity e1 and entity e2 in the given context : {text} is 
        # '''

        if ex != None or ex != []:
            for example in ex:
                rel = self.relation_dict.get(example[1])

                DEFAULT_SYSTEM_PROMPT += f'''
                {example[0]} is 
                (Relation: {rel}={example[1]}, Confidence: 80%)
                '''

        DEFAULT_SYSTEM_PROMPT+= f'''
        {new_text} is '''


        prompt = DEFAULT_SYSTEM_PROMPT 

        conversation = []
        conversation.append({'role': 'user', 'content': prompt})
        conversation = self.ChatGPT_conversation(conversation) 

        return conversation[-1]['content'].strip()


    def data_generate(self, outfile, type):
        
        df = pd.read_csv(self.dataset)
        texts = df['text']
        labels = df['label']


        relations = []
        percentages = []
        # Iterate through each text, pass to the function, and extract the relation and confidence
        for i,text in enumerate(texts):
            label = labels[i]
            response = self.gpt_3_predict(text, label)
            print(response)
            # match = re.search(r"Relation:\s*([\w\-()]+)\s*=\s*(\d+),\s*Confidence:\s*(\d+)%", response)

            match = re.search(r"\s*([\w\-()]+)\s*=\s*(\d+),\s*Confidence:\s*(\d+)%", response)
        
            if match:
                relation = match.group(1)  
                relation_number = int(match.group(2))
                confidence = int(match.group(3)) 

                relations.append(relation_number)
                percentages.append(confidence)
            else:
                relations.append(0)
                percentages.append(0)

            # print(relations, percentages)


        df['relation_temp'+str(type)] = relations
        df['percentage_temp'+str(type)] = percentages

        df.to_csv(outfile, index=False)


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

    sent_train = []
    mod_sent_train = []
    mod_pmpt_train = []

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


    filepath = "/home/n/ngautam/researchscripts/#CustomAUMST/data/train.csv"
    outfile = "/home/n/ngautam/researchscripts/#CustomAUMST/data/train_annotated_oneshot.csv"


    # dataset, BATCH_SIZE, sent_model, sent_train, label_idx, mod_sent_train, mod_pmpt_train, b_labels.item())

    gen = Generate_Psuedo_Labels(dataset=filepath, temp=0.7, sent_model=sent_model)
    gen.data_generate(outfile, 2)

    gen = Generate_Psuedo_Labels(dataset=outfile, temp=0.5, sent_model= sent_model)
    gen.data_generate(outfile, 3)

    # gen = Generate_Psuedo_Labels(dataset=outfile)
    # gen.data_generate(outfile, 1)









    




