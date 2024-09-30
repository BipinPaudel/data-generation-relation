import re


def extract_between_tags(rephrased):
    match = re.search(r'\[CLS\](.*?)\[SEP\]', rephrased)
    if match:
        extracted_text = match.group(0)  # group(0) returns the whole match with the tags
        return extracted_text
    return ''


def sanity_check(json_list, key='rephrase_response'):
    error_ids = []
    for js in json_list:
        response = js[key]
        parsed_text = extract_between_tags(response)
        original_text = js['text']
        is_valid, a,b,c,d = compare_two(original_text, parsed_text)
        if not is_valid:
            error_ids.append((js['id'], a,b,c,d))
    return error_ids


def compare_two(original, rephrased):
    e1_original = re.search(r'<e1>(.*?)</e1>', original).group(1)
    e2_original = re.search(r'<e2>(.*?)</e2>', original).group(1)
    
    e1_rephrased = re.search(r'<e1>(.*?)</e1>', rephrased)
    e2_rephrased = re.search(r'<e2>(.*?)</e2>', rephrased)
    if not e1_rephrased or not e2_rephrased: return False, '', '', '', ''
    e1_rephrased = e1_rephrased.group(1)
    e2_rephrased = e2_rephrased.group(1)
    # print(f'{e1_original}:{e1_rephrased}')
    # print(f'{e2_original}:{e2_rephrased}')
    if not e1_original or not e2_original or not e1_rephrased or not e2_rephrased:
        return False, '', '', '', ''
    
    return (e1_original.strip().lower() == e1_rephrased.strip().lower() and e2_original.strip().lower() == e2_rephrased.strip().lower(), e1_original, e1_rephrased, e2_original, e2_rephrased)

def sent_emb_entity(text, rel, relation_dict_whole):
    pattern = r'<e1>(.*?)<\/e1>.*?<e2>(.*?)<\/e2>'
    matches = re.search(pattern, text)
    if matches:
        e1_entity = matches.group(1).lower()
        e2_entity = matches.group(2).lower()
        # Remove e1 and e2 parts from the text
        modified_text = text.replace("<e1>","").replace("</e1>","").replace("<e2>","").replace("</e2>","").lower()
        relation = relation_dict_whole.get(rel)
        check = True if "(e1, e2)" in relation else False 
        if check:
            new_text = f"Context: {modified_text}. Given, the context, the relation between e1={e1_entity} and e2={e2_entity}"
            ret_text = f"The relation between {e1_entity} and {e2_entity} for the context: {modified_text}"
            return ret_text, new_text
        else:
            new_text = f"Context: {modified_text}. Given, the context, the relation between e2={e2_entity} and e1={e1_entity}"
            ret_text = f"The relation between {e2_entity} and {e1_entity} for the context: {modified_text}"
            return ret_text, new_text

    return text, text

