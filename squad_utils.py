import json

def squad_parse(squad_file):
    texts = []
    labels = []
    labels_index = {}
    labels_rindex = {}
    datasets = squad_load(squad_file)  
    stats = {}
    stats['datasets'] = 0
    stats['paragraphs'] = 0
    stats['questions'] = 0
    stats['answers'] = 0
    for dataset in datasets:
        stats['datasets'] += 1
        for paragraph in dataset['paragraphs']:
            stats['paragraphs'] += 1
            for qas in paragraph['qas']:
                stats['questions'] += 1
                for answer in qas['answers']:
                    label_id = labels_rindex.get(answer['text'])
                    if label_id == None:
                        label_id = len(labels_index)
                        labels_index[label_id] = answer['text']
                        labels_rindex[answer['text']] = label_id
                    texts.append(qas['question'])
                    labels.append(label_id)
                    stats['answers'] += 1
    print stats
    return texts, labels, labels_index

def squad_load(f):
    with open(f) as json_data:
        data = json.load(json_data)
    return data['data']
