import json

with open('./fb_raw_data/personachat_self_original.json', 'r') as f:
    reader = json.load(f)
    train = reader['train']
    valid = reader['valid']

    with open('fb_train', 'w') as f_train:
        for i in train:
            # print(i['utterances'][-1]['history'])
            for j in range(len(i['utterances'][-1]['history'])-1):
                if i['utterances'][-1]['history'][j] != '__ SILENCE __':
                    json.dump([i['utterances'][-1]['history'][j], i['utterances'][-1]['history'][j+1]], f_train)

    with open('fb_valid', 'w') as f_valid:
        for i in valid:
            for j in range(len(i['utterances'][-1]['history'])-1):
                json.dump([i['utterances'][-1]['history'][j], i['utterances'][-1]['history'][j+1]], f_valid)




'''
json structure:

root---"train"- 0   -----"personality"-[]
     |          .      |_"utterances"-- i --"candidates"-[] (with fixed length of 20)
     |          .                       .  |_"history"-[] of length 2*i+1
     |          .                       .
     |          17877                   
     |
     |_"valid"- 0   ----- (the same to "train")
                .
                .
                .
                999
'''