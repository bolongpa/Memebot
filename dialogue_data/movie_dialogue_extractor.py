"""python script to reorganize the dialogues from movies. results put into csv file delimited by semicolon."""

import csv

with open('extracted_dialogue.csv', 'w+') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(['#1', '#2'])

    with open('movie_lines.txt', 'r', encoding="utf-8", errors="ignore") as f1:
        lines1 = f1.readlines()
        line_dic = {}
        for i, l in enumerate(lines1):
            s = l.split(' +++$+++ ')
            line_dic[s[0]] = s[-1].strip('\n')

        with open('movie_conversations.txt', 'r', encoding="utf-8") as f2:
            lines2 = f2.readlines()
            dialogue_code = []
            for i, s in enumerate(lines2):
                temp = lines2[i].split(' +++$+++ ')[-1].strip('\n[]\'').split('\', \'')
                for j in range(len(temp)-1):
                    dialogue_code.append([temp[j], temp[j+1]])

            # extract dialogue to file
            for p in dialogue_code:
                writer.writerow([line_dic[p[0]], line_dic[p[1]]])
