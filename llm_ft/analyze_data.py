import json
import pandas as pd


input_list, label_list, split_list = [], [], []
for split in ['train', 'val', 'test']:
    with open(f'data/legal/2024-02-12/{split}.json', 'r') as f:
        lines = json.load(f)
    for line in lines:
        input_list.append(len(line['instruction'].split())+len(line['input'].split()))
        label_list.append(line['output'].lower())
        split_list.append(split)
label_set = set(label_list)
print(len(label_set))
print(label_set)

df = pd.DataFrame()
df['label'] = label_list
df['input_length'] = input_list
df['split'] = split_list
train_total, val_total, test_total = 0, 0, 0
with open('data/legal/2024-02-12/data_statistics.txt', 'w') as f:
    f.write(f'label\ttrain\tval\ttest\tavg_length\n')
    for label in ["encabezado", "antecedentes", "pretensiones", "intervenciones", "intervención del procurador", 
                "norma(s) demandada(s)", "actuaciones en sede revisión","pruebas", "audiencia(s) pública(s)",
                "competencia", "consideraciones de la corte", "síntesis de la decisión", "decisión", "firmas", 
                "salvamento de voto", "sin sección"]:
        avg = df[df['label']==label]['input_length'].mean()
        total = df[df['label']==label]
        train = len(total[total['split']=='train'])
        train_total += train
        val = len(total[total['split']=='val'])
        val_total += val
        test = len(total[total['split']=='test'])
        test_total += test
        f.write(f'{label}\t{train}\t{val}\t{test}\t{round(avg, 2)}\n')
print('train_total:', train_total)
print('val_total:', val_total)
print('test_total:', test_total)