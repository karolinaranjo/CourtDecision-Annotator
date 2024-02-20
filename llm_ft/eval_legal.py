import pandas as pd

correct_num, total_num = 0., 0.
df = pd.read_csv('data/legal/val2_gpt.csv')
for i in range(len(df)):
    if pd.isna(df.iloc[i, 1]) or pd.isna(df.iloc[i, 2]):
        continue
    if df.iloc[i, 1].strip() == df.iloc[i, 2].strip():
        correct_num += 1
    total_num += 1
print('correct_num =', correct_num)
print('total_num =', total_num)
print('accuracy =', correct_num/total_num)