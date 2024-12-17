import csv
import pandas as pd
df = pd.read_csv('../../tmp/ml-1m/sasrec_format.csv')

print(df)
# print(df.describe())

def get_last_item(sequence_str):
    sequence_str = sequence_str.strip('"') 
    sequence = sequence_str.split(',')  
    return sequence[-1]

def get_past_item(sequence_str):
    sequence_str = sequence_str.strip('"') 
    sequence = sequence_str.split(',')  
    return sequence[:-1]

df['last_item'] = df['sequence_item_ids'].apply(get_last_item)
df['past_item'] = df['sequence_item_ids'].apply(get_past_item)

all_past_items = set()
for items in df['past_item']:
    all_past_items.update(items)  # 将这一行的 past_item 列表元素加入集合

# 检查各行的 last_item 是否在 all_past_items 中出现过
# 若不出现则说明该 last_item 是全新未在前面出现过的项
df['is_last_item_never_appeared'] = ~df['last_item'].isin(all_past_items)

# 打印出那些 last_item 从未在任何 past_item 中出现过的行
print(df[df['is_last_item_never_appeared']][['index','user_id','last_item']])