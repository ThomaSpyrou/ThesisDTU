import pandas as pd
import numpy as np
import ast


df = pd.read_csv("dataset.csv")

# df['softmax_values'] = df['softmax_values'].to_list()

df['soft_list'] = df['softmax_values'].apply(lambda x: ast.literal_eval(x))
# df['soft'] = df['soft_list'].apply(ast.literal_eval)

print(type(df.soft_list[0]))
df['pred_soft'] = df['pred_labels'].apply(lambda x: (df['soft_list'][0][x]))
df['true_soft'] = df['true_labels'].apply(lambda x: (df['soft_list'][0][x]))
# df['soft'] = df['soft_list'].apply(lambda x: pd.Series(x).median())

final_df = df[["pred_labels", "true_labels", "pred_soft", 'true_soft', "soft_list"]]


final_df.to_csv("./final_1.csv", 
                index=False)