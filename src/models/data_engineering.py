import pandas as pd
import numpy as np
import ast


df = pd.read_csv("dataset_1.csv")

# df['softmax_values'] = df['softmax_values'].to_list()

df['soft_list'] = df['softmax_values'].apply(lambda x: ast.literal_eval(x))

df['soft'] = df['soft_list'].apply(lambda x: pd.Series(x).median())

final_df = df[["soft", "pred_labels", "true_labels"]]


final_df.to_csv("./final_1.csv", 
                index=False)