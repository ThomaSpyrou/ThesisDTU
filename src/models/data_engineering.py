import pandas as pd
import numpy as np
import ast


# df = pd.read_csv("dataset_2.csv")

# # df['softmax_values'] = df['softmax_values'].to_list()

# df['soft_list'] = df['softmax_values'].apply(lambda x: ast.literal_eval(x))
# # df['soft'] = df['soft_list'].apply(ast.literal_eval)

# print(type(df.soft_list[0]))
# df['pred_soft'] = df['pred_labels'].apply(lambda x: (df['soft_list'][0][x]))
# df['true_soft'] = df['true_labels'].apply(lambda x: (df['soft_list'][0][x]))
# # df['soft'] = df['soft_list'].apply(lambda x: pd.Series(x).median())

# final_df = df[["pred_labels", "true_labels", "pred_soft", 'true_soft', "soft_list", "accuracies"]]


# final_df.to_csv("./final_1.csv", 
#                 index=False)


true_labels = pd.read_csv('true_labels.txt', header=None, names=['true_label'])
predictions = pd.read_csv('predicted_labels.txt', header = None, names=['prediction'])

# softmax value needs thalpori
softmax_values = pd.read_csv('softmax_values.txt', header=None, sep='\s+', names=[0, 1, 2 , 3, 4, 5, 6, 7 , 8, 9])
softmax_values[0] = softmax_values[0].str.extract('(\d+\.\d+)').astype(float) * 100 
softmax_values[1] = softmax_values[1].str.extract('(\d+\.\d+)').astype(float) * 100 
softmax_values[2] = softmax_values[2].str.extract('(\d+\.\d+)').astype(float) * 100 
softmax_values[3] = softmax_values[3].str.extract('(\d+\.\d+)').astype(float) * 100 
softmax_values[4] = softmax_values[4].str.extract('(\d+\.\d+)').astype(float) * 100 
softmax_values[5] = softmax_values[5].str.extract('(\d+\.\d+)').astype(float) * 100 
softmax_values[6] = softmax_values[6].str.extract('(\d+\.\d+)').astype(float) * 100  
softmax_values[7] = softmax_values[7].str.extract('(\d+\.\d+)').astype(float) * 100 
softmax_values[8] = softmax_values[8].str.extract('(\d+\.\d+)').astype(float) * 100  
softmax_values[9] = softmax_values[9].str.extract('(\d+\.\d+)').astype(float) * 100 

softmax_values['soft_list'] = softmax_values.values.tolist()
softmax_values = softmax_values.drop([0, 1, 2 , 3, 4, 5, 6, 7 , 8, 9], axis=1)

tmp = pd.concat([true_labels, predictions], axis=1, ignore_index=True)
df = pd.concat([tmp, softmax_values], axis=1, ignore_index=True)
df = df.rename(columns={0: "true", 1: "pred", 2: "soft"})
df['pred_soft'] = df['pred'].apply(lambda x: (df['soft'][0][x]))
df['true_soft'] = df['true'].apply(lambda x: (df['soft'][0][x]))


final_df = df[["pred", "true", "pred_soft", 'true_soft', "soft"]]
final_df.to_csv("./final_val.csv", index=False)
