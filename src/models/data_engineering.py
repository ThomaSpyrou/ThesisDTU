import pandas as pd
import numpy as np
import argparse


def clean_target_output():
    true_labels = pd.read_csv('true_labels100.txt', header=None, names=['true_label'])
    predictions = pd.read_csv('predicted_labels100.txt', header = None, names=['prediction'])

    # softmax value needs thalpori
    cols = [i for i in range(100)]
    softmax_values = pd.read_csv('softmax_values100.txt', header=None, sep='\s+', names=cols)
    for index in range(100):
        softmax_values[index] = softmax_values[index].str.extract('(\d+\.\d+)').astype(float)

    softmax_values['soft_list'] = softmax_values.values.tolist()
    softmax_values = softmax_values.drop(cols, axis=1)

    tmp = pd.concat([true_labels, predictions], axis=1, ignore_index=True)
    df = pd.concat([tmp, softmax_values], axis=1, ignore_index=True)
    df = df.rename(columns={0: "true", 1: "pred", 2: "soft"})
    
    df['pred_soft'] = df.apply(take_correct_value_pred, axis=1)
    df['true_soft'] = df.apply(take_correct_value_true, axis=1)

    final_df = df[["pred", "true", "pred_soft", 'true_soft', "soft"]]
    final_df.to_csv("./final.csv", index=False)


def modify_soft(row):
    true_soft = row['true_soft']
    true_index = row['true']
    soft_list =  [0] * 100
    soft_list[true_index] = true_soft

    return str(soft_list)


def take_correct_value_pred(row):
    index_pos = row['pred']
    values_list = row['soft']
    selected_value = values_list[index_pos]

    return selected_value


def take_correct_value_true(row):
    index_pos = row['true']
    values_list = row['soft']
    selected_value = values_list[index_pos]

    return selected_value


def split_dataset(source_path, chunksize=1500000):
    for i,chunk in enumerate(pd.read_csv(source_path, chunksize=chunksize)):
        chunk.to_csv('chunk{}.csv'.format(i), index=False)


if __name__ == '__main__':
    # parsers
    parser = argparse.ArgumentParser(description='Create dataset')
    parser.add_argument('--method', default='create', type=str, help='either split big csv or create')
    parser.add_argument('--chunksize', default=1500000, type=int, help='size of the chunk')
    parser.add_argument('--source_path', default='final.csv', type=str, help='where to store and which folder to split')
    args = parser.parse_args()

    if args.method == 'split':
        print('Splitting')
        split_dataset(source_path=args.source_path, chunksize=args.chunksize)
    else:
        print('preparing df')
        clean_target_output()
