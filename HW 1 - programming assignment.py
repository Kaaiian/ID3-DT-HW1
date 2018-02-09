# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:03:00 2018

@author: Kaai
"""

import random as rnd
import numpy as np
import pandas as pd
from collections import Counter


# %%
# read in data to train and test tree
df_test = pd.read_csv(r'Datasets_in_homework_1\train.csv', header=None)
df_train = pd.read_csv(r'Datasets_in_homework_1\test.csv', header=None)

# %%
class ID3():

    def __init__(self, df_train, max_depth=100, method='information gain'):
        if df_train.columns.values[-1] != 'label':
            df = df_train.copy()
            df.rename(columns={df.columns.values[-1] : 'label'}, inplace=True)
        else:
            df = df_train.copy()
        self.root_node = df
        self.root_label_mode = self.root_node['label'].mode().values
        self.method = method
        self.max_depth = max_depth
        self.depth = 0
        self.attribute_sets = {}
        for A in df.drop('label', axis=1):
            self.attribute_sets[A] = set(df[A])
            acknowledge = 'no'
        if self.method not in ['information gain', 'majority error']:
            while acknowledge == 'the secret password':
                for n in range(0, 20):
                    print('\nERROR!')
                print("INCORRECT METHOD")
                acknowledge = input('Type to acknowledge the fact that you put the method in incorect..\n\nDo you accept your error(y/n)? \n\n')
                break

    def make_branches(self, split):
        A = list(split.keys())[0]
        if self.depth < self.max_depth:
            self.depth += 1
            for value in set(self.root_node[A]):
                if value in split[A]:
                    leaf = self.evaluate_node(split[A][value])
                    if leaf is not None:
                        split[A][value] = {'leaf': leaf}
                    else:
                        df = split[A][value].drop(A, axis=1)
                        child = self.split_df(df)
                        split[A][value] = child
                        self.make_branches(child)
                else:
                    split[A][value] = {'leaf': rnd.choices(self.root_label_mode)}
            return split
        else:
            for value in set(self.root_node[A]):
                if value in split[A]:
                    leaf = self.evaluate_node(split[A][value])
                    if leaf is not None:
                        split[A][value] = {'leaf': leaf}
                    else:
                        split[A][value] = {'leaf': rnd.choices(split[A][value]['label'].mode().values)}
                else:
                    split[A][value] = {'leaf': rnd.choices(split[A][value]['label'].mode().values)}
        return split

    def evaluate_node(self, node):
        terminal = None
        if len(set(node['label'])) == 1:
            terminal = rnd.choices(node['label'].mode().values)
        elif len(node['label']) == 0:
            terminal = rnd.choices(self.root_label_mode)
        return terminal

    def calc_percent(self, label):
        label = pd.Series(list(label))
        percent = 0
        number_unique_labels = int(len(set(label.values)))
        if number_unique_labels == 1:
            percent = 0
        elif len(label) == 0:
            percent = 0
        else:
            percent = (label.value_counts().sum()-label.value_counts().iloc[1])/label.value_counts().sum()
        return float(percent)

    def calc_majority_error(self, split):
        total_error = {}
        for feature in split:
            majority_error = 0
            if feature != 'label':
                for value in split[feature]:
                    if isinstance(len(split[feature][value]) / len(split), float or int):
                        frac = len(split[feature][value]) / len(split['label'])
                    else:
                        frac = 0
                    fractional_percent= self.calc_percent(split[feature][value]['label'])
                    majority_error += frac * fractional_percent
                total_error[feature] = majority_error
        min_total_error = min(total_error, key=total_error.get)
        self.test = min_total_error
        return min_total_error

    def calc_entropy(self, label):
        label = pd.Series(list(label))
        entropy = 0
        number_unique_labels = int(len(set(label.values)))
        if number_unique_labels == 1:
            entropy += 0
        else:
            for i in range(0, number_unique_labels):
                entropy += - label.value_counts().iloc[i]/label.value_counts().sum() * np.log2(label.value_counts().iloc[i]/label.value_counts().sum())
        return float(entropy)

    def calc_information_gain(self, split):
        gain = {}
        for feature in split:
            entropy_gain = self.calc_entropy(split['label'])
            entropy = 0
            if feature != 'label':
                for value in split[feature]:
                    if isinstance(len(split[feature][value]) / len(split), float or int):
                        frac = len(split[feature][value]) / len(split['label'])
                    else:
                        frac = 0
                    fractional_entropy = self.calc_entropy(split[feature][value]['label'])
                    entropy += frac * fractional_entropy
                    entropy_gain -= frac * fractional_entropy
                gain[feature] = entropy_gain
        self.gain = gain
        max_gain = max(gain, key=gain.get)
        return max_gain

    def get_best_split(self, split):
        if self.method == 'majority error':
            feature = self.calc_majority_error(split)
        else: 
            feature = self.calc_information_gain(split)
        return feature

# node is a dictionary with the key denoting it's parents, and it's associated df as output.
    def split_df(self, df):
        split = {}
        for x in df:
            split[x] = {}
            if x != 'label':
                for y in self.attribute_sets[x]:
                    split[x][y] = self.root_node.loc[df.index[df[x] == y]]
            else:
                split[x] = df['label']
        split = {self.get_best_split(split): split[self.get_best_split(split)]}
        return split


    def create_tree(self):

        for attribute in self.root_node:
            if len(self.root_node.mode()) == 1:
                self.root_node[attribute].fillna(self.root_node[attribute].mode(), inplace=True)
            else:
                self.root_node[attribute].fillna(rnd.choice(self.root_node[attribute].mode()), inplace=True)

        if len(set(self.root_node['label'])) == 1:
            self.leaf_root_node = self.root_node['label'].columns.values

        self.root_split = self.split_df(self.root_node)
        self.trained_tree = self.make_branches(self.root_split)
#        return self.trained_tree

    def climb_branches(self, prediction_row, trained_tree):
        feature = list(trained_tree.keys())[0]
        if feature == 'leaf':
            self.output =  trained_tree[feature]
        else:
            value = prediction_row[feature]
            if value in trained_tree[feature]:
                self.climb_branches(prediction_row, trained_tree[feature][value])
            else:
                self.output =  rnd.choices(self.root_label_mode)
        return self.output

    def predict(self, df_test):
        self.df_test_origional = df_test
        df_test = df_test.copy()
        
        for attribute in self.df_test_origional:
            if len(self.df_test_origional[attribute].mode()) == 1:
                self.df_test_origional[attribute].fillna(self.df_test_origional[attribute].mode(), inplace=True)
            else:
                self.df_test_origional[attribute].fillna(rnd.choice(self.df_test_origional[attribute].mode()), inplace=True)
        self.df_test_copy = df_test

        if isinstance(self.trained_tree, dict) is True:
            df_test['prediction'] = pd.Series({'Prediction': range(0, len(df_test))})
            for i in range(0, len(df_test)):
                prediction_row = df_test.iloc[i]
                value = self.climb_branches(prediction_row, self.trained_tree)
                df_test.iloc[[i], df_test.columns.get_loc('prediction')] = value
        else:
            ''' please train a tree using the create_tree() method'''
        return df_test


# %%
def accuracy(df):
    df = df.copy()
    correct = 0
    wrong = 0
    for act, pred in zip(df[6], df['prediction']):
        if act == pred:
            correct += 1
        else:
            wrong += 1
    percent_correct = correct/(correct+wrong)
    return percent_correct

def run_algorithm(df_train, df_test, max_depth=7, method='information gain'):
    
    tree = ID3(df_train, max_depth, method)
    tree.create_tree()
    
    predict_train =tree.predict(df_train)
    predict_test = tree.predict(df_test)
    return [['max depth: ' + str(max_depth), 'method: '+ str(method), 'train accuracy: ' + str(accuracy(predict_train)), 'test accuracy: ' + str(accuracy(predict_test))], [predict_train, predict_test]]
# %%

for depth in range(1,8):
#    depth = 6

    output_IG = run_algorithm(df_train, df_test, max_depth=depth, method='information gain')
    output_ME = run_algorithm(df_train, df_test, max_depth=depth, method='majority error')

    print('IG:', output_IG[0],'\nME: ', output_ME[0])
