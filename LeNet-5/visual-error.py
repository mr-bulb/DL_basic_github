#!/usr/bin/env Python
# coding=utf-8

##************************************************************************** 
##-------------------------------------------------------------------------- 
## This is the code for visualizing the error of LeNet-5 Model
## 
## Author         Hao Zeng 
## version        dev 0.1.0 
## Date           2018.07.02 
##
## Description    这是用于可视化训练集和测试集上error曲线的函数
##
##-------------------------------------------------------------------------- 
##************************************************************************** 

import sys
reload(sys)
sys.setdefaultencoding('gbk')

import matplotlib.pyplot as plt
import numpy as np
import math
# plt.switch_backend('agg')

# -----------------------------------------------------
# Parse data in every line and put them into dictionary
# -----------------------------------------------------
def parse_text_and_put_into_dict(dict_train, dict_test, text_list):
    idx = 0
    rows = text_list
    while idx < len(rows):

        if len(rows[idx]) != 0:  # not null string

            # Location the first line(hyperparameter setting) in the train output
            start_idx = rows[idx].find('Namespace')
            if start_idx != -1:
                row = rows[idx]

                # Locate the learning rate position in this line
                lr_idx = row.find('lr=')
                if lr_idx != -1:

                    # Get Learning Rate in this output block
                    i = lr_idx+len('lr=')
                    while row[i] != ',' and row[i] != ')' :
                        i += 1

                    str_lr = row[lr_idx+len('lr='): i]
                    print('Learning Rate : {0}'.format(str_lr))
                else:
                    print(
                        'Not Find Learning Rate in the Hyperparameter Setting, May be There are Some Syntax Errors in the Output FIle')
                    break

                # Put all data in corresponding dict
                idx += 1  # Move to Next Line

                list_train_error = []
                list_test_error = []

                while(len(rows[idx]) != 0):
                    row = rows[idx]
                    split_row = row.split(" ")
                    print float(split_row[0]), float(split_row[1])

                    # list_train_error.append(1-float(split_row[0]))
                    # list_test_error.append(1-float(split_row[1]))

                    list_train_error.append(math.log10(1-float(split_row[0])))
                    list_test_error.append(math.log10(1-float(split_row[1])))


                    idx += 1

                # add list to set
                if(test_batchsize_train_error.has_key(str_lr) == False):            # Key not exist, create
                    test_batchsize_train_error[str_lr] = []
                    test_batchsize_train_error[str_lr].append(list_train_error)
                else:                                                       # Key already exist, append
                    test_batchsize_train_error[str_lr].append(list_train_error)

                if(test_batchsize_test_error.has_key(str_lr) == False):            # Key not exist, create
                    test_batchsize_test_error[str_lr] = []
                    test_batchsize_test_error[str_lr].append(list_test_error)
                else:                                                       # Key already exist, append
                    test_batchsize_test_error[str_lr].append(list_test_error)

            else:
                idx = idx+1

        # This line is a null string
        else:
            idx = idx+1


# -----------------------------------------------------
# Draw Line Chart for Each Test Batch Size
# -----------------------------------------------------
def draw_line_chart_for_test_batchsize(test_batchsize_dict_train_error, title_name):

    sort_keys = test_batchsize_dict_train_error.keys()
    sort_keys.sort()

    for key in sort_keys:

        # initial all element to zero
        avg_lr_train_error = [0 for x in range(len(test_batchsize_dict_train_error[key][0]))]
        set_of_train_list = test_batchsize_dict_train_error[key]

        for sub_list_idx in range(len(set_of_train_list)):
            for elem_idx in range(len(set_of_train_list[sub_list_idx])):
                avg_lr_train_error[elem_idx] += 1.0/len(set_of_train_list) * set_of_train_list[sub_list_idx][elem_idx]

        x = range(1, len(avg_lr_train_error)+1)

        plt.plot(x, avg_lr_train_error, label='learning rate={0}'.format(key)) 
        plt.xlabel('epoch') 
        plt.ylabel('log10(error)') 
        plt.title(title_name) 
        plt.legend() 


# ----------------------------------------
# Open File and Read Data
# ----------------------------------------
f = open("LeNet-5_accuracy.txt", 'r')
data = f.read()
rows = data.split('\n')


# different test batch size correspond to different set of list
# the key of dict is learning
# the value of dict are a dict contain five repeat experiments' data in the same hyperparameter

test_batchsize_train_error = {}
test_batchsize_test_error = {}

parse_text_and_put_into_dict(test_batchsize_train_error, test_batchsize_test_error, rows)


# ----------------------------------------
# Draw Line Chart of Training Set

plt.figure()
draw_line_chart_for_test_batchsize(test_batchsize_train_error, 'Learning Process --- Training Set')

# ----------------------------------------
# Draw Line Chart of Test Set

plt.figure()
draw_line_chart_for_test_batchsize(test_batchsize_test_error, 'Learning Process --- Test Set')

plt.show()
