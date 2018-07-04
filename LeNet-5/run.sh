#!/bin/bash

# 删除之前保存的txt文件
rm ./LeNet-5_accuracy.txt
rm ./LeNet-5_loss.txt

# 开始循环执行python 程序
repeat_num=1 # 重复执行LeNet-5.py的次数
epochs=5    # 神经网络训练epoch的数量

# -------------------------------------
learning_rate=0.6
echo repeat_num : $repeat_num
echo learning_rate : $learning_rate

for((i=0; i<$repeat_num; i++))
do
	echo loop num : $i # 输出循环到第几次了
    echo python LeNet-5.py  --epochs $epochs --lr $learning_rate # 输出执行的指令

    python LeNet-5.py  --epochs $epochs --lr $learning_rate

    echo
    echo
done

# -------------------------------------
learning_rate=0.4
echo repeat_num : $repeat_num
echo learning_rate : $learning_rate

for((i=0; i<$repeat_num; i++))
do
	echo loop num : $i # 输出循环到第几次了
    echo python LeNet-5.py  --epochs $epochs --lr $learning_rate # 输出执行的指令

    python LeNet-5.py  --epochs $epochs --lr $learning_rate

    echo
    echo
done

# -------------------------------------
learning_rate=0.1
echo repeat_num : $repeat_num
echo learning_rate : $learning_rate

for((i=0; i<$repeat_num; i++))
do
	echo loop num : $i # 输出循环到第几次了
    echo python LeNet-5.py  --epochs $epochs --lr $learning_rate # 输出执行的指令

    python LeNet-5.py  --epochs $epochs --lr $learning_rate

    echo
    echo
done

# -------------------------------------
learning_rate=0.05
echo repeat_num : $repeat_num
echo learning_rate : $learning_rate

for((i=0; i<$repeat_num; i++))
do
	echo loop num : $i # 输出循环到第几次了
    echo python LeNet-5.py  --epochs $epochs --lr $learning_rate # 输出执行的指令

    python LeNet-5.py  --epochs $epochs --lr $learning_rate

    echo
    echo
done

# -------------------------------------
learning_rate=0.01
echo repeat_num : $repeat_num
echo learning_rate : $learning_rate

for((i=0; i<$repeat_num; i++))
do
	echo loop num : $i # 输出循环到第几次了
    echo python LeNet-5.py  --epochs $epochs --lr $learning_rate # 输出执行的指令

    python LeNet-5.py  --epochs $epochs --lr $learning_rate

    echo
    echo
done
