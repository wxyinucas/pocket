#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==============================
# @Time    : 2018/8/19 21:32
# @Author  : Xiaoyu Wang
# @File    : change_to_tex.py
===============================
To change table.txt to the form which can be used by tex.
"""

import re

head = '''\\begin{table}[!htbp]
\centering
\caption{Simulation result $n = 200$}\label{tab:simulation}
\\begin{tabular}{l c c c c}

\\toprule
'''

mid = '''\cline{2-5}
& Bias & ESE & ASE & CP\\\\
\hline'''

bottom = '''\\bottomrule
\end{tabular}
\end{table}'''


def extract(sentence):
    settings = re.findall('(?<=\[).*?(?=\])', sentence)
    if len(settings):
        par = [list(map(float, setting.split())) for setting in settings]
        return par
    else:
        output = re.findall('[\d\.]+', sentence)
        return output


with open('./table.txt', 'r') as f:
    txt = f.readlines()
    tables = [txt[i * 10:(i + 1) * 10] for i in range(len(txt) // 10)]

    for table in tables:
        paras_str = table[0]
        result_200_str = table[3]
        result_500_str = table[8]

        paras = extract(paras_str)
        result_200 = extract(result_200_str)
        result_500 = extract(result_500_str)

        with open('./tex.txt', 'a+') as w:
            print(head, file=w)
            print(
                f'\multirow{2}*{{}} & $\\alpha_1 = {paras[0]}$ & $\\alpha_2 = {paras[1]}$ & $\\beta_1 = {paras[2]}$ & $\\beta_2 = {paras[3]}$ \\\\',
                file=w)

            print(mid, file=w)

            for num in [result_200[i * 2:(i + 1)*2] for i in range(4)]:
                print(f'\\a & {num[0]} & & & \\\\', file=w)
                print(f' & {num[1]} & & & \\\\', file=w)
            print('\hline', file=w)
            for num in result_500:
                print(f'{num} & & & & \\\\', file=w)

            print(bottom + '\n', file=w)
            print('========================', file=w)
