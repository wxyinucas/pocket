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
\caption{Simulation result replicate $= 1000$}\label{tab:simulation}
\\begin{tabular}{l c c c c c c c c}

\\toprule
'''

mid = '''\cline{2-5} \cline{6-9}
& Bias & ESE & ASE & CP & Bias & ESE & ASE & CP\\\\
\hline'''

bottom = '''\\bottomrule
\end{tabular}
\end{table}'''

row_names = ['$\\alpha_1$', '$\\alpha_2$', '$\\beta_1$', '$\\beta_2$']
r_ = '}'
l_ = '{'


def extract(sentence):
    settings = re.findall('(?<=\[).*?(?=\])', sentence)
    if len(settings):
        par = [list(map(float, setting.split())) for setting in settings]
        return par
    else:
        output = re.findall('[\d\.]+', sentence)
        output = [output[i:(i + 1) * 2] for i in range(4)]
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
                f'\multirow{{2}}*{{}} & \\multicolumn{{2}}{{c}}{{$\\alpha_1 = {paras[0]}$}} & '
                f'\\multicolumn{{2}}{{c}}{{$\\alpha_2 = {paras[1]}$}} & '
                f'\\multicolumn{{2}}{{c}}{{$\\beta_1 = {paras[2]}$}} & '
                f'\\multicolumn{{2}}{{c}}{{$\\beta_2 = {paras[3]}$}} \\\\',
                file=w)

            print(mid, file=w)

            for num_200, num_500, row_name in zip(result_200, result_500, row_names):
                print(f'\multirow{{2}}*{l_}{row_name}{r_} & {num_200[0]} & & & & {num_500[0]} & & & \\\\', file=w)
                print(f' & {num_200[1]} & & & & {num_500[1]} & & &\\\\', file=w)

            print(bottom + '\n', file=w)
            print('========================', file=w)
