#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 18:37:49 2018

@author: yq

郑强转换文件代码
"""

import re
import pandas as pd
import pickle
import pymysql
from pymysql import *
from mock.mock import inplace
import os

CHOSEN_LIST = ['Time', 'Code', 'Content']

#从数据库中提取数据，并下载为df格式文件
def load_data():
    config = {
          'host':'192.168.1.150',
          'port':3306,#MySQL默认端口
          'user':'root',#mysql默认用户名
          'password':'sdai',
          'db':'Announcement',#数据库
          'charset':'utf8mb4',
          'cursorclass':pymysql.cursors.DictCursor,
          }
    con= pymysql.connect(**config)
    # 执行sql语句
    try:
        with con.cursor() as cursor:
            sql= 'SELECT * FROM Announcement.announcement where title  regexp "质押" and title not regexp "(解押|解除)"  and Time not regexp "2016-05.*" and Content is not Null;'
            cursor.execute(sql)
            result=cursor.fetchall() 
    finally:
        con.close()
    results = pd.DataFrame(result)  
    pledge = results.ix[:,CHOSEN_LIST]
    file_name = "pledge_18710.df"
    fw = open(file_name,'wb')  
    pickle.dump(pledge, fw, -1)  
    fw.close()

#数据转换为txt格式文件，文件命名规则：time_code.txt 
def transfer(PATH):
    """
    PATH:数据存储路径，本函数处理的是csv文件
    """
    #读取需要处理的数据
    fr = open('pledge_18710.df','rb')  
    df_pledge_data = pickle.load(fr)  
    fr.close()
    #生成存储文件夹
    df_pledge_data_sort = df_pledge_data.sort_values(by = ["Time","Code"],ascending = True)
    if not os.path.exists("./pledge_data"):
        os.mkdir("./pledge_data")
        
    temp_path = []
    for _, row in df_pledge_data_sort.dropna().astype(str).iterrows():
        path = './pledge_data/{}_{}.txt'.format(row['Time'],row["Code"])
        if path in temp_path:
            path = './pledge_data/{}_{}_{}.txt'.format(row['Time'],row["Code"],len(temp_path))
            with open(path, 'w') as f:
                content = re.sub(r'<table>[\s\S]*?</table>', '', str(row['Content']))
                f.write(content)
            temp_path.append(path)
        else:
            temp_path = []
            with open(path, 'w') as f:
                content = re.sub(r'<table>[\s\S]*?</table>', '', str(row['Content']))
                f.write(content)
            temp_path.append(path)


if __name__ == '__main__':
    PATH = r'query_result.df'
    transfer(PATH)
