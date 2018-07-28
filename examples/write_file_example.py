import os
import pandas as pd

path = os.getcwd() + '/test.txt'
a = pd.DataFrame([[u'哈哈', 2, 3], [2, 3, 4]], columns=list('abc'))

with open(path, 'w') as f:
    f.write(str(a['a'].values))
