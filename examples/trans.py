import re
import os

file_path = '/Users/xyzh/PycharmProjects/WXY_Project/examples/tmp.txt'

with open(file_path, 'r+') as f:
    txt = f.read()
    txt = re.sub('\n', '\n \n', txt)
    f.seek(os.SEEK_SET)
    f.write(txt)

# with open(file_path, 'w+') as f:
#     txt = f.write(txt)

