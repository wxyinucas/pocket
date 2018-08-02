import os

with open('./tmp.txt', 'a+') as f:
    f.write('the first sentence')

with open('./tmp.txt', 'a+') as f:
    # f.seek(os.SEEK_END)
    f.write('\nthe second sentence')

