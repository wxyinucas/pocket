# -*- coding: utf-8 -*-
import re
from datetime import datetime

label = '2016-7-6'
patterns = []

label_time = datetime.strptime(label, '%Y-%m-%d')
patterns.append(re.compile(re.escape(label_time.strftime(u'%Y年%m月%d日'))))
patterns.append(re.compile(re.escape(label_time.strftime('%Y年%-m月%-d日'))))

for p in patterns:
    if p.search('2016年7月6日'):
        print(f'Got by {p.pattern}')
        # print 'Got by {}'.format(p.pattern)
    else:
        print(f'{p.pattern} catch nothing!!')
        # print '{} catch nothing'.format(p.pattern)
