def pos_lf_start_time(c, visual=False):
    pattern = re.compile(ur"于" + u'TARGET_SPAN' + u'[^。，；]*(进行|办理)[^，。；]*质押(登记|手续|交易)', re.UNICODE)
    pattern8 = re.compile(ur"于" + u'TARGET_SPAN' + u'将其持有', re.UNICODE)
    pattern4 = re.compile(ur"手续[^，。；]*于" + u'TARGET_SPAN' + u'[^，。；]*(办理|完成)', re.UNICODE)
    pattern1 = re.compile(ur"自" + u'TARGET_SPAN' + u'(质押登记日)?起', re.UNICODE)
    pattern2 = re.compile(ur"(?<!购)(?<!回)(开始日期|登记日|期限|时间|起始日|交易日)[自为是：]?" + u'TARGET_SPAN', re.UNICODE)
    pattern6 = re.compile(ur"质押(登记日|期限|时间|起始日|交易日)分别[^至。；，]*" + u'TARGET_SPAN', re.UNICODE)
    pattern3 = re.compile(ur'TARGET_SPAN' + ur'[^接收。；]*将[^。，；]*质押', re.UNICODE)
    pattern5 = re.compile(ur'TARGET_SPAN' + ur'[^接收。；]*提交[^。，；]*申请', re.UNICODE)
    pattern7 = re.compile(u'TARGET_SPAN' + u'至', re.UNICODE)

    patterns = [pattern, pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8]

    sentences = sorted(c[0].sentence.document.sentences, key=lambda x: x.position)
    text = c[0].sentence.text[:c[0].char_start] + u'TARGET_SPAN' + c[0].sentence.text[c[0].char_end + 1:]
    for p in patterns:
        if re.search(p, text) is not None:
            if visual:
                print
                re.search(p, text).group()
                print
                p.pattern
            return 1
    return 0


def pos_dis_start_time(c__, visual=False):
    target = c__[0].get_span()

    #     try:
    #         target = str(target)
    #     except:
    #         return 0
    target_patterns = []

    try:
        doc_name = session.query(Distant_pledge_time).filter(Distant_pledge_time.id == c__.id).all()[0][
            0].sentence.document.name
        df_target_source = distant[distant['OldID'] == int(doc_name)]
    except:
        return 0

    labels = df_target_source[ur'质押起始日']

    for i in labels:
        target_patterns.extend(pattern_expand(i))

    for p in target_patterns:
        if p.search(target):
            if visual:
                print
                '###############'
                print
                re.search(p, target).group()
                print
                target
                print
                p
            return 1

    # 一小块debug代码，用于显示dis lf未标记的数据。
    if visual:
        print('-------')
        print("Doc name:" + doc_name)
        print('Input:' + target)
        for p in target_patterns:
            print('Distant:' + p.pattern)
            print
            p.search(target)

    return 0


#################


def neg_lf_start_time(c, visual=False):
    pattern = re.compile(ur"自" + u'TARGET_SPAN', re.UNICODE)
    pattern1 = re.compile(ur"于" + u'TARGET_SPAN' + u'接到', re.UNICODE)
    pattern2 = re.compile(ur"截(至|止)" + u'TARGET_SPAN', re.UNICODE)
    pattern3 = re.compile(u'TARGET_SPAN' + u'，?[^，。；]*(接|收)到[^，；。]*(（.*）)?[^，；。]*(通知|函)', re.UNICODE)
    pattern4 = re.compile(u'TARGET_SPAN' + u'数据', re.UNICODE)
    pattern5 = re.compile(u'TARGET_SPAN' + u'解除', re.UNICODE)
    pattern6 = re.compile(ur"(回购|购回)(开始日期|登记日|期限|时间|起始日|交易日)[自为是：]?" + u'TARGET_SPAN', re.UNICODE)

    patterns = [pattern, pattern1, pattern2, pattern3, pattern4, pattern5, pattern6]
    sentences = sorted(c[0].sentence.document.sentences, key=lambda x: x.position)
    #     text = '\n'.join([s.text for s in sentences])
    text = c[0].sentence.text[:c[0].char_start] + u'TARGET_SPAN' + c[0].sentence.text[c[0].char_end + 1:]
    for p in patterns:
        if re.search(p, text) is not None:
            if visual:
                print
                re.search(p, text).group()
                print
                p.pattern
            return -1
    return 0


#################

hand_defined_lfs = [pos_lf_start_time, pos_dis_start_time, neg_lf_start_time]

import numpy as np


def neg_lf_not_covered(c):
    for lf in hand_defined_lfs:
        res = lf(c)
        if res != 0:
            return 0
    return -1


lfs = [pos_lf_start_time, pos_dis_start_time, neg_lf_start_time, neg_lf_not_covered]
# lfs = [pos_lf_start_time, neg_lf_start_time, neg_lf_not_covered]
