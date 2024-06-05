import os
import random

q_jiudianleixing = [
    '酒店 属于 什么 类型 啊', '那 这家 酒店 是 什么 类型 的 啊',
    '那 酒店 类型 是 什么', '请问 该 酒店 是 什么 类型',
    '他家 是 什么 类型 的 酒店 啊', '请问 该 酒店 属于 什么 类型 的', '那 你 知道 这个 酒店 是 什么 类型 吗'
]
q_fuwusheshi = [
    '这家 酒店 有 什么 服务', '他家 有 什么 服务 吗',
    '能 帮我 查一下 酒店 提供 什么 服务 吗', '那 它 有 什么 设施 吗',
    '这家 酒店 是否 有 设施 啊', '酒店 是否 有 设施', '酒店 设施 里 提供 什么',
]
q_zhoubian = [
    '附近 有 什么 酒店 呢', '这家 酒店 周边 都 有 什么 餐馆 呢', '附近 有 好吃 的 餐馆 么', '周边 还有 别的 餐馆 么',
    '请 问 该 酒店 周边 都 有 哪些 景点 啊', '周边 有 什么 好吃 的 餐馆 吗', '餐馆 周边 的 景点 有 哪些 呢',
    '餐馆 的 周边 有 哪些 别的 做菜 好吃的 餐馆 啊', '它 周边 还有 哪些 值得 一去 的 景点', '帮 我 看看 周边 都有 什么 酒店',
    '这个 景点 周边 有 方便 住宿 的 酒店 吗', '那 周边 有 什么 景点 吗', '旁边 有 什么 好玩 的 景点 吗',
    '这个 景点 的 周边 有 哪些 景点 啊', '周边 有 啥 景点 能 玩玩 吗', '周边 有 特色 的 餐馆 给 介绍 几个 吧',
    '这个 酒店 周边 有 什么 好玩 的 景点 吗', '能 告诉 我 一下 这个 餐馆 周边 都 有 哪些 景点 吗',
    '那 景点 周边 还有 什么 其他 值得 去 逛一逛 的 景点 吗', '它 附近 有 什么 酒店 吗', '周边 有 酒店 吗',
    '请问 这个 餐馆 的 周边 都有 哪些 酒店', '你 看 他家 的 周边 有 什么 景点 可以 溜溜 吗', '附近 有 什么 吃饭 的 地方',
    '麻烦 你 在 看一下 他 周边 有没有 吃饭 的 地儿 呢', '这 附近 还有 其他 别的 餐馆 吗', '附近 有 什么 酒店 吗',
    '这家 餐馆 的 周边 还有 什么 餐馆', '附近 还有 什么 餐馆 推荐 吗', '它 的 附近 有 什么 好的 餐馆 吗', '附近 有 什么 餐馆 呢',
    '该 景点 周边 都有 哪些 餐馆 啊', '请问 那 周边 有 什么 景点', '附近 有 什么 好玩 的 景点 么', '那 它 周边 有啥 景点 呢',
    '周边 有 什么 吃饭 的 地儿 吗', '周边 都有 什么 酒店 呢', '附近 有 什么 好玩的 景点 吗', '这个 景点 周边 都有 哪些 酒店 啊',
    '附近 有 什么 景点 呢', '酒店 周边 有 哪些 餐馆 呢', '附近 有 什么 玩 的 景点 么', '这家 餐馆 周边 有 方便 住宿 的 酒店 吗',
    '他的 周边 有 酒店 吗', '附近 有 什么 饭店 呢', '这家 酒店 的 周边 有 哪些 好玩 的 景点 啊',
    '这个 酒店 周边 有没有 可以 游玩 的 景点 呢', '旁边 还有 什么 饭店 吗',
    '那 请问 餐厅 周围 还有 别的 用餐 地点 吗', '想 去 哪 玩', '你 觉得 去 哪个 餐馆 比较 合适', '请问 去 哪个 景点 好玩 呢',
    '有啥 好吃 的 餐馆 吗', '那 周围 还有 啥 别的 景点 吗', '有 好的 餐馆 推荐 一下 呗', '麻烦 帮 我 找 一个 可以 玩的 景点 吧',
    '想想 明天 去 哪里 玩 呢', '中午 想去 哪 吃饭', '晚上 去 哪里 吃饭 呢',
    '推荐 一个 好玩的 地方 吧', '他家 周边 还有 其它 做菜 好吃 的 餐馆 吗', '那 酒店 内 是否 有 设施 呢',
    '你 再 帮 我 查一下 酒店 周边 都有 哪些 餐馆', '餐馆 周边 都有 什么 酒店', '那 周边 还有 什么 餐馆 值得 推荐 吗',
    '餐馆 周边 有 哪些 景点 可以 游玩', '我想 了解 一下 这个 餐馆 附近 有 什么 酒店', '这个 餐馆 周边 有 啥 景点 吗',
    '这个 景点 的 周边 有 哪些 做菜 好吃 的 餐馆 啊', '那 周边 有 什么 景点 可以 玩玩 吗', '酒店 周边 吃饭 的 餐馆 都有 啥',
    '有 什么 推荐的 餐馆 吗', '那 附近 有 吃饭 的 地方 吗', '请 问 周边 景点 有 哪些', '这是 什么 类型 的 酒店 呢',
    '帮忙 找 一家 餐馆', '最后 在 看 一下 这个 景点 的 周边 还有 值得 去的 景点 吗', '那 周边 有 什么 酒店 可以 住宿 的 吗',
    '景点 的 周边 还有 哪些 好玩的 景点 啊', '查一下 周围 评分 高 的 景点', '那它 周边 的 酒店 都有 哪些 呢'
]
q_cai = [
    '他家 有 啥 推荐菜 吗', '餐馆 都有 啥 好吃 的 推荐菜 啊',
    '那 请问 餐馆 都有 啥 推荐菜 呢', '餐馆 好吃 的 推荐菜 有啥 菜 啊',
    '有 哪些 推荐菜 呢', '这家 餐馆 有 什么 招牌菜 推荐 一下', '这家 餐馆 有啥 推荐菜', '他家 有 什么 特色菜 么'
]

q_cai_neg = q_zhoubian + q_fuwusheshi + q_jiudianleixing
q_fuwusheshi_neg = q_zhoubian + q_cai + q_jiudianleixing
q_zhoubian_neg = q_cai + q_fuwusheshi + q_jiudianleixing
q_jiudianleixing_neg = q_cai + q_fuwusheshi + q_zhoubian

cs = open('train_noneg.txt', encoding='utf-8').readlines()
nf = open('train.txt', 'w', encoding='utf-8')

nonq = set()
for c in cs:
    c = c.strip()
    q = c.split('Q')[-1]
    if q in q_cai:
        c1 = c + 'Q' + q_cai_neg[random.randint(0, len(q_cai_neg) - 1)] + '\n'
    elif q in q_fuwusheshi:
        c1 = c + 'Q' + q_fuwusheshi_neg[random.randint(0, len(q_fuwusheshi_neg) - 1)] + '\n'
    elif q in q_zhoubian:
        c1 = c + 'Q' + q_zhoubian_neg[random.randint(0, len(q_zhoubian_neg) - 1)] + '\n'
    elif q in q_jiudianleixing:
        c1 = c + 'Q' + q_jiudianleixing_neg[random.randint(0, len(q_jiudianleixing_neg) - 1)] + '\n'
    else:
        nonq.add(q)
    nf.write(c1)
nf.close()
# print(nonq)

# cs = open('dev_noneg.txt', encoding='utf-8').readlines()
# nf = open('dev_test.txt', 'w', encoding='utf-8')
# for c in cs:
#     c = c.strip()
#     c1 = c + 'Q' + 'a' + '\n'
#     nf.write(c1)
# nf.close()


# zhoubian_set = set()
# cai_set = set()
# leixing_set = set()
# fuwusheshi_set = set()
# else_set = set()
#
# cs = open('train_noneg.txt', encoding='utf-8').readlines()
# for c in cs:
#     c = c.strip()
#     q = c.split('Q')[-1]
#     if '周边' in q or '附近' in q or '旁边' in q:
#         zhoubian_set.add(q)
#     elif '推荐菜' in q or '招牌菜' in q:
#         cai_set.add(q)
#     elif '类型' in q:
#         leixing_set.add(q)
#     elif '服务' in q or '设施' in q:
#         fuwusheshi_set.add(q)
#     else:
#         else_set.add(q)
# print(zhoubian_set)
# print('=================================')
# print(cai_set)
# print('=================================')
# print(leixing_set)
# print('=================================')
# print(fuwusheshi_set)
# print('=================================')
# print(else_set)

# es = open('E_final.txt', encoding='utf-8').readlines()
# ts = open('T_final.txt', encoding='utf-8').readlines()
#
# # others1 = open('840_trainTE.txt', 'w', encoding='utf-8')
#
# dict1 = set()
# dictt = set()
#
# for c in cs:
#     c = c.strip()
#     wds = c[12: -8].split(' ')
#     for wd in wds:
#         # print(wd)
#         if wd == '':
#             print(c)
#         dict1.add(wd)
#
# for e in es:
#     e = e.strip()
#     wd = e.split(' ')[1]
#     # print(wd)
#     dictt.add(wd)
#
# for t in ts:
#     t = t.strip()[5:]
#     # print(t)
#     wds = t.split(' ')
#     for wd in wds:
#         dictt.add(wd)
# # print(len(dictt))
# # print(dict1)
# print(dictt - dict1)
# print(dict1 - dictt)

# cs = open('C_final.txt', encoding='utf-8').readlines()
# es = open('30040.txt', encoding='utf-8').readlines()
# others1 = open('30040_q.txt', 'w', encoding='utf-8')
# others2 = open('30040_noq.txt', 'w', encoding='utf-8')
#
#
# for c in cs:
#     flag = True
#     r_num = c.strip()[-6: -1]
#     for e in es:
#         if r_num in e:
#             others1.write(e)
#             flag = False
#     if flag:
#         for p in ['P001', 'P002', 'P003', 'P004', 'P005']:
#             for i in ['01', '02']:
#                 others2.write('/ssd5/dataset_SLR/Response_first_112/' + p + '/' + r_num + '_' + p + '_' + i + 'Q' + c[12: -8] + '\n')


# signs = open('sign_q_30040.txt', encoding='utf-8').readlines()
# others1 = open('q_vocab.txt', 'w', encoding='utf-8')
# others2 = open('a_vocab.txt', 'w', encoding='utf-8')
# others3 = open('q_vocab_single.txt', 'w', encoding='utf-8')
#
# q_vocab = set()
# a_vocab = set()
# q_vocab2 = set()
#
# for sign in signs:
#     sign_list = sign.strip().split('Q')
#     qs = sign_list[2].split(' ')
#     aws = sign_list[1].split(' ')
#     for qw in qs:
#         q_vocab.add(qw)
#     for aw in aws:
#         a_vocab.add(aw)
#     for ch in sign_list[2].strip(''):
#         q_vocab2.add(ch)
#
# for w in q_vocab:
#     others1.write(w + '\n')
# for w in a_vocab:
#     others2.write(w + '\n')
# for w in q_vocab2:
#     others3.write(w + '\n')
#
# print(a_vocab - dict1)
# print(dict1 - a_vocab)






    # c_list = c.split('Q')
    # r = c_list[0] + 'Q' + c_list[0] + 'Q' + c_list[1] + 'Q1Q1Q0Q0Q0Q0'
    # others1.write(r)
    # others1.write('\n')
#
# others1.close()
# others2.close()
# others3.close()