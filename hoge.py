# coding: UTF-8


import math


prime_0 = 3
prime_1 = 5
prime_2 = 7
idx = 300

prime_list = [prime_0, prime_1, prime_2]
num_list = []
prime_combs = []
i = 2
# prime_comb = [0, 0, 0]
# while len(num_list) + 1 < idx:
#     if 2 not in prime_list and i % 2 != 0:
#         prime_comb = [0, 0, 0]
#         flags = [i % prime == 0 for prime in prime_list]
#         if sum(flags) > 0:
#             i_tmp = i
#             for j, prime in enumerate(prime_list):
#                 while i_tmp % prime == 0:
#                     i_tmp = i_tmp // prime
#                     prime_comb[j] += 1
#                 if i_tmp == 1 or i_tmp in num_list:
#                     # prime_combs.append(prime_comb)
#                     num_list.append(i)
#                     # print(i, prime_comb)
#                     break
#     i += 1
#
#
# for i, (num, comb) in enumerate(zip(num_list, prime_combs)):
#     print(i, num, comb)

# for c in range(100):
#     for b in range(100):
#         for a in range(100):
#             num_list.append(prime_list[0] ** a * prime_list[1] ** b * prime_list[2] ** c)
#
# num_list.sort()
# print(num_list[1000])
print(100 >> 2)