# -*- coding:utf-8 -*-

# python3 the function raw_input rename input
user = input('Enter login name:')
print('your login name is:', user)

num = input('Enter a number:')
print('Doubling you number: %d' % (int(num) * 2))

print(r'C:\some\name')  # note the r before the quote   输出字符串前添加 r 就不用手动转移特殊字符了