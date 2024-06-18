def reverse_str(str):
  new_str = ''
  for i in range(len(str) - 1, -1, -1):
    new_str += str[i]

  return new_str

def reverse_str1(str):
  return ''.join(reversed(str))

def reverse_str2(str):
  return str[::-1]

# print(reverse_str('abcdef'))
# print(reverse_str1('abcdef'))
# print(reverse_str2('abcdef'))