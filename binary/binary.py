def decimal_to_binary(number, result=''):
  if number == 0:
    return result
  result = str(number % 2) + result
  return decimal_to_binary(number // 2, result)


# print(decimal_to_binary(25))
# print(decimal_to_binary(10))

def bit_and(n):
  bit_str = decimal_to_binary(n)
  result = 0
  for i in bit_str:
    result &= int(i)
  return result

# print(bit_and(5))