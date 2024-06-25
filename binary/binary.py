def dec_to_bin(n):
  bin_str = ''
  if n == 1:
    return bin_str + '1'
  bin_str += str(n % 2)
  
  return bin_str + dec_to_bin(n // 2)

print(dec_to_bin(25))