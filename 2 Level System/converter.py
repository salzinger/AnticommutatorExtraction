import array
import numpy as np

def convert(s):
#The function that converts the string to float
   s = s.strip().replace(',', '.')
   return float(s)


data = array.array('d')  # an array of type double (float of 64 bits)

with open("10MHz_gamma.csv", 'r') as f:
    for l in f:
        strnumbers = l.split('\t')
        data.extend( (convert(s) for s in strnumbers if s!='') )
# A generator expression here.


print(np.array(data))

lines = np.array(data)

with open('10MHz_gamma.txt', 'w') as f:
    for line in lines:
        f.write(str(line))
        f.write('\n')