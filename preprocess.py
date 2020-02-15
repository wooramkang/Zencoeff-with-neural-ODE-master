import os
import sys

file = open('Zernike_labels.txt', 'r')

for i in range(1,30001,1):
    out = open('Zernik_label/' + str(i) + '.label', 'w')
    out.write(file.readline())
    out.close()
