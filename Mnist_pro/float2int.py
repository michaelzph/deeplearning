#! /bin/env python
#! -*- coding:utf-8 -*-

import numpy as np

line_num = 0
outf = open("submission.csv", "w")
for line in open("submission_cnn.csv", "r"):
    fields = line.strip().split(",")
    if line_num == 0:
        outf.write(",".join(fields) + "\n")
    else:
        idx, cls = line.strip().split(",")
        new_line = [str(int(float(idx))), str(int(float(cls)))]
        outf.write(",".join(new_line) + "\n")
    line_num += 1
outf.close()
print("{} lines.".format(line_num-1))




