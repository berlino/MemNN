#!/usr/bin/python
from __future__ import division
import matplotlib.pyplot as plt

#TBD for doc retrieve

N = 9952
nums = [5, 10, 20, 30, 40, 50, 60, 70, 80]
cover = [6763, 7847, 8765, 9237, 9367, 9405, 9429, 9445, 9447]

ratio = [x / N for x in cover]


plt.figure(1)

plt.plot(nums, ratio)
plt.title("Answer hit(%) in top-N ranked sentences")
plt.axis([0,100, 0.6, 1])
plt.xlabel("Number of retrieved sentences")
plt.ylabel("% of correct answers coverred")
plt.show()

