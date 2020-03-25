import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

input_file_name = sys.argv[1]


BS = []
StoB = []
loss = []
acc = []
with open(input_file_name) as fobj:
    for line in fobj:
        row = line.split()
        if float(row[1]) == -1:
            continue
        BS.append(float(row[0]))
        StoB.append(float(row[1]))
        # loss.append(float(row[2]))
        # acc.append(float(row[3]))

if len(sys.argv) >= 4:
    start = float(sys.argv[2])
    end = float(sys.argv[3])
else:
    start = min(BS) - 1
    end = max(BS) + 1

BS = np.array(BS)
StoB = np.array(StoB)
loss = np.array(loss)
acc = np.array(acc)

print "Making graph from {} to {}".format(start,end)

plt.figure()
plt.plot(BS,StoB,'k^',color='blue')
plt.xlim(start,end)
plt.ylabel(r"$S/\sqrt{B}$")
plt.xlabel("batch size")
plt.grid(True)
plt.show()
plt.savefig("test.png")
