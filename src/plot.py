import matplotlib.pyplot as plt

log_file = 'log.txt'

# 'Round %d timestep %d group_num %d average_group_size %f max_group_size %d'

pig_num = []
group_num = []
mean_size = []
variance_size = []
max_size = []

with open(log_file)as fin:
    for line in fin:
        line = line.split()
        pig_num.append(line[5])
        group_num.append(line[7])
        mean_size.append(line[9])
        variance_size.append(line[11])
        max_size.append(line[13])

x = range(len(pig_num))

plt.figure(figsize=(8, 6))
plt.plot(x, pig_num, label='pig number')
plt.xlabel('timestep')
plt.ylabel('pig number')
plt.grid()
plt.legend(['pig number'], loc='upper left')
plt.savefig('pig number.pdf')

plt.figure(figsize=(8, 6))
plt.plot(x, group_num, label='group number')
plt.xlabel('timestep')
plt.ylabel('group number')
plt.grid()
plt.legend(['group number'], loc='upper left')
plt.savefig('group number.pdf')

plt.figure(figsize=(8, 6))
plt.plot(x, mean_size, label='mean size')
plt.xlabel('timestep')
plt.ylabel('mean size')
plt.grid()
plt.legend(['mean size'], loc='upper left')
plt.savefig('mean size.pdf')

plt.figure(figsize=(8, 6))
plt.plot(x, variance_size, label='group number')
plt.xlabel('timestep')
plt.ylabel('variance size')
plt.grid()
plt.legend(['variance size'], loc='upper left')
plt.savefig('variance size.pdf')

plt.figure(figsize=(8, 6))
plt.plot(x, max_size, label='max size')
plt.xlabel('timestep')
plt.ylabel('max size')
plt.grid()
plt.legend(['max size'], loc='upper left')
plt.savefig('max size.pdf')
