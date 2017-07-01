import matplotlib.pyplot as plt

log_file = 'log.txt'

# 'Round %d timestep %d group_num %d average_group_size %f max_group_size %d'

group_num = []
average_group_size = []
max_group_size = []
agent_num = []
pig_num = []

with open(log_file)as fin:
    for line in fin:
        line = line.split()
        pig_num.append(int(line[5]))
        agent_num.append(int(line[18]))

x = range(len(agent_num))
x = x[8000:]
agent_num = agent_num[8000:]
pig_num = pig_num[8000:]

length = len(x)

agent_num_avg = []
pig_num_avg = []

for i in xrange(0, length, 10):
    agent_tot = 0
    pig_tot = 0
    for j in xrange(i, min(i + 10, len(x))):
        agent_tot += agent_num[j]
        pig_tot += pig_num[j]

    agent_tot = 1. * agent_tot / 10.
    pig_tot = 1. * pig_tot / 10.
    agent_num_avg.append(agent_tot)
    pig_num_avg.append(pig_tot)

      
    

print agent_num
print pig_num

plt.figure(figsize=(8, 6))
#plt.plot(x, agent_num, label='agent number')
#plt.plot(x, pig_num, label='pig number')
plt.plot(agent_num_avg, pig_num_avg, label='number')
#plt.plot(agent_num_avg, pig_num_avg, label='number')
plt.xlabel('agent number')
plt.ylabel('pig number')
plt.legend(['agent number', 'pig number'], loc='upper left')
plt.grid()
plt.savefig('two species.pdf')

#plt.figure(figsize=(8, 6))
#plt.plot(x, group_num, label='Number of groups')
#plt.xlabel('timestep')
#plt.ylabel('group number')
#plt.grid()
#plt.savefig('group_num.pdf')
#
#plt.figure(figsize=(8, 6))
#plt.plot(x, average_group_size, label='Average group size')
#plt.xlabel('timestep')
#plt.ylabel('average size')
#plt.grid()
#plt.savefig('avg_group_size.pdf')
#
#plt.figure(figsize=(8, 6))
#plt.plot(x, max_group_size, label='Max group size')
#plt.xlabel('timestep')
#plt.ylabel('Max size')
#plt.grid()
#plt.savefig('max_group_size.pdf')
