import matplotlib.pyplot as plt

log_file = 'log.txt'

# 'Round %d timestep %d group_num %d average_group_size %f max_group_size %d'

agent_num = []
pig_num = []
rabbit_num = []
join_ratio = []
leave_ratio = []
group_proportion = []

with open(log_file)as fin:
    for line in fin:
        line = line.split()
        pig_num.append(int(line[5]))
        rabbit_num.append(int(line[15]))
        agent_num.append(int(line[21]))
        group_num = int(line[7])
        mean_size = float(line[9])
        grouped_agents_num = group_num * mean_size
        group_proportion.append(1.0 * grouped_agents_num / int(line[21]))

x = range(len(agent_num))

st = 4000
ed = 5000
x = x[st:ed]
agent_num = agent_num[st:ed]
pig_num = pig_num[st:ed]
rabbit_num = rabbit_num[st:ed]
group_proportion = group_proportion[st:ed]

plt.figure(figsize=(8, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

# ax1.plot(x, agent_num, label='agent number')
ax1.plot(x, pig_num, color='r',label='pig number')
ax1.plot(x, rabbit_num, color='b', label='rabbit number')
ax1.set_xlabel('time step')
ax1.set_ylabel('number')
ax1.legend(['pig number', 'rabbit number'], loc='upper left')

ax2.plot(x, group_proportion, color='y', label='group proportion')
ax2.set_ylabel('group proportion')
plt.grid()

plt.savefig('three species and group proportion from %d to %d.pdf' % (st, ed))
# plt.savefig('three species and group proportion all.pdf')


# plt.figure(figsize=(8, 6))
# plt.plot(x, agent_num, label='agent number')
# plt.plot(x, pig_num, label='pig number')
# plt.plot(x, rabbit_num, label='rabbit number')
# plt.xlabel('time step')
# plt.ylabel('number')
# plt.legend(['agent number', 'pig number', 'rabbit number'], loc='upper left')
# plt.grid()
# plt.savefig('three species.pdf')
