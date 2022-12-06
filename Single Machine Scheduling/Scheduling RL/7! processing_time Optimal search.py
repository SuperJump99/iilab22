import itertools
import numpy as np

# weight = [1, 1, 1]
# ready_time = [0, 0, 0]
# processing_time = [3, 4, 5, 5, 2, 1, 4]
# duedate = [8, 10, 7, 8, 3, 5, 5]
# processing_time = [3, 4, 5]
# duedate = [5, 10, 7]

processing_time = [3, 4, 5, 5, 2, 1, 4]
due_date = np.array([8, 10, 7, 9, 8, 6, 5])

tardlist = []

nPr = itertools.permutations(processing_time, len(processing_time))
for i in list(nPr):
    flowtime = 0
    total_tardiness = 0

    for j in range(len(list(i))):
        m = 0
        processingtime_val = list(i)[j]

        a = [k for k in range(len(processing_time)) if processing_time[k] == list(i)[j]]
        if due_date[a[m]] == 0:
            m += 1
        flowtime = flowtime + list(i)[j]

        tardiness = due_date[a[m]] - flowtime
        if tardiness >= 0:
            tardiness = 0

        total_tardiness += tardiness
        due_date[a[m]]=0

        # print('{}-{}={}'.format(due_date[a[m]], flowtime, tardiness),'>>', processing_time[a[m]])

    print(list(i), '>>', total_tardiness)
    tardlist.append(total_tardiness)
    due_date = np.array([8, 10, 7, 9, 8, 6, 5])

print(max(tardlist))
