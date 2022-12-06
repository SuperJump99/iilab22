import itertools
import numpy as np

# weight = [1, 1, 1]
# ready_time = [0, 0, 0]
# processing_time = [3, 4, 5, 5, 2, 1, 4]
# duedate = [8, 10, 7, 8, 3, 5, 5]
# processing_time = [3, 4, 5]
# duedate = [5, 10, 7]

processing_time = [3, 4, 5, 5, 2, 1, 4,]
due_date = [8, 10, 7, 13, 8, 6, 5]

tardlist = []

nPr = itertools.permutations(due_date, len(due_date))
for i in list(nPr):
    flowtime = 0
    total_tardiness = 0

    for j in range(len(list(i))):
        m = 0
        due_date_val = list(i)[j]


        a = [k for k in range(len(due_date)) if due_date[k] == list(i)[j]]
        if processing_time[a[m]] == 0:
            m += 1

        flowtime = flowtime + processing_time[a[m]]
        tardiness = list(i)[j] - flowtime

        if tardiness >= 0:
            tardiness = 0

        total_tardiness += tardiness

        processing_time[a[m]]= 0
        # print('{}-{}={}'.format(list(i)[j], flowtime, tardiness),'>>', processing_time[a[m]])

    print(list(i), '>>', total_tardiness)
    tardlist.append(total_tardiness)

    processing_time = np.array([3, 4, 5, 5, 2, 1, 4])
print(max(tardlist))
