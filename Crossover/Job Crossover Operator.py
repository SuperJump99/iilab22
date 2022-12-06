import random
p1 = [[1,1], [2,1], [1,2], [2,2], [5,1], [5,2], [5,3], [1,3], \
    [2,3], [1,4], [3,1], [4,1], [3,2], [3,3], [2,4], [4,2]]
p2 = [[5,1], [1,1], [3,1], [4,1], [2,1], [1,2], [3,2], [2,2], \
    [3,3], [2,3], [5,2], [1,3], [2,4], [5,3], [1,4], [4,2]]

job_list = [1,2,3,4,5]

def crossover2(p1, p2, job_list):

    take_job = random.sample(job_list, random.randint(1, len(job_list)))
    print("selected job in p1:", take_job)

    p1_mini = []
    for i in range(len(p1)):
        if p1[i][0] in take_job:
            p1_mini.append(p1[i])
    print("selected job and operation in p1:", p1_mini)

    p2_mini = []
    for i in p2:
        if i not in p1_mini:
            p2_mini.append(i)
    print("selected job and operation in p2:", p2_mini)

    a = 0
    child = []
    for i in range(len(p1)):
        if p1[i][0] in take_job:
            child.append(p1[i])
            a += 1
        else:
            child.append(p2_mini[i-a])
    return child

print("Child:", crossover2(p1, p2, job_list))