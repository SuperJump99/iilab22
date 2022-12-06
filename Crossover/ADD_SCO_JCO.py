import random
p1 = [[1,1], [2,1], [1,2], [2,2], [5,1], [5,2], [5,3], [1,3], \
      [2,3], [1,4], [3,1], [4,1], [3,2], [3,3], [2,4], [4,2]]
p2 = [[5,1], [1,1], [3,1], [4,1], [2,1], [1,2], [3,2], [2,2], \
    [3,3], [2,3], [5,2], [1,3], [2,4], [5,3], [1,4], [4,2]]

def crossover1(p1, p2):

    print("****Calculate start****")
    child_front = []
    child_back = []

    i = random.choice(p1)  # i는 기준점

    print("> Crossover Point : ", i)
    position = p1.index(i)                                 # 기준점의 위치를 찾아 처음부터 기준점까지 값을 child에 넣는다.
    print("> {} Index Number : ".format(i), position)
                                                        # p1에서 i의 값에 해당하는 인덱스 찾기
    slicing_p1 = p1[0:position]                            # p1에서 i 전까지 자르기
    child_front.extend(slicing_p1)                      # child에 추출된 p1요소 넣기
    print("> child_front : ", child_front)

    for i in p2:                                        # p2에서 child_front와 겹치는 부분을 제거 후 순서대로 정렬
        if i not in child_front:
            child_back.append(i)
    print("> child_back : ", child_back)
    child = child_front + child_back

    return child

def crossover2(p1, p2):
    # random select 작업선택
    print("****Calculate start****")
    select = random.sample(p1, 2)
    while True:
        if select[0][0] == select[1][0]:
            select = random.sample(p1, 2)
        else:
            break
    first_num = select[0][0]
    second_num = select[1][0]
    print("> Take job{} and job{}".format(first_num, second_num))

    find_index_1 = []
    for i in range(len(p1)):
        if p1[i][0] == first_num:
            find_index_1.append(i)
    print("> Index in job{}:".format(first_num), find_index_1)

    find_index_2 = []
    for j in range(len(p1)):
        if p1[j][0] == second_num:
            find_index_2.append(j)
    print("> Index in job{}:".format(second_num), find_index_2)
    # 2개의 작업 인덱스를을 더해 내림차순으로 하여 해당 작업의 실제 data 값을 가져옴
    index_sort = find_index_1 + find_index_2
    find_index = find_index_1 + find_index_2
    index_sort.sort()
    find_index.sort()
    print("> Need index :", index_sort)

    jobs = []
    for i in range(len(find_index)):
        find_index[i] = p1[find_index[i]]
        jobs.append(find_index[i])
    print("> Need data from p1 : ", jobs)

    child = []
    for i in p2:  # p2에서 p1의 실제 data와 겹치지 않는 부분을 제거 후 child에 넣음
        if i not in find_index:
            child.append(i)
    print("> Need data from p2 :", child)
    # p1의 해당 위치 값을 child에 넣음
    for i in range(len(jobs)):
        if index_sort[i] < len(child):
            child.insert(index_sort[i], jobs[i])
        else:
            child.append(jobs[i])

    return child

print("----Single Point Crossover Operator----\n>>>", crossover1(p1, p2))
print()
print("----Job Crossover Operator----\n>>> ", crossover2(p1,p2))