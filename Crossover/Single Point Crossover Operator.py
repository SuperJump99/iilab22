import random
p1 = [[1,1], [2,1], [1,2], [2,2], [5,1], [5,2], [5,3], [1,3], \
      [2,3], [1,4], [3,1], [4,1], [3,2], [3,3], [2,4], [4,2]]
p2 = [[5,1], [1,1], [3,1], [4,1], [2,1], [1,2], [3,2], [2,2], \
    [3,3], [2,3], [5,2], [1,3], [2,4], [5,3], [1,4], [4,2]]

def crossover1(p1, p2):

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

print("----Single Point Crossover Operator----\n>", crossover1(p1, p2))