p1 = [[1,1,2], [2,1,4], [1,2,4], [2,2,3], [5,1,2], [5,2,2], [5,3,2], [1,3,4], \
      [2,3,2], [1,4,4], [3,1,1], [4,1,1], [3,2,1], [3,3,1], [2,4,1], [4,2,4]]
p2 = [[5,1,4], [1,1,2], [3,1,2], [4,1,3], [2,1,1], [1,2,1], [3,2,1], [2,2,4], \
      [3,3,3], [2,3,1], [5,2,4], [1,3,4], [2,4,4], [5,3,1], [1,4,3], [4,2,1]]

def crossover3(p1, p2): # 입력값이 다름

      child = []
      cal_p2 = []

      for i in range(len(p1)):
            del p1[i][2]

      for i in range(len(p2)):
            cal_p2.append(p2[i][:2])

      for i in range(len(p1)):
            for j in range(len(p2)):
                  if p1[i] == cal_p2[j]:
                        p1[i].append(p2[j][2])              # del 주의할점 : 파괴적 함수!!
      child += p1
      return child

print("----Assignment Crossover Operator----\n>", crossover3(p1, p2))