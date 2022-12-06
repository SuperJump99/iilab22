ready_time = [0, 0, 0]
processing_time = [3, 4, 5]
due_date = [8, 10, 7]
weight = [1, 1, 1]

class Scheduling():
    def __init__(self):
        self.ready_time = np.array(ready_time)
        self.processing_time = np.array(processing_time)
        self.due_date = np.array(due_date)
        self.weight = weight
        # self.x = 0
        self.count = 0
        self.state = np.zeros(len(processing_time))
        self.flow_time = 0

    def step(self,a):
        # 0번 액션: 왼쪽, 1번 액션: 위, 2번 액션: 오른쪽, 3번 액션: 아래쪽
        if a == 0:
            self.selecet_duedate()
            reward = self.cal_tardiness()


        elif a == 1:
            self.select_processing()
            reward = self.cal_tardiness()

        self.count += +1
        done = self.is_done()
        return self.state, reward, done, self.count


# 초기 상태[count(총작업개수)-(한 스텝), 선택한 작업, 작업 가능 여부, 현재 완료 시간 ,processing time, duedate]
    def cal_complete_time(self):
        self.flow_time = self.flow_time + self.processing_time[index]

# reward 계산 부분
    def cal_tardiness(self):
        self.tardiness = self.due_date[index] - self.flow_time
        if self.tardiness >= 0:
            reward = 0
        else:
            reward = self.tardiness

        return reward

    def selecet_duedate(self):
        global index

        action = min(self.due_date)
        print(type(self.due_date))
        val = np.min(ma.masked_where(self.due_date == 0,self.due_date))
        print(val)
        index = np.argmin(ma.masked_where(self.due_date == 0, self.due_date))
        print(index)
        self.state[self.count] = index + 1
        self.due_date[index] = 0
        self.processing_time[index] = 0

    def select_processing(self):
        global index
        action = min(self.processing_time)
        index = np.argmin(ma.masked_where(self.processing_time == 0, self.processing_time))
        self.state[self.count] = index
        self.due_date[index] = 0
        self.processing_time[index] = 0

    def is_done(self):
        if self.count == len(processing_time):
            return True
        else:
            return False

    def reset(self):
        self.state = np.zeros([len(processing_time)])
        self.count = 0
        self.flow_time = 0
        return self.state