class SCHEDULE:
    def __init__(self, Schedule, List_1, PROCESSING_TIME, INITIAL_SETUP, SETUP_TIME):
        self.Schedule = Schedule
        self.List_1 = List_1
        self.PROCESSING_TIME = PROCESSING_TIME
        self.INITIAL_SETUP = INITIAL_SETUP
        self.SETUP_TIME = SETUP_TIME

    def rule_1(self):
        # 기계 시간 초기화
        machine_time = []
        machine_name = []
        for machine in self.Schedule.keys():
            machine_time.append(0)
            machine_name.append(machine)
        # print('#Machine Time Reset', machine_time)
        print(machine_name)
        pct_list = []
        for i in self.List_1:
            pct = i.split("-")[1]
            pct_list.append(pct)

        # List_1을  time으로 변환 (str -> int)
        PCT = []
        for i in range(len(pct_list)):
            for key, value in self.PROCESSING_TIME.items():
                if pct_list[i] == key:
                    PCT.append(value)
                    break
        # print(PCT)

        next_machine_index = 0
        for work in self.List_1:

            if next_machine_index == 0:
                self.Schedule[machine_name[0]].append(work)
                if self.INITIAL_SETUP[machine_name[0]] == work.split("-")[1]:  # 셋업 시간 계산
                    pass
                elif self.INITIAL_SETUP[machine_name[0]][0] == work.split("-")[1][0]:
                    machine_time[next_machine_index] += self.SETUP_TIME['homogeneous_setup']
                else:
                    machine_time[next_machine_index] += self.SETUP_TIME['heterogeneous_setup']
                self.INITIAL_SETUP[machine_name[0]] = pct_list[0]
                machine_time[next_machine_index] += PCT[0]

            elif next_machine_index == 1:
                self.Schedule[machine_name[1]].append(work)
                if self.INITIAL_SETUP[machine_name[1]] == work.split("-")[1]:  # 셋업 시간 계산
                    pass
                elif self.INITIAL_SETUP[machine_name[1]][0] == work.split("-")[1][0]:
                    machine_time[next_machine_index] += self.SETUP_TIME['homogeneous_setup']
                else:
                    machine_time[next_machine_index] += self.SETUP_TIME['heterogeneous_setup']
                self.INITIAL_SETUP[machine_name[1]] = pct_list[0]
                machine_time[next_machine_index] += PCT[0]

            elif next_machine_index == 2:
                self.Schedule[machine_name[2]].append(work)
                if self.INITIAL_SETUP[machine_name[2]] == work.split("-")[1]:  # 셋업 시간 계산
                    pass
                elif self.INITIAL_SETUP[machine_name[2]][0] == work.split("-")[1][0]:
                    machine_time[next_machine_index] += self.SETUP_TIME['homogeneous_setup']
                else:
                    machine_time[next_machine_index] += self.SETUP_TIME['heterogeneous_setup']
                self.INITIAL_SETUP[machine_name[2]] = pct_list[0]
                machine_time[next_machine_index] += PCT[0]

            elif next_machine_index == 3:
                self.Schedule[machine_name[3]].append(work)
                if self.INITIAL_SETUP[machine_name[3]] == work.split("-")[1]:  # 셋업 시간 계산
                    pass
                elif self.INITIAL_SETUP[machine_name[3]][0] == work.split("-")[1][0]:
                    machine_time[next_machine_index] += self.SETUP_TIME['homogeneous_setup']
                else:
                    machine_time[next_machine_index] += self.SETUP_TIME['heterogeneous_setup']
                self.INITIAL_SETUP[machine_name[3]] = pct_list[0]
                machine_time[next_machine_index] += PCT[0]

            del pct_list[0], PCT[0]
            next_machine_index = machine_time.index(min(machine_time))  # 다음 할당할 머신 선택(시간이 최소 이며, 가장 빠른 넘버)
            # print('Result\n',Schedule, machine_time)
            # print('Remain\n', PCT)
            # print('INITIAL_SETUP=', INITIAL_SETUP)
            # print('Next selected machin\n', machine_time.index(min(machine_time)))

        return self.Schedule