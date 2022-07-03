List_1 = ['LOT_1-A_1', 'LOT_2-B_1', 'LOT_1-A_2', 'LOT_1-A_3', 'LOT_3-B_1', 'LOT_3-B_2', 'LOT_2-B_2']

Schedule = {'M_1': [],
            'M_2': [],
            'M_3': [],
            'M_4': []}

PROCESSING_TIME = {'A_1': 4, 'A_2': 4, 'B_1': 5, 'A_3': 1, 'B_2': 2, 'C_1': 3, 'C_2': 3}
INITIAL_SETUP = {'M_1': 'A_1', 'M_2': 'A_1', 'M_3': 'A_1', 'M_4': 'A_1'}

SETUP_TIME = {'homogeneous_setup': 2, 'heterogeneous_setup': 4}

# 기계 시간 초기화
machine_time = []
machine_name = []
for machine in Schedule.keys():
    machine_time.append(0)
    machine_name.append(machine)
# print('#Machine Time Reset', machine_time)

pct_list =[]
for i in List_1:
    pct = i.split("-")[1]
    pct_list.append(pct)

# List_1을  time으로 변환 (str -> int)
PCT = []
for i in range(len(pct_list)):
    for key, value in PROCESSING_TIME.items():
        if pct_list[i] == key:
            PCT.append(value)
            break
# print(PCT)

next_machine_index = 0
for work in List_1:

    if next_machine_index == 0:
        Schedule[machine_name[0]].append(work)
        if INITIAL_SETUP[machine_name[0]] == work.split("-")[1]:             # 셋업 시간 계산
            pass
        elif INITIAL_SETUP[machine_name[0]][0] == work.split("-")[1][0]:
            machine_time[next_machine_index] += SETUP_TIME['homogeneous_setup']
        else:
            machine_time[next_machine_index] += SETUP_TIME['heterogeneous_setup']
        INITIAL_SETUP[machine_name[0]] = pct_list[0]
        machine_time[next_machine_index] += PCT[0]

    elif next_machine_index == 1:
        Schedule[machine_name[1]].append(work)
        if INITIAL_SETUP[machine_name[1]] == work.split("-")[1]:             # 셋업 시간 계산
            pass
        elif INITIAL_SETUP[machine_name[1]][0] == work.split("-")[1][0]:
            machine_time[next_machine_index] += SETUP_TIME['homogeneous_setup']
        else:
            machine_time[next_machine_index] += SETUP_TIME['heterogeneous_setup']
        INITIAL_SETUP[machine_name[1]] = pct_list[0]
        machine_time[next_machine_index] += PCT[0]

    elif next_machine_index == 2:
        Schedule[machine_name[2]].append(work)
        if INITIAL_SETUP[machine_name[2]] == work.split("-")[1]:  # 셋업 시간 계산
            pass
        elif INITIAL_SETUP[machine_name[2]][0] == work.split("-")[1][0]:
            machine_time[next_machine_index] += SETUP_TIME['homogeneous_setup']
        else:
            machine_time[next_machine_index] += SETUP_TIME['heterogeneous_setup']
        INITIAL_SETUP[machine_name[2]] = pct_list[0]
        machine_time[next_machine_index] += PCT[0]

    elif next_machine_index == 3:
        Schedule[machine_name[3]].append(work)
        if INITIAL_SETUP[machine_name[3]] == work.split("-")[1]:  # 셋업 시간 계산
            pass
        elif INITIAL_SETUP[machine_name[3]][0] == work.split("-")[1][0]:
            machine_time[next_machine_index] += SETUP_TIME['homogeneous_setup']
        else:
            machine_time[next_machine_index] += SETUP_TIME['heterogeneous_setup']
        INITIAL_SETUP[machine_name[3]] = pct_list[0]
        machine_time[next_machine_index] += PCT[0]

    del pct_list[0], PCT[0]
    next_machine_index = machine_time.index(min(machine_time))      # 다음 할당할 머신 선택(시간이 최소 이며, 가장 빠른 넘버)
    # print('Result\n',Schedule, machine_time)
    # print('Remain\n', PCT)
    # print('INITIAL_SETUP=', INITIAL_SETUP)
    # print('Next selected machin\n', machine_time.index(min(machine_time)))


print(Schedule)