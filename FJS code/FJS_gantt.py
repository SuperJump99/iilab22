from copy import deepcopy as cp
import pandas as pd
from plotly.figure_factory import create_gantt
from gantt import *

PRODUCTION_REQUIREMENT = {'A' : 1, 'B' : 2, 'C' : 1} # A-3, B-2, C-2
SCHEDULE = {'M_1': ['LOT_1-A_1', 'LOT_1-A_2', 'LOT_2-B_1'],
         'M_2': ['LOT_1-A_3'],
         'M_3': ['LOT_3-B_1', 'LOT_3-B_2', 'LOT_2-B_2'],
         'M_4': ['LOT_4-C_1', 'LOT_4-C_2']}


PROCESSING_TIME = { 'M_1-A_1': 4, 'M_1-A_2': 4, 'M_1-B_1': 5, 'M_2-A_3': 1, 'M_3-B_1': 4, 'M_3-B_2': 2, 'M_4-C_1': 3, 'M_4-C_2': 3}
# SETUP_TIME # homogeneous setup ex) 1_1 -> 1_3 : 2 / heterogeneous ex) 1_1 -> 2_2 : 4
INITIAL_SETUP = {'M_1' : 'A_1', 'M_2' : 'A_1', 'M_3' : 'A_1', 'M_4' : 'A_1'}
JOBTYPE_JOB = {'A': 0, 'B': 1, 'C': 2}

SETUP_TIME = {'homogeneous_setup': 2, 'heterogeneous_setup': 4}

num_of_lot = 4
num_of_job_type = 3
num_of_machine = 4

job_completion_time_list = []
machine_completion_time_list = []
job_current_finished_operation_index_list = []
completion_list = []
latest_completion_dict = {}

for i in range(num_of_job_type):
    job_completion_time_list.append(0)
for i in range(num_of_machine):
    machine_completion_time_list.append(0)

remain_operation_dict = cp(PROCESSING_TIME)
remain_SCHEDULE_dict = cp(SCHEDULE)

#job별로 현재까지 수행한 operation
for i in range(num_of_lot):
    job_current_finished_operation_index_list.append(0)


remain_operation_lst = SCHEDULE.values() # machine 구분 있음 #초기 리스트

# 중간에 있는 operation의 시작 시간 = max(같은 job의 이전 operation의 완료시간, 같은 기계의 직전 operation의 완료시간)
gantt_data = []
Job = []
while True:

    remain_num = 0  # machine별 가장 처음 operation이 1로 끝나는게 있는지 확인하는 변수
    # 딕셔너리에 남은 OPERATION이 없을 때 코드 종료
    for values in remain_SCHEDULE_dict.values():
        if not values:
            remain_num += 1
    if remain_num == num_of_machine:
        break


    for key, value in remain_SCHEDULE_dict.items():

        if value:
            operation = value[0]
            lot_num = int(operation.split('-')[0].split('_')[1])
            if int(operation.split('_')[2]) - 1 == job_current_finished_operation_index_list[lot_num-1]:
                completion_list.append(operation)
                processing_time = PROCESSING_TIME[key + '-' + operation.split('-')[1]]
                max_value = max(job_completion_time_list[JOBTYPE_JOB[operation.split('-')[1][0]]], machine_completion_time_list[int(key.split('_')[1]) - 1])
                job_completion_time_list[JOBTYPE_JOB[operation.split('-')[1][0]]] = max_value + PROCESSING_TIME[key + '-' + operation.split('-')[1]]  # 끝나는 시간 = 시작시간 + 생산시간
                machine_completion_time_list[int(key.split('_')[1]) - 1] = max_value + PROCESSING_TIME[key + '-' + operation.split('-')[1]]
                job_current_finished_operation_index_list[lot_num - 1] += 1

                if key not in latest_completion_dict.keys():
                    setup = INITIAL_SETUP[key]
                else:
                    setup = latest_completion_dict[key]

                operation_cut = operation.split('-')[1]
                latest_completion_dict[key] = operation_cut
                setup_time = 0
                job = operation_cut.split('_')[0]
                if setup.split('_')[0] == operation_cut.split('_')[0]:

                    if setup.split('_')[1] == operation_cut.split('_')[1]:
                        pass
                    else:
                        setup_time = SETUP_TIME['homogeneous_setup']
                        job_completion_time_list[JOBTYPE_JOB[operation_cut.split('_')[0]]] += SETUP_TIME['homogeneous_setup']
                        machine_completion_time_list[int(key.split('_')[1]) - 1] += SETUP_TIME['homogeneous_setup']
                else:
                    setup_time = SETUP_TIME['heterogeneous_setup']
                    job_completion_time_list[JOBTYPE_JOB[operation_cut.split('_')[0]]] += SETUP_TIME['heterogeneous_setup']
                    machine_completion_time_list[int(key.split('_')[1]) - 1] += SETUP_TIME['heterogeneous_setup']

                gantt_data.append([key, "SETUP",  max_value, max_value + setup_time])
                gantt_data.append([key, operation_cut, max_value + setup_time, max_value + setup_time + processing_time])
                Job.append(operation_cut)
                del (remain_SCHEDULE_dict[key][0])
                break
print(gantt_data) # 간트 데이터 받아오기




first = Gantt(gantt_data,Job)
first.xlsx_Gantt_Chart()

# first.df_Gantt_Chart(gantt_data)

# second = Gantt(gantt_data,Job)
# super = second.xlsx_Gantt_Chart()

