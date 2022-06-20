from copy import deepcopy as cp
import pandas as pd
from plotly.figure_factory import create_gantt


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

                gantt_data.append([key, "SETUP",  max_value, max_value + setup_time,"S"])
                gantt_data.append([key, operation_cut, max_value + setup_time, max_value + setup_time + processing_time, job])
                Job.append(job)
                del (remain_SCHEDULE_dict[key][0])
                break
print(gantt_data) # 간트 데이터 받아오기


def Color(Job):
    set_Job_type = set(Job)
    Job = list(set_Job_type)
    print(Job)

    if len(Job) == 1:
        colors = {'A': 'rgb(247, 195, 52)',
                  'S': 'rgb(100,100,100)'}
    elif len(Job) == 2:
        colors = {'A': 'rgb(247, 195, 52)',
                  'B': 'rgb(212, 44, 46)',
                  'S': 'rgb(100,100,100)'}
    elif len(Job) == 3:
        colors = {'A': 'rgb(247, 195, 52)',
                  'B': 'rgb(212, 44, 46)',
                  'C': 'rgb(68, 61, 235)',
                  'S': 'rgb(100,100,100)'}
    return colors

def Save_gantt_data_to_xlsx(gantt_data):
    # 간트 데이터 엑셀로 저장
    gantt_data = pd.DataFrame(gantt_data)
    gantt_data.columns = ["Task", "Job_Type", "Start", "Finish",'Color']  # 컬럼 이름 만들기
    gantt_data = gantt_data.sort_values('Start') # 시작시간으로 내림차순 정렬
    writer = pd.ExcelWriter('gantt_data.xlsx')
    gantt_data.to_excel(writer, sheet_name='Time', index=False)  # 시간순 엑셀 저장
    gantt_data = gantt_data.sort_values('Task')
    gantt_data.to_excel(writer, sheet_name='Machine', index=False)  # 기계순 엑셀 저장
    gantt_data = gantt_data.sort_values('Job_Type')
    gantt_data.to_excel(writer, sheet_name='Job_Type', index=False)  # 작업별 엑셀 저장
    return writer.save()

def df_Gantt_Chart(gantt_data):
    gantt_data = pd.DataFrame(gantt_data)
    gantt_data.columns = ["Task", "Job_Type", "Start", "Finish",'Color']  # 컬럼 이름 만들기
    gantt_data = gantt_data.sort_values('Start')  # 시작시간으로 내림차순 정렬
    # gantt_data = gantt_data.set_index(["Task"]) # Task(Machine 기준으로 인덱스 설정

    fig = create_gantt(gantt_data, index_col='Job_Type', bar_width=0.15, show_colorbar=True,colors=Color(Job),
                       showgrid_x=True, showgrid_y=False, title="Time Line",
                       show_hover_fill=True, group_tasks=True, )
    fig.layout.xaxis.type = "linear"

    return fig.show()

def xlsx_Gantt_Chart():
    df = pd.read_excel('gantt_data.xlsx',sheet_name="Machine")

    fig = create_gantt(df, index_col='Color', bar_width=0.15, show_colorbar=True,colors=Color(Job),
                       showgrid_x=True, showgrid_y=False, title="Time Line",
                       show_hover_fill=True, group_tasks=True, )
    fig.layout.xaxis.type = "linear"
    fig.show()
    return


# gantt_data = pd.DataFrame(gantt_data)
# gantt_data.columns = ["Task", "Job_Type", "Start", "Finish", 'Color']  # 컬럼 이름 만들기
# print(gantt_data)
#
# # Create a figure with Plotly colorscale
# fig = create_gantt(gantt_data, index_col='Color', bar_width=0.15, show_colorbar=True, colors=Color(Job),
#                    showgrid_x=True, showgrid_y=False, title="이채원 바보",
#                 show_hover_fill=True, group_tasks=True, )
# fig.layout.xaxis.type = "linear"
# fig.show()
#
