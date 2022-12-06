import pandas as pd
from csv import writer

file = pd.read_csv('Transposed data Single Machine Scheduling - CSV file.csv', index_col=0)
#  index_col=0 -> 첫번째 열의 인덱스를 없앨때 사용 or 파일에서 Jobs라고 지정하면 index_col='Jobs' 사용

file["order"] = [2, 3, 1]                                                                                               # DataFrame에 열추가 하는 방법
# file.to_csv("C:/python2022/study/파이썬스터디/Transposed data Single Machine Scheduling - CSV file.csv")                  # 8행 DataFrame에 열추가한 것을 csv파일로 저장
                                                                                                                        # 작업순서를 일일이 쓰지 않고 enmerate해서 하고, 작업순서를 출력하려면?
print(file)

ready_time_take = file['readytime']                                                                                     # .values => ready_time_take = file['출제시간'] 즉, 특정 열(여기서는 file의 출제시간)의 값만 추출
ready_time = ready_time_take.values.tolist()                                                                            # 추출한 값의 타입을 .tolist() 값을 리스트로 저장
print(ready_time)

processing_time_take = file['processing time']
processing_time = processing_time_take.values.tolist()
print(processing_time)

due_date_take = file['due date']
due_date = due_date_take.values.tolist()
print(due_date)

weight_take = file['weight']
weight = weight_take.values.tolist()
print(weight)

schedule_take = file['order']
schedule = schedule_take.values.tolist()
print(schedule)


def calculate_makespan(ready_time, processing_time, due_date, weight, schedule):

    makespan = 0
    completion_time_list = []

    for i in range(len(schedule)):
        job_index = schedule[i] - 1
        if i == 0 :
            completion_time = ready_time[job_index] + processing_time[job_index]
            completion_time_list.append(completion_time)
        else:
            if completion_time_list[i - 1] < ready_time[job_index]:
                completion_time = ready_time[job_index] + processing_time[job_index]
            else:
                completion_time = completion_time_list[i - 1] + processing_time[job_index]
            completion_time_list.append(completion_time)

    makespan = completion_time_list[len(schedule) - 1]
    return makespan

def calculate_flowtime(ready_time, processing_time, due_date, weight, schedule):

    flowtime = 0
    completion_time_list = []

    for i in range(len(schedule)):
        job_index = schedule[i] - 1
        if i == 0 :
            completion_time = ready_time[schedule[0] - 1] + processing_time[schedule[0] - 1]
            completion_time_list.append(completion_time)
        else:
            if completion_time_list[i - 1] < ready_time[job_index]:
                completion_time = ready_time[job_index] + processing_time[job_index]
            else:
                completion_time = completion_time_list[i - 1] + processing_time[job_index]
            completion_time_list.append(completion_time)

    for i in range(len(schedule)):
                                                                                                     # 현재 for 문에는 job_index가 선언되지 않았음, global을 통해 위에 for문에 사용된 job_index를 가져옴
        flowtime += completion_time_list[i] - ready_time[job_index]
    return flowtime

def calculate_total_tardiness(ready_time, processing_time, due_date, weight, schedule):

    total_tardiness = 0
    completion_time_list = []
    total_tardiness_list = []

    for i in range(len(schedule)):
        job_index = schedule[i] - 1

        if i == 0:
            completion_time = ready_time[job_index] + processing_time[job_index]
            completion_time_list.append(completion_time)
            tardiness = completion_time_list[i] - due_date[job_index]                                                   # completion_time_list[job_index] 의 요소를 중간값으로 사용 하기위해 job_index말고 i 사용
            if tardiness <= 0 :
                tardiness = 0
            else:
                tardiness = completion_time_list[i] - due_date[job_index]
            total_tardiness_list.append(tardiness)
        else:
            if completion_time_list[i - 1] <= ready_time[job_index]:
                completion_time = ready_time[job_index] + processing_time[job_index]
                completion_time_list.append(completion_time)
                tardiness = completion_time_list[i] - due_date[job_index]
                if tardiness <= 0:
                    tardiness = 0
                else:
                    tardiness = completion_time_list[i] - due_date[job_index]
                total_tardiness_list.append(tardiness)
            else:
                completion_time = completion_time_list[i - 1] + processing_time[job_index]
                completion_time_list.append(completion_time)
                tardiness = completion_time_list[i] - due_date[job_index]
                if tardiness <= 0:
                    tardiness = 0
                else:
                    tardiness = completion_time_list[i] - due_date[job_index]
                total_tardiness_list.append(tardiness)
        total_tardiness += total_tardiness_list[i]
    return total_tardiness

def calculate_total_weighted_tardiness(ready_time, processing_time, due_date, weight, schedule):

    total_weight_tardiness = 0
    completion_time_list = []
    total_tardiness_list = []

    for i in range(len(schedule)):
        job_index = schedule[i] - 1

        if i == 0:
            completion_time = ready_time[job_index] + processing_time[job_index]
            completion_time_list.append(completion_time)
            tardiness = completion_time_list[i] - due_date[job_index]
            if tardiness <= 0:
                tardiness = 0
            else:
                tardiness = completion_time_list[i] - due_date[job_index]
            total_tardiness_list.append(tardiness)
        else:
            if completion_time_list[i - 1] <= ready_time[job_index]:
                completion_time = ready_time[job_index] + processing_time[job_index]
                completion_time_list.append(completion_time)
                tardiness = completion_time_list[i] - due_date[job_index]
                if tardiness <= 0:
                    tardiness = 0
                else:
                    tardiness = completion_time_list[i] - due_date[job_index]
                total_tardiness_list.append(tardiness)
            else:
                completion_time = completion_time_list[i - 1] + processing_time[job_index]
                completion_time_list.append(completion_time)
                tardiness = completion_time_list[i] - due_date[job_index]
                if tardiness <= 0:
                    tardiness = 0
                else:
                    tardiness = completion_time_list[i] - due_date[job_index]
                total_tardiness_list.append(tardiness)
        total_weight_tardiness += total_tardiness_list[i] * weight[job_index]
    return total_weight_tardiness

print(calculate_makespan(ready_time, processing_time, due_date, weight, schedule))
print(calculate_flowtime(ready_time, processing_time, due_date, weight, schedule))
print(calculate_total_tardiness(ready_time, processing_time, due_date, weight, schedule))
print(calculate_total_weighted_tardiness(ready_time, processing_time, due_date, weight, schedule))

                                                                                                                        # csv파일에 쓰기
solution = []
solution.append(calculate_makespan(ready_time, processing_time, due_date, weight, schedule))
solution.append(calculate_flowtime(ready_time, processing_time, due_date, weight, schedule))
solution.append(calculate_total_tardiness(ready_time, processing_time, due_date, weight, schedule))
solution.append(calculate_total_weighted_tardiness(ready_time, processing_time, due_date, weight, schedule))
print(solution)
name = ['makesapn', 'flowtime', 'tardiness', 'total_weighted_tardiness']

for optimization in range(len(solution)):                                                                               # 새로운 행에 이름, 값 출력
    list_data = [name[optimization], solution[optimization]]

    with open('Transposed data Single Machine Scheduling - CSV file.csv', 'a', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(list_data)
        f_object.close()

# 작업순서 쓰기
order_list = ['order', schedule]
with open('Transposed data Single Machine Scheduling - CSV file.csv', 'a', newline='') as order:
    writer_object = writer(order)
    writer_object.writerow(order_list)                                                                                  # 231이 아니라 작업의 이름을 쓰려면?
    order.close()