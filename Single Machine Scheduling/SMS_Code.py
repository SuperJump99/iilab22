ready_time = [0, 0, 0]
processing_time = [3, 4, 5]

due_date = [5, 10, 7]
weight = [1, 1, 1]

schedule = [2, 3, 1] # 2번 과제 - 3번 과제 - 1번 과제 순서

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