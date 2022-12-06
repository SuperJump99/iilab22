import math

import numpy as np

weight = [1, 1, 1, 1, 1, 1, 1]
ready_time = [0, 0, 0]
processing_time = [3, 4, 5, 5, 2, 1, 4]
due_date = [8, 10, 7, 9, 8, 6, 5]

k = 1.5
schedule = []
ATC_list = list(np.zeros(len(processing_time)))
is_done = True
m = 0
while is_done:
    for index in range(len(due_date)):

        if due_date[index] != 0:

            w= weight[index]
            slack_time= due_date[index] - processing_time[index]
            p_bar = sum(processing_time) / (len(due_date) -m)  # 1은 최초 시작할때를 의미, 이 후 2,3 ... 씩 진행

            I = w / processing_time[index] * math.exp(-max(slack_time, 0) / (k * p_bar))

            ATC_list[index] = I     # 각각의 I 값 ATC_list에 저장
    print(ATC_list)
    reset = (ATC_list.index(max(ATC_list)))     # 가장 큰 I 값 인덱스 찾기
    schedule.append(reset+1)
    processing_time[reset] = 0          # 사용한 I 값 duedate, processingtime의 인덱스 0으로 만들기
    due_date[reset] = 0
    ATC_list = list(np.zeros(len(processing_time)))        # ATC_list 초기화
    m += 1
    if due_date == ATC_list:
        is_done = False

print(schedule)