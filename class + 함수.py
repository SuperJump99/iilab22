class gantt_chart():
    def __init__(self,Task,Job_type,Start,Finish):
        self.Task = Task
        self.Job_type = Job_type
        self.Start = Start
        self.Finish = Finish

gantt_data =

def gantt_data():
    gantt_data = pd.DataFrame(gantt_data)
    gantt_data.columns = ["Task", "Job_Type", "Start", "Finish"]  # 컬럼 이름 만들기
    gantt_data = gantt_data.sort_values('Start')  # 시작시간으로 내림차순 정렬
    # gantt_data = gantt_data.set_index(["Task"]) # Task(Machine 기준으로 인덱스 설정
    print(gantt_data)

    fig = create_gantt(gantt_data, index_col='Job_Type', bar_width=0.1, show_colorbar=True,
                       showgrid_x=True, showgrid_y=False, title="이채원 바보",
                       show_hover_fill=True, group_tasks=True, )
    fig.layout.xaxis.type = "linear"

    fig.show()