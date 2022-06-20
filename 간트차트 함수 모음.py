
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
