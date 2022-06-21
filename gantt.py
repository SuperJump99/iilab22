import pandas as pd
from plotly.figure_factory import create_gantt

class Gantt:
    def __init__(self, gantt_data, Job):
        self.gantt_data = gantt_data
        self.Job = Job

    def Color(self):
        set_Job_type = set(self.Job)
        Job = list(set_Job_type)
        print(Job)

        if len(Job) == 1:
            colors = {'A': 'rgb(247, 195, 52)',
                      'S': 'rgb(100,100,100)'}
        elif len(Job) == 2:
            colors = {'A': 'rgb(247, 195, 52)',
                      'B': 'rgb(212, 44, 46)',
                      'S': 'rgb(100,100,100)'}
        elif len(Job) == 7:
            colors = {'A_1': "rgb(247, 195, 52)",
                      'A_2': "rgb(247, 195, 51)",
                      'A_3': "rgb(247, 195, 50)",
                      'B_1': "rgb(212, 44, 46)",
                      'B_2': "rgb(212, 44, 45)",
                      'C_1': "rgb(68, 61, 235)",
                      'C_2': "rgb(68, 61, 234)",
                      'SETUP': "rgb(100,100,100)"}
        return colors

    def Color_Lot(self, Lots):
        set_Job_type = set(self.Job)
        Job = list(set_Job_type)
        print(Job)
        colors = {}
        if len(self.Job) == 1:
            colors = {'A': 'rgb(247, 195, 52)',
                      'S': 'rgb(100,100,100)'}
        elif len(Job) == 2:
            colors = {'A': 'rgb(247, 195, 52)',
                      'B': 'rgb(212, 44, 46)',
                      'S': 'rgb(100,100,100)'}
        # elif len(Job) == 3:
        #     colors = {'A': 'rgb(247, 195, 52)',
        #               'B': 'rgb(212, 44, 46)',
        #               'C': 'rgb(68, 61, 235)',
        #               'S': 'rgb(100,100,100)'}
        elif len(Job) == 3:
            default_colors = {'1-A_1': 'rgb(247, 195, 52)','1-A_2': 'rgb(247, 195, 53)','1-A_3': 'rgb(247, 195, 54)',
                      '2-B_1': 'rgb(212, 44, 46)', '2-B_2': 'rgb(212, 44, 47)',
                      '3-B_1': 'rgb(212, 44, 48)', '3-B_2': 'rgb(212, 44, 49)',
                      '4-C_1': 'rgb(68, 61, 235)','4-C_2': 'rgb(68, 61, 236)',
                      'S': 'rgb(100,100,100)'}
            for lot in Lots:
                for color in default_colors.keys():
                    if color in lot:
                        colors[lot] = default_colors[color]
            colors['SETUP'] = 'rgb(100,100,100)'
            # colors = {'LOT_1-A_1': 'rgb(247, 195, 52)',
            #           'B': 'rgb(212, 44, 46)',
            #           'C': 'rgb(68, 61, 235)',
            #           'S': 'rgb(100,100,100)'}
        return colors

    def Save_gantt_data_to_xlsx(self):
        # 간트 데이터 엑셀로 저장
        gantt_data = pd.DataFrame(self.gantt_data)
        gantt_data.columns = ["Task", "Job_Type", "Start", "Finish"]  # 컬럼 이름 만들기
        gantt_data = gantt_data.sort_values('Start') # 시작시간으로 내림차순 정렬
        writer = pd.ExcelWriter('gantt_data.xlsx')
        gantt_data.to_excel(writer, sheet_name='Time', index=False)  # 시간순 엑셀시트 저장
        gantt_data = gantt_data.sort_values('Task')
        gantt_data.to_excel(writer, sheet_name='Machine', index=False)  # 기계순 엑셀시트 저장
        gantt_data = gantt_data.sort_values('Job_Type')
        gantt_data.to_excel(writer, sheet_name='Job_Type', index=False)  # 작업별 엑셀시트 저장
        return writer.save()

    def df_Gantt_Chart(self):
        gantt_data = pd.DataFrame(self.gantt_data)
        gantt_data.columns = ["Task", "Job_Type", "Start", "Finish"]  # 컬럼 이름 만들기
        gantt_data = gantt_data.sort_values('Task')  # 시작시간으로 내림차순 정렬
        # gantt_data = gantt_data.set_index(["Task"]) # Task(Machine 기준으로 인덱스 설정

        fig = create_gantt(gantt_data, index_col='Job_Type', bar_width=0.15, show_colorbar=True, colors=self.Color(),
                           showgrid_x=True, showgrid_y=False, title="Time Line",
                           show_hover_fill=True, group_tasks=True, )
        fig.layout.xaxis.type = "linear"
        return fig.show()

    def xlsx_Gantt_Chart(self):
        df = pd.read_excel('gantt_data.xlsx', sheet_name="Machine")  # xlsx 파일 불러오기
        fig = create_gantt(df, index_col='Job_Type', bar_width=0.15, show_colorbar=True, colors=self.Color(),
                           showgrid_x=True, showgrid_y=False, title="Time Line",
                           show_hover_fill=True, group_tasks=True,)

        fig.layout.xaxis.type = "linear"
        return fig.show()