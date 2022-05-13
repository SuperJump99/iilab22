# import plotly.plotly as py
# import cufflinks as cf
# cf.go_offline(connected=True)
# data = pd.Series(range(10))
# data.iplot(kind='bar', title="제목", xTitle="x축", yTitle="y축")
# df.iplot(kind='barh', barmode='stack')
#############################################
# import matplotlib.pyplot as plt
#
# Machine = ['M-1', 'M-2', 'M-3', 'M-4', 'M-5']
# time = [2, 3, 4, 1, 3]
#
# plt.barh(Machine, time)
# plt.title('Gant Chart')
# plt.ylabel('Machine')
# plt.xlabel('Time')
# plt.show()
#################################################
import plotly.express as px
df = px.data.tips()
fig = px.bar(df, x="total_bill", y="day", orientation='h')
fig.show()