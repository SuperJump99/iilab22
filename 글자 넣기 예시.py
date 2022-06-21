# imports
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.core.display import display, HTML
import plotly.figure_factory as ff
import plotly.graph_objs as go

# setup
display(HTML("<style>.container { width:50% !important; } .widget-select > select {background-color: gainsboro;}</style>"))
init_notebook_mode(connected=True)

#%qtconsole --style vim

# dates
StartA = '2009-01-01'
StartB = '2009-03-05'
StartC = '2009-02-20'

FinishA='2009-02-28'
FinishB='2009-04-15'
FinishC='2009-05-30'

LabelDateA='2009-01-25'
LabelDateB='2009-03-20'
LabelDateC='2009-04-01'

# sample data
df = [dict(Task="Task A", Start=StartA, Finish=FinishA),
      dict(Task="Task B", Start=StartB, Finish=FinishB),
      dict(Task="Task C", Start=StartC, Finish=FinishC)]

# figure
fig = ff.create_gantt(df)
print(fig)
# add annotations
annots =  [dict(x=LabelDateA,y=0,text="Task label A", showarrow=False, font=dict(color='white')),
           dict(x=LabelDateB,y=1,text="Task label B", showarrow=False, font=dict(color='White')),
           dict(x=LabelDateC,y=2,text="Task label C", showarrow=False, font=dict(color='White'))]

# plot figure
fig['layout']['annotations'] = annots

fig.show()