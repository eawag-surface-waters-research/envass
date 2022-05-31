import numpy as np
import warnings
import plotly.graph_objs as go
import plotly.offline as py
from ipywidgets import interactive, HBox, VBox

def check_data(x, time):
    if type(x).__module__ != np.__name__:
        raise TypeError("Input must be a numpy array.")
    if len(x)!=len(time):
        if len(time) in x.shape:
            if x.shape[0]>x.shape[1]:
                warnings.warn("Numbers of rows is greater than numbers of columns, rows must be depth and columns is time !")
        else: 
            raise TypeError("Time and variable data should be of the same length")

def to_dict(kwargs):
    for key in list(kwargs):
        if type(kwargs[key])!=dict:
            if kwargs[key]==False:
                kwargs.pop(key)
            else:
                kwargs[key]={key:kwargs[key]}
    return kwargs

def check_parameters(test, kwargs, parameters):
    for arguments in kwargs[test].keys():
        if arguments not in parameters[test]:
            
            warnings.warn(f"argument {arguments} is not given to the function")

def isnt_number(n):
    try:
        if np.isnan(float(n)):
            return True
    except ValueError:
        return True
    else:
        return False

def init_flag(time, prior_flags):
    try: 
        if len(prior_flags):
            flag = np.array(np.copy(prior_flags),dtype=bool)
    except:
        flag = np.zeros(time.shape, dtype=bool)
    return flag


def interp_nan(time, y):
    vec=np.copy(y)
    nans, x = np.isnan(vec), lambda z: z.nonzero()[0]
    vec[nans]=np.interp(time[nans], time[~nans],vec[~nans])
    return vec

def plot_quality_assurance(df_sub):
    variables=df_sub.columns
    py.init_notebook_mode()
    f = go.FigureWidget([go.Scatter(y = df_sub.index, x = df_sub.index, mode = 'markers')])
    scatter = f.data[0]
    N = len(df_sub)
    scatter.marker.opacity = 0.8
    def update_axes(xaxis, yaxis):
        scatter = f.data[0]
        scatter.x = df_sub[xaxis]
        scatter.y = df_sub[yaxis]

        with f.batch_update():
            f.layout.xaxis.title = xaxis
            f.layout.yaxis.title = yaxis
            if "_qual" not in yaxis:
                f.add_trace(go.Scatter(y = df_sub[yaxis][df_sub[yaxis+"_qual"]==0], x = df_sub[xaxis][df_sub[yaxis+"_qual"]==0], mode = 'markers', marker = dict(color = 'blue'), name = f'{yaxis} Trusted (=0)'))
                f.add_trace(go.Scatter(y = df_sub[yaxis][df_sub[yaxis+"_qual"]==1], x = df_sub[xaxis][df_sub[yaxis+"_qual"]==1], mode = 'markers', marker = dict(color = 'darkred'), name = f'{yaxis} Not trusted (=1)'))
            else: 
                scatter.x = df_sub[xaxis]
                scatter.y = df_sub[yaxis]

    axis_dropdowns = interactive(update_axes, yaxis = df_sub.columns, xaxis = df_sub.columns)

    # Create a table FigureWidget that updates on selection from points in the scatter plot of f
    t = go.FigureWidget([go.Table(
    header=dict(values=variables,
                fill = dict(color='#C2D4FF'),
                align = ['left'] * 5),
    cells=dict(values=[df_sub[col] for col in variables],
               fill = dict(color='#F5F8FF'),
               align = ['left'] * 5))])

    def selection_fn(trace,points,selector):
        t.data[0].cells.values = [df_sub.loc[points.point_inds][col] for col in variables]

    scatter.on_selection(selection_fn)

    # Put everything together
    return VBox((HBox(axis_dropdowns.children),f,t))

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]
