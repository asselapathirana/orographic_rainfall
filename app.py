import sys
import dash
import dash_core_components as dcc
import dash_html_components as html

import numpy as np
from itertools import cycle

app = dash.Dash('Orographic rainfall demo app', static_folder='static')
server = app.server
value_range = [-5, 5]

app.layout = html.Div([
    html.Div([
        html.Div(dcc.Graph(animate=True, id='graph-1'), className="four columns"),
        html.Div(dcc.Graph(animate=True, id='graph-2'), className="eight columns"),
        dcc.Interval(
            id='counter',
                    interval=1*1000, # in milliseconds
                    n_intervals=0
        )        
        
    ], className="row"),
    dcc.RangeSlider(
        id='slider',
        min=value_range[0],
        max=value_range[1],
        step=1,
        value=[-2, 2],
        marks={i: i for i in range(value_range[0], value_range[1]+1)}
    )
], className="container")


#@app.callback(
    #dash.dependencies.Output('graph-1', 'figure'),
    #[dash.dependencies.Input('slider', 'value')])
#def update_graph_1(value):
    #x = np.random.rand(50) 
    #y = np.random.rand(50) 
    #return {
        #'data': [dict({'x': x, 'y': y}, **trace)],
        #'layout': {
            #'xaxis': {'range': [np.min(x), np.max(x)]},
            #'yaxis': {'range': [np.min(y), np.max(y)]}
        #}
    #}

XMAX = 100
XSTEP = 5
iVALUES = np.arange(XMAX/XSTEP)
iVALUES = np.append(iVALUES,[np.NaN,np.NaN])
LX=len(iVALUES)

@app.callback(
    dash.dependencies.Output('graph-2', 'figure'),
    [dash.dependencies.Input('counter', 'n_intervals')
     ]
)
def update_graph_2(counterval):
    xall= XSTEP*iVALUES[:counterval % LX]
    yall= 1000/(1+((xall-50.)/20)**2.)
    if xall[-1]:
        x = [xall[-1]]
        y = [yall[-1]]
    else:
        x = []
        y = []
        xall=[]
        yall=[]
    
    print(x,y, file=sys.stderr)
    
    size = [55 if y>500  else 15 for y in yall ]
        
    trace1={'mode': 'markers',
        'marker': {
            'size': size[-1],
            'opacity': 1.0,
            'color' : 'blue',
            'line': {
                'width': 0.5,
                'color': 'blue'
            }
        }    
    }
    trace2={'mode': 'markers',
        'marker': {
            'size': size,
            'opacity': 0.25,
            'color' : 'blue',
            'line': {
                'width': 0.5,
                'color': 'white'
            }
        }    
    }    
    return {
        'data': [dict({'x': x, 'y': y}, **trace1),
                 dict({'x': xall[:-1], 'y': yall[:-1]}, **trace2)],
        'layout': {
            'xaxis': {'range': [0,100]},
            'yaxis': {'range': [0,1100]}
        }
    }

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == '__main__':
    app.run_server(debug=True)