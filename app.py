# -*- coding: utf-8 -*-
import sys
import dash
import dash_core_components as dcc
import dash_html_components as html
import math

import numpy as np
from itertools import cycle

app = dash.Dash('Orographic rainfall demo app', static_folder='static')
server = app.server
value_range = [-5, 5]
ANIM_DELTAT = 500
MAXMNHT = 5000
WINDMTRATIO = 2
WINDMTOFFSET = 1000

XMAX = 100
XSTEP = 5
iVALUES = np.arange(0,XMAX/XSTEP) # do one number more than XMAX/XSTEP

app.layout = html.Div([     

    html.Div([
        html.Div(dcc.Graph(animate=True, id='graph-1'), className="four columns"),
        html.Div(dcc.Graph(animate=False, id='graph-2'), className="eight columns"),
        dcc.Interval(
            id='ncounter',
                    interval=ANIM_DELTAT, 
                    n_intervals=0
        )        
        
    ], className="row"),
    html.Div(
        [html.Div('Mountain Height'), 
         dcc.Slider(id='height', min=0, max=MAXMNHT, step=500, value=1500, 
                   marks={i: i for i in range(0,MAXMNHT+1,1000)}
        ),
         ],
        className="four columns"),
    html.Div(
        [html.Div('Humidity (%)'), 
         dcc.Slider(id='humid', min=0, max=100, step=1, value=50, 
                   marks={i: i for i in range(0,100+1,10)}
        ),
         ],
        className="four columns"),
    html.Div(
        [html.Div('Temperature near surface (°C)'), 
         dcc.Slider(id='temp', min=0, max=100, step=1, value=50, 
                   marks={i: i for i in range(0,100+1,10)}
        ),
         ],
        className="four columns", style={"margin-top": "25px"}),
     
], className="container")


@app.callback(
    dash.dependencies.Output('ncounter', 'n_intervals'),
[dash.dependencies.Input('height','value'),
 dash.dependencies.Input('temp','value'),
 dash.dependencies.Input('humid','value'),
 ],    
)
def reset_counter(height,temp,humid):
    # update_graph_2(-1, height, temp, humid)
    return 0


@app.callback(
    dash.dependencies.Output('graph-2', 'figure'),
    [dash.dependencies.Input('ncounter', 'n_intervals')
     ],
    [dash.dependencies.State('height','value'),
     dash.dependencies.State('temp','value'),
     dash.dependencies.State('humid','value'),
     ]
)
def update_graph_2(counterval, height, temp, humid):
    wh = windh(height)
    length = min([counterval,len(iVALUES)])
    xall= XSTEP*iVALUES[:1 + (length)]
    yall= wh/(1+((xall-50.)/20)**2.)
    x = [xall[-1]]
    y = [yall[-1]]    
    size = [55 if v>500  else 15 for v in yall ]
        
    
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
            'yaxis': {'range': [0,1.1*windh(MAXMNHT)]}
        }
    }

def windh(height):
    return height*WINDMTRATIO+WINDMTOFFSET
    
    
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == '__main__':
    app.run_server(debug=True)