# -*- coding: utf-8 -*-
import sys
import dash
import dash_core_components as dcc
import dash_html_components as html
import math

import numpy as np

import metpy.calc as mc
from metpy.units import units

from itertools import cycle

app = dash.Dash('Orographic rainfall demo app', static_folder='static')
server = app.server
value_range = [-5, 5]
ANIM_DELTAT = 500
MAXMNHT = 2500
WINDMTRATIO = 2
WINDMTOFFSET = 1000
XPEAK = 50 # x value at which peak occures
SHAPEFA = 20

XMAX = 100
XSTEP = 5
XVALUES = np.arange(0,XMAX+.01,XSTEP) # do one number more than XMAX/XSTEP
MTNX=np.arange(-XMAX*.1,XMAX*1.2,1)
# calculate a pressure profile. 

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
        [html.Div('Humidity of air (%)'), 
         dcc.Slider(id='humid', min=0, max=100, step=1, value=40, 
                   marks={i: i for i in range(0,100+1,10)}
        ),
         ],
        className="four columns"),
    html.Div(
        [html.Div('Temperature of air (°C)'), 
         dcc.Slider(id='temp', min=0, max=100, step=1, value=30, 
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
    length = min([counterval,len(XVALUES)])
    windx= XVALUES[:length]
    mtny=windh(MTNX, height, ratio=1, 
              yoffset=0)
    yall= windh(windx, height)
    x = [windx[-1]]
    y = [yall[-1]]    
    
    temp_ = temp*units.degC
    initp = mc.height_to_pressure_std(WINDMTOFFSET*units.meters)
    dewpt = mc.dewpoint_rh(temp_,humid/100.)
    lcl_ = mc.lcl(initp, temp_, dewpt, max_iters=50, eps=1e-5)
    LCL = mc.pressure_to_height_std(lcl_[0])
    ## check if LCL is below the top of the wind profile. 
    #pressures = mc.height_to_pressure_std(yall*units.meters)
    #if max(mtny) < LCL:
        #temps = mc.dry_lapse(pressures, temp_)
    #else:
        #LCLx=
        
    print(humid, lcl_[0], LCL,  file=sys.stderr) 
    size, symbol = zip(*[ (25, 'star') if v*units.meters>LCL and x <= XPEAK else (15, 'circle') for x,v in zip(windx,yall) ])
        
    
    trace1={'mode': 'markers',
        'marker': {
            'symbol': symbol[-1],
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
            'symbol': symbol,
            'size': size,
            'opacity': 0.25,
            'color' : 'blue',
            'line': {
                'width': 0.5,
                'color': 'white'
            }
        }    
    } 
    
    trace3={#'mode': 'markers',
        #'marker': {
        #    'symbol': symbol,
        #    'size': size,
        #    'opacity': 0.25,
        #    'color' : 'blue',
        #    'line': {
        #        'width': 0.5,
        #        'color': 'white'
        #    }
        #}    
        'fill' : 'tozeroy',
    }     
    return {
        'data': [dict({'x': x, 'y': y}, **trace1),
                 dict({'x': windx[:-1], 'y': yall}, **trace2),
                 dict({'x': MTNX, 'y': mtny}, **trace3),
                 ],
        'layout': {
            'xaxis': {'range': [0,XMAX*1.05]},
            'yaxis': {'range': [0,1.1*windh(0, MAXMNHT,  xoffset=0)]}
        }
    }

def windh(xval, maxht, xoffset=XPEAK, div=SHAPEFA, ratio=WINDMTRATIO, yoffset=WINDMTOFFSET):
    return maxht*ratio/(1+((xval-xoffset)/div)**2.) + yoffset
    
    
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == '__main__':
    app.run_server(debug=True)