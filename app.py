# -*- coding: utf-8 -*-
import sys
import dash
import dash_core_components as dcc
import dash_html_components as html
import math

import numpy as np

import metpy.calc as mc
from metpy.units import units, concatenate

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

XMAX = XPEAK*2 # 
XSTEP = 5
XVALUES = np.arange(0,XMAX+.01,XSTEP) # do one number more than XMAX/XSTEP
MTNX=np.arange(-XMAX*.1,XMAX*1.2,1)


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
    windy= windh(windx, height)
    
    
    temp_ = temp*units.degC
    initp = mc.height_to_pressure_std(WINDMTOFFSET*units.meters)
    dewpt = mc.dewpoint_rh(temp_,humid/100.)
    lcl_ = mc.lcl(initp, temp_, dewpt, max_iters=50, eps=1e-5)
    LCL = mc.pressure_to_height_std(lcl_[0])
    ## check if LCL is below the top of the wind profile. 
    pressures = mc.height_to_pressure_std(windy*units.meters)
    # now calculate the air parcel temperatures
    if (LCL>=max(windy)*units.meters):
        T=mc.dry_lapse(pressures, temp_)
    else:
        mini=np.argmin(pressures)
        p1=pressures[:mini]
        p2=pressures[mini-1:] # with an overlap
        
        T1=mc.parcel_profile(p1, temp_, dewpt)
        dwtop=mc.dewpoint_rh(T1[-1], 1.0) # staurated
        T2=mc.dry_lapse(p2,T1[-1])
        T=concatenate((T1,T2[1:]))
        

        
    print(T,humid, lcl_[0], LCL,  file=sys.stderr) 
    size, symbol = zip(*[ (25, 'star') if v*units.meters>LCL and x <= XPEAK else (15, 'circle') for x,v in zip(windx,windy) ])
    
    x = [windx[-1]]
    y = [windy[-1]]        
    
    TC=T.to("degC")
    txt=["{:.1f} °C".format(t) for t in TC.magnitude]
        
    
    trace1={'mode': 'markers',
        'marker': {
            'symbol': symbol[-1],
            'size': size[-1],
            'opacity': 1.0,
            'color' : TC.magnitude,
            'colorbar' : 'Hot',
            'line': {
                'width': 0.5,
                'color': 'blue'
            }
        },
        'text': txt,
        'hoverinfo': 'text',
    }
    trace2={'mode': 'markers',
        'marker': {
            'symbol': symbol,
            'size': size,
            'opacity': 0.25,
               'color' : TC.magnitude,
               'colorscale' : 'Hot',
               'colorbar': {'title':'temp'},
            'line': {
                'width': 0.5,
                'color': 'blue'
            }
        },
        'text': txt,
        'hoverinfo': 'text',
        
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
        'hoverinfo' : 'none',
    }     
    return {
        'data': [dict({'x': x, 'y': y}, **trace1),
                 dict({'x': windx[:-1], 'y': windy}, **trace2),
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
    #update_graph_2(100, 3.897692586860594*1000, 25, 20)
    #update_graph_2(100, 1500, 25, 50)