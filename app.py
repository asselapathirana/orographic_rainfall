# -*- coding: utf-8 -*-
import sys
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import math
import json

import numpy as np

import metpy.calc as mc
from metpy.units import units, concatenate, check_units

from itertools import cycle

#########################################################################################
def parcel_profile(pressure, temperature, dewpt):
    r"""Calculate the profile a parcel takes through the atmosphere.

    The parcel starts at `temperature`, and `dewpt`, lifted up
    dry adiabatically to the LCL, and then moist adiabatically from there.
    `pressure` specifies the pressure levels for the profile.

    Parameters
    ----------
    pressure : `pint.Quantity`
        The atmospheric pressure level(s) of interest. The first entry should be the starting
        point pressure.
    temperature : `pint.Quantity`
        The starting temperature
    dewpt : `pint.Quantity`
        The starting dew point

    Returns
    -------
    `pint.Quantity`
        The parcel temperatures at the specified pressure levels.

    See Also
    --------
    lcl, moist_lapse, dry_lapse

    """
    # Find the LCL
    lcl_pressure = mc.lcl(pressure[0], temperature, dewpt)[0].to(pressure.units)

    # Find the dry adiabatic profile, *including* the LCL. We need >= the LCL in case the
    # LCL is included in the levels. It's slightly redundant in that case, but simplifies
    # the logic for removing it later.
    press_lower = concatenate((pressure[pressure >= lcl_pressure], lcl_pressure))
    t1 = mc.dry_lapse(press_lower, temperature)

    # Find moist pseudo-adiabatic profile starting at the LCL
    press_upper = concatenate((lcl_pressure, pressure[pressure < lcl_pressure]))
    t2 = mc.moist_lapse(press_upper, t1[-1]).to(t1.units)

    # Return LCL *without* the LCL point
    if t2.size>1:
        return concatenate((t1[:-1], t2[1:]))
    else:
        return t1[:-1]

#########################################################################################

app = dash.Dash('Orographic rainfall demo app', static_folder='static')
server = app.server
value_range = [-5, 5]
ANIM_DELTAT = 500
MAXMNHT = 2500
WINDMTRATIO = 2
WINDMTOFFSET = 1000
XPEAK = 100 # x value at which peak occures
SHAPEFA = 20

XMAX = XPEAK*2 # 
XSTEP = 10
XVALUES = np.append(-99999999,np.arange(0,XMAX+.01,XSTEP),99999999) # do one number more than XMAX/XSTEP then have inf on each side. 
MTNX=np.arange(-XMAX*.1,XMAX*1.2,1)

# symbol size and name
sym_nop = (15, 'circle')
sym_lp = (25, "star")
sym_ip = (30, 'hexagram')
sym_parcel = (50, 'y-right-open')

row1 =  html.Div([ # row 1 start ([
        html.Div(dcc.Graph(animate=False, id='graph-2'), className="eight columns"),
        html.Div(
            [html.Div(dcc.Graph(animate=False, id='graphRHEl') , className="row"),
            html.Div(dcc.Graph(animate=False, id='graphTEl') , className="row"),
            html.Div(dcc.Interval(id='ncounter', interval=ANIM_DELTAT, n_intervals=0 )),  # no display
            html.Div(id='calculations_store', style={'display': 'none'})                  # no display
            ], className="four columns"), 
    ], className="row") # row 1 end ])

slider1=html.Div([
    html.Div('Mountain Height'), dcc.Slider(id='height', min=0, max=MAXMNHT, step=500, value=1500,  marks={i: i for i in range(0,MAXMNHT+1,1000)}),
    ],className="four columns")
slider2=html.Div([
    html.Div('Humidity of air (%)'), dcc.Slider(id='humid', min=0, max=100, step=1, value=40,  marks={i: i for i in range(0,100+1,10)}),
    ], className="four columns")
slider3=html.Div([
    html.Div('Temperature of air (°C)'), dcc.Slider(id='temp', min=-20, max=50, step=1, value=30, marks={i: i for i in range(-20,50+1,5)}, ),
    ],className="four columns",) #style={"margin-top": "25px"}


row2 = html.Div([ # begin row 2
    slider1,
    slider2,
    slider3,
    ], className = "row") # end row 2

app.layout = html.Div([  # begin container
    row1,
    row2,
    ], className="container") # end container


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
dash.dependencies.Output('calculations_store','children'),
[dash.dependencies.Input('height','value'),
 dash.dependencies.Input('temp','value'),
 dash.dependencies.Input('humid','value'),
 ]    
)
def calculate_set(height,temp,humid):
    st=json.dumps(saveCalc(humid, temp, height))
    return st

@app.callback(
    dash.dependencies.Output('graphRHEl', 'figure'),
    [dash.dependencies.Input('ncounter', 'n_intervals'),
     ],
    [dash.dependencies.State('calculations_store','children'),
    ]
)
def update_RHElGraph(counterval, calculation_store_data):
    windy, windx, mtny, TC, RH, trace = json.loads(calculation_store_data)
    length = min([counterval,len(XVALUES)])    
    print("HEEEE",counterval, length, RH[:length],   file=sys.stderr)
    return {
    'data': [{'x':RH[:length],'y':windy[:length]}], 
    'layout':{'xaxis': {'range': [0,105], 'title': 'RH (%)'},
            'yaxis': {'range': [min(windy),max(windy)], 'title': 'Elevation (m)'},
            'height': 200,
            'margin': {
                'l': 60,
                'r': 40,
                'b': 40,
                't': 10,
                'pad': 4
              },             
            },
    }


@app.callback(
    dash.dependencies.Output('graphTEl', 'figure'),
    [dash.dependencies.Input('ncounter', 'n_intervals'),
     ],
    [dash.dependencies.State('calculations_store','children'),
    ]
)
def update_TElGraph(counterval, calculation_store_data):
    windy, windx, mtny, TC, RH, trace = json.loads(calculation_store_data)
    length = min([counterval,len(XVALUES)])    
    print("HEEEE",counterval, length, RH[:length],   file=sys.stderr)
    return {
    'data': [{'x':TC[:length],'y':windy[:length]}], 
    'layout':{'xaxis': {'range': [min(TC),max(TC)], 'title': 'T (°C)'},
            'yaxis': {'range': [min(windy),max(windy)], 'title': 'Elevation (m)'},
            'height': 200,
            'margin': {
                'l': 60,
                'r': 40,
                'b': 40,
                't': 10,
                'pad': 4
              },             
            },
    }




@app.callback(
    dash.dependencies.Output('graph-2', 'figure'),
    [dash.dependencies.Input('ncounter', 'n_intervals')
     ],
    [dash.dependencies.State('calculations_store','children'),
    ]
)
def update_mainGraph(counterval, calculation_store_data):
    
    windy, windx, mtny, TC, RH, trace = json.loads(calculation_store_data)
    length = min([counterval,len(XVALUES)])
    x = [windx[length-1]]
    y = [windy[length-1]] 
    
    return {
        'data': [dict({'x': windx[:length], 'y': windy[:length]}, **trace[1]),
                 dict({'x': x, 'y': y}, **trace[0]),  
                 dict({'x': MTNX, 'y': mtny}, **trace[2]),
                 dict({'x':[-99999], 'y':[-99999]}, **trace[3]),                 
                 dict({'x':[-99999], 'y':[-99999]}, **trace[4]),
                 dict({'x':[-99999], 'y':[-99999]}, **trace[5]),
                 dict({'x':[-99999], 'y':[-99999]}, **trace[6]),
                 ],
        'layout': {
            'xaxis': {'range': [0,XMAX*1.05], 'title': 'Distance (km)'},
            'yaxis': {'range': [0,1.1*windh(0, MAXMNHT,  xoffset=0)], 'title': 'Elevation (m)'},
            'margin': {
                'l': 60,
                'r': 40,
                'b': 40,
                't': 10,
                'pad': 4
              },   
            'legend': {'x':.75, 'y':.8},
        }
    }

def saveCalc(humid, temp, height):
    windx, mtny, windy, lcl_, LCL, TC, RH = atmCalc(height, temp, humid)
           
    print(RH, TC,humid, lcl_[0], LCL,  file=sys.stderr) 
    txt=["{:.1f} °C/ {:.0f} %".format(t,rh*100.) for t,rh in zip(TC.magnitude,RH.magnitude)]
    
    
    colorscale='Viridis'
   
    
    size, symbol = zip(*[ sym_nop if v*units.meters<LCL or x > XPEAK else sym_lp if t>0*units.degC  else sym_ip for x,v, t in zip(windx,windy,TC) ])
    
    trace1={'mode': 'markers',
            'marker': {
            'size': sym_parcel[0],
            'color': 'black',
            'symbol': sym_parcel[1],},
            'showlegend': False,
            'hoverinfo' : 'none',
    }
    trace2={'mode': 'markers',
        'marker': {
            'symbol': symbol,
            'size': size,
            'opacity': 1.0,
               'color' : (RH.magnitude*100.).tolist(), # no numpy
               'colorscale' : colorscale,
               'cmin' : 0,
               'cmax' : 100.,
               'reversescale': True,
               'colorbar': {'title':'RH (%)'},
         #   'line': {
         #       'width': 0.5,
         #       'color': 'black'
         #   }
        },
        'text': txt,
        'hoverinfo': 'text',
        'showlegend': False,
        
    }
    trace3={    
        'fill' : 'tozeroy',
        'hoverinfo' : 'none',
        'showlegend': False,
        
    }   
    
    tr1 = {'mode': 'markers',
        'marker': {
            'symbol': sym_parcel[1],
            'size': 15,
            'color': 'black',
            },
            'name' : 'Air parcel',
        'showlegend': True,
        }
    tr2 =  {'mode': 'markers',
        'marker': {
            'symbol': sym_nop[1],
            'size': 15,
            'color': 'black',
            },
            'name' : 'No precip',
        'showlegend': True,
        }
    tr3 =  {'mode': 'markers',
        'marker': {
            'symbol': sym_lp[1],
            'size': 15,
            'color': 'black',
            },
            'name' : 'Liquid precip',
        'showlegend': True,
        }
    tr4 =  {'mode': 'markers',
        'marker': {
            'symbol': sym_ip[1],
            'size': 15,
            'color': 'black',
            },
            'name' : 'Ice precip',
        'showlegend': True,
        }
    
    trace=[trace1, trace2, trace3, tr1, tr2, tr3, tr4]
    RH=RH*100.
    return windy.tolist(), windx.tolist(), mtny.tolist(), TC.magnitude.tolist(), RH.magnitude.tolist(), trace # no numpy

def atmCalc(height, temp, humid):
    windx= XVALUES
    mtny=windh(MTNX, height, ratio=1, 
              yoffset=0)
    windy= windh(windx, height)
    
    
    temp_ = temp*units.degC
    initp = mc.height_to_pressure_std(windy[0]*units.meters)
    dewpt = mc.dewpoint_rh(temp_,humid/100.)
    lcl_ = mc.lcl(initp, temp_, dewpt, max_iters=50, eps=1e-5)
    LCL = mc.pressure_to_height_std(lcl_[0])
    pressures = mc.height_to_pressure_std(windy*units.meters)
    
    wvmr0 = mc.mixing_ratio_from_relative_humidity(humid/100., temp_, initp)
    
    # now calculate the air parcel temperatures and RH at each position
    if (lcl_[0]<=min(pressures)):
        T=mc.dry_lapse(pressures, temp_)
        RH= [ mc.relative_humidity_from_mixing_ratio(wvmr0, t, p) for t,p in zip(T,pressures)]
    else:
        mini=np.argmin(pressures)
        p1=pressures[:mini+1]
        p2=pressures[mini:] # with an overlap
        T1=parcel_profile(p1, temp_, dewpt) # see thero.py 354
        dwtop=mc.dewpoint_rh(T1[-1], 1.0) # staurated
        T2=mc.dry_lapse(p2,T1[-1])
        T=concatenate((T1,T2[1:]))
        wvmrtop = mc.saturation_mixing_ratio(pressures[mini],T[mini])
        
        RH= [ mc.relative_humidity_from_mixing_ratio(wvmr0, *tp) if tp[1]>lcl_[0] and i<=mini else 1.0 if i<mini else  
              mc.relative_humidity_from_mixing_ratio(wvmrtop, *tp)
              for i,tp in enumerate(zip(T,pressures))]

    RH=concatenate(RH)
    return windx, mtny, windy, lcl_, LCL, T.to("degC"), RH

def windh(xval, maxht, xoffset=XPEAK, div=SHAPEFA, ratio=WINDMTRATIO, yoffset=WINDMTOFFSET):
    return maxht*ratio/(1+((xval-xoffset)/div)**2.) + yoffset
    
    
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == '__main__':
    app.run_server(debug=True)
    #d=calculate_set(3.897692586860594*1000, 25, 20)
    #d=calculate_set(1500, 25, 50)
    #d=calculate_set(1000, 30, 40)
    #d=calculate_set(1500,30,20)
    #d=calculate_set(1500,30,20)
    #update_mainGraph(150,d)