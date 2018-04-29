# -*- coding: utf-8 -*-
import sys
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import math

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
         dcc.Slider(id='temp', min=-20, max=50, step=1, value=30, 
                   marks={i: i for i in range(-20,50+1,5)}
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
    initp = mc.height_to_pressure_std(windy[0]*units.meters)
    dewpt = mc.dewpoint_rh(temp_,humid/100.)
    lcl_ = mc.lcl(initp, temp_, dewpt, max_iters=50, eps=1e-5)
    LCL = mc.pressure_to_height_std(lcl_[0])
    ## check if LCL is below the top of the wind profile. 
    pressures = mc.height_to_pressure_std(windy*units.meters)
    
    wvmr0 = mc.mixing_ratio_from_relative_humidity(humid/100., temp_, initp)
    
    # now calculate the air parcel temperatures and RH at each position
    if (lcl_[0]<=min(pressures)):
        T=mc.dry_lapse(pressures, temp_)
        RH= [ mc.relative_humidity_from_mixing_ratio(wvmr0, t, p) for t,p in zip(T,pressures)]
    else:
        mini=np.argmin(pressures)
        p1=pressures[:mini]
        p2=pressures[mini-1:] # with an overlap
        T1=parcel_profile(p1, temp_, dewpt) # see thero.py 354
        dwtop=mc.dewpoint_rh(T1[-1], 1.0) # staurated
        T2=mc.dry_lapse(p2,T1[-1])
        T=concatenate((T1,T2[1:]))
        wvmrtop = mc.saturation_mixing_ratio(pressures[mini],T[mini])
        
        RH= [ mc.relative_humidity_from_mixing_ratio(wvmr0, *tp) if tp[1]>lcl_[0] and i<=mini else 1.0 if i<mini else  
              mc.relative_humidity_from_mixing_ratio(wvmrtop, *tp)
              for i,tp in enumerate(zip(T,pressures))]

    RH=concatenate(RH)
    
    
    x = [windx[-1]]
    y = [windy[-1]]        
    rhlast=float(RH.magnitude[-1]*100.)
    TC=T.to("degC")
    print(rhlast, RH.magnitude[-1]*100., RH, TC,humid, lcl_[0], LCL,  file=sys.stderr) 
    txt=["{:.1f} °C/ {:.0f} %".format(t,rh*100.) for t,rh in zip(TC.magnitude,RH.magnitude)]
    size, symbol = zip(*[ (15, 'circle') if v*units.meters<LCL or x > XPEAK else (25, "star") if t>0*units.degC  else (30, 'hexagram') for x,v, t in zip(windx,windy,TC) ])
    
        
    
    colorscale='Viridis'
    
    trace1={'mode': 'markers',
            'marker': {
            'size': 40,
            'color': 'black',
            'symbol': 'y-right-open',},
            'showlegend': False,
            'hoverinfo' : 'none',
    }
    trace2={'mode': 'markers',
        'marker': {
            'symbol': symbol,
            'size': size,
            'opacity': 1.0,
               'color' : RH.magnitude*100.,
               'colorscale' : colorscale,
               'cmin' : 0,
               'cmax' : 100.,
               'reversescale': True,
               'colorbar': {'title':'RH (%)'},
            'line': {
                'width': 0.5,
                'color': 'black'
            }
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
    return {
        'data': [dict({'x': windx, 'y': windy}, **trace2),
                 dict({'x': x, 'y': y}, **trace1),  
                 dict({'x': MTNX, 'y': mtny}, **trace3),
                 ],
        'layout': {
            'xaxis': {'range': [0,XMAX*1.05]},
            'yaxis': {'range': [0,1.1*windh(0, MAXMNHT,  xoffset=0)]},
        }
    }

def windh(xval, maxht, xoffset=XPEAK, div=SHAPEFA, ratio=WINDMTRATIO, yoffset=WINDMTOFFSET):
    return maxht*ratio/(1+((xval-xoffset)/div)**2.) + yoffset
    
    
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == '__main__':
    app.run_server(debug=True)
    #update_graph_2(100, 3.897692586860594*1000, 25, 20)
    #update_graph_2(100, 1500, 25, 50)
    #update_graph_2(100, 1000, 30, 40)
    #update_graph_2(100,1500,30,20)
    #update_graph_2(5,1500,30,20)