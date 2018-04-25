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

trace = {
    'mode': 'markers',
    'marker': {
        'size': 12,
        'opacity': 0.5,
        'line': {
            'width': 0.5,
            'color': 'white'
        }
    }
}

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


XVALUES = np.arange(0,100,5)
XVALUES = np.append(XVALUES,[np.NaN,np.NaN])
LX=len(XVALUES)

@app.callback(
    dash.dependencies.Output('graph-2', 'figure'),
    [dash.dependencies.Input('counter', 'n_intervals')
     ]
)
def update_graph_2(counterval):
    x= XVALUES[counterval % LX]
    if x:
        y = [np.sin(x/100.*np.pi)]
        x = [x]
    else:
        x = []
        y = []
    return {
        'data': [dict({'x': x, 'y': y}, **trace)],
        'layout': {
            'xaxis': {'range': [0,100]},
            'yaxis': {'range': [0,1.1]}
        }
    }

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == '__main__':
    app.run_server(debug=True)