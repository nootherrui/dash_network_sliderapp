import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import ast

app = dash.Dash(__name__)
server = app.server
# ---------- Step 1: 数据读取与预处理 ----------
df_all = pd.read_csv('nw_id_4p.csv')

def parse_neighbors(val):
    try:
        neighbors = ast.literal_eval(val) if pd.notna(val) else []
        return [str(n) for n in neighbors]
    except (ValueError, SyntaxError):
        return []

df_all['neighbor_list'] = df_all['neighbor_choice'].apply(parse_neighbors)

# ---------- Step 2: 图形生成 ----------
def build_figure(graph_data, hover_node_id=None):
    if not graph_data or not graph_data.get('nodes'):
        return go.Figure()
    nodes_with_data = graph_data['nodes']
    edges = graph_data['edges']
    node_pos = {int(k): v for k, v in graph_data['node_pos'].items()}
    neighbor_map = graph_data['neighbor_map']
    node_ids = [data[0] for data in nodes_with_data]
    highlight_nodes = set()
    info_text = ""
    annotations = []
    if hover_node_id and str(hover_node_id) in neighbor_map:
        hovered_id_str = str(hover_node_id)
        neighbors_str = neighbor_map.get(hovered_id_str, [])
        highlight_nodes = {hovered_id_str} | set(neighbors_str)
        hovered_id_int = int(hovered_id_str)
        node_info = next((n[1] for n in nodes_with_data if n[0] == hovered_id_int), {})
        info_text = f"<b>ID:</b> {hovered_id_str}<br><b>Category:</b> {node_info.get('cat', 'N/A')}<br><b>Opinion:</b> {node_info.get('op', 'N/A')}<br><b>Neighbors:</b> {neighbors_str}"
        annotations.append(dict(
            x=0.01, y=0.01, xref='paper', yref='paper', xanchor='left', yanchor='bottom',
            text=info_text, showarrow=False, align='left', font=dict(size=14),
            bgcolor='rgba(245, 245, 245, 0.85)', bordercolor='black', borderwidth=1, borderpad=4
        ))
    node_sizes = [35 if str(nid) in highlight_nodes else 29 for nid in node_ids]
    border_colors = ['red' if str(nid) in highlight_nodes else 'black' for nid in node_ids]
    edge_x, edge_y = [], []
    for i, j in edges:
        x0, y0 = node_pos.get(i, [None, None])
        x1, y1 = node_pos.get(j, [None, None])
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#888'), mode='lines',
                            showlegend=False, hoverinfo='none')
    node_x = [node_pos[nid][0] for nid in node_ids]
    node_y = [node_pos[nid][1] for nid in node_ids]
    node_colors = [data[1]['color'] for data in nodes_with_data]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='none',
        customdata=[str(nid) for nid in node_ids],
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=3, color=border_colors)),
        showlegend=False
    )
    in_legend = go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=18, color='skyblue'), name='Introvert'
    )
    ex_legend = go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=18, color='orange'), name='Extrovert'
    )
    text_trace = go.Scatter(
        x=node_x, y=node_y, text=[str(nid) for nid in node_ids],
        mode='text', textposition="middle center", hoverinfo='none',showlegend=False
    )
    fig = go.Figure(data=[edge_trace, node_trace, text_trace, in_legend, ex_legend], layout=go.Layout(
        showlegend=True, hovermode='closest', margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(
                x=0.99, y=0.99,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.6)',
                borderwidth=2),
        annotations=annotations,
        transition={'duration': 100}
    ))
    return fig

# ---------- Step 3: Dash布局 ----------
slider_map = {0: 'R1_pre', 1: 'R1_post', 2: 'R2_pre', 3: 'R2_post'}
initial_phase = slider_map[0]
df_initial = df_all[df_all['phase_label'] == initial_phase].copy()
def process_data_for_phase(df):
    if df.empty:
        return {'nodes': [], 'edges': [], 'node_pos': {}, 'neighbor_map': {}}
    nodes = df['old_user_id'].values
    scores = df['op'].values
    edges, dists = [], []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            dist = abs(scores[i] - scores[j])
            edges.append((nodes[i], nodes[j]))
            dists.append(dist)
    weights = []
    if dists:
        scaler = MinMaxScaler()
        weights = 1 - scaler.fit_transform(pd.DataFrame(dists)).flatten()
    G = nx.Graph()
    color_map = {'Introvert': 'skyblue', 'Extrovert': 'orange'}
    for _, row in df.iterrows():
        node_id = row['old_user_id']
        G.add_node(node_id, op=row['op'], cat=row['cat'], color=color_map.get(row['cat'], 'gray'))
    for (u, v), w in zip(edges, weights):
        if w >= 0.5:
            G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G, weight='weight', seed=42)
    node_pos = {n: [p[0], p[1]] for n, p in pos.items()}
    neighbor_map = dict(zip(df['old_user_id'].astype(str), df['neighbor_list']))
    return {
        'nodes': list(G.nodes(data=True)),
        'edges': list(G.edges()),
        'node_pos': node_pos,
        'neighbor_map': neighbor_map
    }
initial_graph_data = process_data_for_phase(df_initial)
initial_figure = build_figure(initial_graph_data, hover_node_id=None)
container_width = '1000px'

app.layout = html.Div([
    dcc.Store(id='graph-data-store', data=initial_graph_data),
    dcc.Store(id='reset-pending', data=False),
    html.H4("Drag the Phase Slider and Hover Over the Nodes for Details", style={
        'textAlign': 'center',
        'fontSize': '21px',
        'fontWeight': 'bold',
        'letterSpacing': '0.5px',
        'color': '#222'
    }),
    html.Div([
        html.Div(
            dcc.Slider(
                id='phase-selector',
                min=0, max=3, step=None, value=0,
                marks={i: v for i, v in slider_map.items()},
            ),
            style={
                'width': '500px',
                'marginRight': '80px'
            }
        ),
        html.Button("Reset", id='reset-btn', style={
            'fontWeight': 'bold',
            'padding': '6px 18px',
            'fontSize': '18px',
            'height': '38px',
            'alignSelf': 'center',
            'borderRadius': '8px',
            'marginRight': '10px',
            'boxShadow': '0 2px 6px rgba(0,0,0,0.12)'
        }),
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'alignItems': 'center',
        'justifyContent': 'center',
        'width': container_width,
        'margin': '0 auto',
        'padding': '18px 0 22px 0',
    }),
    dcc.Graph(id='network-graph', figure=initial_figure, style={
        'height': '80vh',
        'width': container_width,
        'margin': '0 auto',
        'display': 'block',
        'background': '#e8eef6'
    })
])

# ----------- 关键回调部分（重点只要这两个回调即可） ------------

# 回调1：点击reset才清空高亮，切换phase保留高亮
@app.callback(
    [Output('graph-data-store', 'data'),
     Output('reset-pending', 'data', allow_duplicate=True),
     Output('network-graph', 'hoverData', allow_duplicate=True)],
    [Input('phase-selector', 'value'),
     Input('reset-btn', 'n_clicks')],
    [State('network-graph', 'hoverData')],
    prevent_initial_call=True
)
def handle_slider_and_reset(selected_phase_val, n_clicks, prev_hoverData):
    from dash import ctx
    phase = slider_map[selected_phase_val]
    df_phase = df_all[df_all['phase_label'] == phase].copy()
    if ctx.triggered_id == 'reset-btn':
        return dash.no_update, True, None      # reset：清空hover
    else:
        return process_data_for_phase(df_phase), False, prev_hoverData  # phase切换：保留hover

# 回调2：主图渲染
@app.callback(
    [Output('network-graph', 'figure'),
     Output('reset-pending', 'data', allow_duplicate=True)],
    [Input('graph-data-store', 'data'),
     Input('network-graph', 'hoverData'),
     Input('reset-pending', 'data')],
    prevent_initial_call=True
)
def update_graph_figure(graph_data, hoverData, reset_pending):
    if reset_pending:
        return build_figure(graph_data, hover_node_id=None), False
    hover_node = None
    if hoverData and 'points' in hoverData and hoverData['points'][0]['curveNumber'] == 1:
        hover_node = hoverData['points'][0]['customdata']
    return build_figure(graph_data, hover_node), False

if __name__ == '__main__':
    app.run(debug=False)
