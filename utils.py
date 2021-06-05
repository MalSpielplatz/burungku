# From here: https://github.com/gaborfodor/wave-bird-recognition/blob/main/birds/display_utils.py
from plotly import graph_objects as go
from plotly import io as pio


def top_bird_bar_plot(top_birds, dx, dy):
    fig = go.Figure(
        go.Bar(
            x=top_birds.p,
            y=top_birds.ebird,
            marker=dict(color=top_birds.color),
            orientation="h",
        )
    )
    _ = fig.update_layout(
        xaxis=dict(showgrid=False, visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        width=dx * 134 - 10,
        height=dy * 76 - 10,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig
