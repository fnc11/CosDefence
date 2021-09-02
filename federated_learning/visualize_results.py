import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

## project base path
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def compare_histo_plots(trust_df1, trust_df2, overlap=False):
    if overlap:
        pass
    else:
        histo_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
        histo_fig.add_trace(px.histogram(trust_df1, x="modified_trust", color="client_label", color_discrete_map={
                "honest": '#00CC96',
                "minor_offender": '#EF553B',
                "major_offender": '#AB63FA'}), row=1, col=1)
        histo_fig.add_trace(px.histogram(trust_df2, x="modified_trust", color="client_label", color_discrete_map={
                "honest": '#00CC96',
                "minor_offender": '#EF553B',
                "major_offender": '#AB63FA'}), row=1, col=1)
        histo_fig.update_layout(title='Trust given by validation clients', barmode="group")
        histo_fig.show()


def compare_trust_curve_plots(trust_score_df1, trust_score_df2):
    pass


def compare_acc_poison_plots(acc_poison_df1, acc_poison_df2, overlap=False):
    pass
