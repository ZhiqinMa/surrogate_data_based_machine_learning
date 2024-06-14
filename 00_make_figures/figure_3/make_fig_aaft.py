# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 19:39:06 2021

Make a figure that includes:
- Trajectory and smoothing
- Variance
- Lag-1 AC
- SDML predictions

@author: Zhiqin Ma and Thomas M. Bury
"""

import os
import time
start_time = time.time()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


try:
    os.mkdir('output_data')
except:
    print('output_data directory already exists!')

try:
    os.mkdir('output_figures')
except:
    print('output_figures directory already exists!')
try:
    os.mkdir('output_figures/single_figs')
except:
    print('output_figures/single_figs directory already exists!\n')


# Define anoxia transitions function
def create_sub_df_anoxia(df, core, id, record, tsid_new):
    sub_df = df[(df['Core'] == core) & (df['ID'] == id)]
    return pd.DataFrame({
        'Age': sub_df['Time (kyr BP)'],
        'Proxy': sub_df['Mo [ppm]'],
        'X_axis_label': 'Time (kyr BP)',
        'Y_axis_label': 'Mo [ppm]',
        'Record': record,
        'Transition_start': -sub_df['t_transition_start'],
        'Transition_end': -sub_df['t_transition_end'],
        'tsid_new': tsid_new
    })

# Define ews anoxia function
def create_ews_df_anoxia(df, var_label, tsid, tsid_new):
    ews_df = df[(df['Variable_label'] == var_label) & (df['tsid'] == tsid)]
    return pd.DataFrame({
        'Age': ews_df['Age [ka BP]'],
        'state': ews_df['state'],
        'smoothing': ews_df['smoothing'],
        'residuals': ews_df['residuals'],
        'variance': ews_df['variance'],
        'ac1': ews_df['ac1'],
        'tsid_new': tsid_new
    })


# Define  paleoclimate function
def create_sub_df_paleoclimate(df, period, record, tsid_new):
    sub_df = df[(df['Period'] == period)]
    return pd.DataFrame({
        'Age': sub_df['Age'],
        'Proxy': sub_df['Proxy'],
        'X_axis_label': 'Time (yr BP)',
        'Y_axis_label': "\u03B4D (\u2030)",
        'Record': record,
        'Transition_start': -sub_df['Transition_start'],
        'Transition_end': -sub_df['Transition_end'],
        'tsid_new': tsid_new
    })

# Define ews paleoclimate function
def create_ews_df_paleoclimate(df, period, tsid_new):
    ews_df = df[(df['Period'] == period)]
    return pd.DataFrame({
        'Age': ews_df['Age'],
        'state': ews_df['state'],
        'smoothing': ews_df['smoothing'],
        'residuals': ews_df['residuals'],
        'variance': ews_df['variance'],
        'ac1': ews_df['ac1'],
        'tsid_new': tsid_new
    })


# Define tree_felling function
def create_sub_df_tree_felling(df, period, record, tsid_new):
    sub_df = df[(df['Period'] == period)]
    return pd.DataFrame({
        'Age': sub_df['Age'],
        'Proxy': sub_df['tree_felling'],
        'X_axis_label': 'Time (yr AD)',
        'Y_axis_label': 'Construction activity',
        # 'Y_axis_label': 'CA',
        # 'Y_axis_label': 'Number of Trees',
        # 'Y_axis_label': 'Tree numbers',
        'Record': record,
        'Transition_start': sub_df['Transition_start'],
        'Transition_end': sub_df['Transition_end'],
        'tsid_new': tsid_new
    })

# Define ews tree_felling function
def create_ews_df_tree_felling(df, period, tsid_new):
    ews_df = df[(df['Period'] == period)]
    return pd.DataFrame({
        'Age': ews_df['Age'],
        'state': ews_df['state'],
        'smoothing': ews_df['smoothing'],
        'residuals': ews_df['residuals'],
        'variance': ews_df['variance'],
        'ac1': ews_df['ac1'],
        'tsid_new': tsid_new
    })




# # ------------------------
# # 01 Load transition data
# # ------------------------

# # Load anoxia transition data
df_transitions_anoxia = pd.read_csv('input_data/transitions/anoxia_transitions.csv')
df_transitions_anoxia['Time (kyr BP)'] = -df_transitions_anoxia['Age [ka BP]']
# print(df_transitions_anoxia)

# # Load paleoclimate transition data
df_transitions_paleoclimate = pd.read_csv('input_data/transitions/paleoclimate_transitions_interpolate.csv')
df_transitions_paleoclimate['Age'] = -df_transitions_paleoclimate['Age']
# print(df_transitions_paleoclimate)

# # Load tree_felling transition data
df_transitions_tree_felling = pd.read_csv('input_data/transitions/tree_felling_transitions.csv')
# print(df_transitions_tree_felling)

# # Define column names
columns = ['Age', 'Proxy', 'X_axis_label', 'Y_axis_label' 'Record', 'Transition_start', 'Transition_end', 'Tsid']
# Create new DataFrame
df_transitions = pd.DataFrame(columns=columns)

# MS21-S1
df_transitions_anoxia_MS21_S1 = create_sub_df_anoxia(df_transitions_anoxia, 'MS21', 'S1', 'MS21-S1', 1)

# MS66-S1
df_transitions_anoxia_MS66_S1 = create_sub_df_anoxia(df_transitions_anoxia, 'MS66', 'S1', 'MS66-S1', 2)
# MS66-S3
df_transitions_anoxia_MS66_S3 = create_sub_df_anoxia(df_transitions_anoxia, 'MS66', 'S3', 'MS66-S3', 3)

# 64PE-S3
df_transitions_anoxia_64PE_S3 = create_sub_df_anoxia(df_transitions_anoxia, '64PE', 'S3', '64PE-S3', 4)
# 64PE-S4
df_transitions_anoxia_64PE_S4 = create_sub_df_anoxia(df_transitions_anoxia, '64PE', 'S4', '64PE-S4', 5)
# 64PE-S5
df_transitions_anoxia_64PE_S5 = create_sub_df_anoxia(df_transitions_anoxia, '64PE', 'S5', '64PE-S5', 6)
# 64PE-S6
df_transitions_anoxia_64PE_S6 = create_sub_df_anoxia(df_transitions_anoxia, '64PE', 'S6', '64PE-S6', 7)

# End of glaciation I
df_transitions_paleoclimate_I = create_sub_df_paleoclimate(df_transitions_paleoclimate, 'I', 'Glaciation-I', 8)
# End of glaciation II
df_transitions_paleoclimate_II = create_sub_df_paleoclimate(df_transitions_paleoclimate, 'II', 'Glaciation-II', 9)

# P II
df_transitions_tree_felling_PII = create_sub_df_tree_felling(df_transitions_tree_felling, 'PII', 'PII', 10)
# Early P III
df_transitions_tree_felling_Early_PIII = create_sub_df_tree_felling(df_transitions_tree_felling, 'EPIII', 'Early PIII', 11)
# Early P III
df_transitions_tree_felling_Late_PIII = create_sub_df_tree_felling(df_transitions_tree_felling, 'LPIII', 'Late PIII', 12)

# concat df
df_transitions = pd.concat([df_transitions_anoxia_MS21_S1,
                            df_transitions_anoxia_MS66_S1,
                            df_transitions_anoxia_MS66_S3,
                            df_transitions_anoxia_64PE_S3,
                            df_transitions_anoxia_64PE_S4,
                            df_transitions_anoxia_64PE_S5,
                            df_transitions_anoxia_64PE_S6,
                            df_transitions_paleoclimate_I,
                            df_transitions_paleoclimate_II,
                            df_transitions_tree_felling_PII,
                            df_transitions_tree_felling_Early_PIII,
                            df_transitions_tree_felling_Late_PIII,
                            ], axis=0, ignore_index=True)

# print results = 打印结果
# print(df_transitions)
df_transitions.to_csv('output_data/df_transitions_all.csv', index=False)



# # ------------------------
# # 02 Load EWS data
# # ------------------------
df_ews_anoxia = pd.read_csv('input_data/ews/anoxia_df_ews.csv')
df_ews_paleoclimate = pd.read_csv('input_data/ews/paleoclimate_df_ews_interpolate.csv')
df_ews_paleoclimate_tree_felling = pd.read_csv('input_data/ews/tree_felling_df_ews.csv')

# # Define column names
columns = ['time', 'state', 'smoothing', 'residuals', 'variance', 'ac1', 'tsid_new']
# Create new DataFrame
df_ews = pd.DataFrame(columns=columns)

# MS21-S1
df_ews_anoxia_MS21_S1 = create_ews_df_anoxia(df_ews_anoxia, 'Mo', 1, 1)

# MS66-S1
df_ews_anoxia_MS66_S1 = create_ews_df_anoxia(df_ews_anoxia, 'Mo', 2, 2)
# MS66-S3
df_ews_anoxia_MS66_S3 = create_ews_df_anoxia(df_ews_anoxia, 'Mo', 4, 3)

# 64PE-S3
df_ews_anoxia_64PE_S3 = create_ews_df_anoxia(df_ews_anoxia, 'Mo', 5, 4)
# 64PE-S4
df_ews_anoxia_64PE_S4 = create_ews_df_anoxia(df_ews_anoxia, 'Mo', 7, 5)
# 64PE-S5
df_ews_anoxia_64PE_S5 = create_ews_df_anoxia(df_ews_anoxia, 'Mo', 9, 6)
# 64PE-S6
df_ews_anoxia_64PE_S6 = create_ews_df_anoxia(df_ews_anoxia, 'Mo', 10, 7)

# End of glaciation I
df_ews_paleoclimate_I = create_ews_df_paleoclimate(df_ews_paleoclimate, 'I', 8)
# End of glaciation II
df_ews_paleoclimate_II = create_ews_df_paleoclimate(df_ews_paleoclimate, 'II', 9)

# P II
df_ews_tree_felling_PII = create_ews_df_tree_felling(df_ews_paleoclimate_tree_felling, 'PII', 10)
# Early P III
df_ews_tree_felling_Early_PIII = create_ews_df_tree_felling(df_ews_paleoclimate_tree_felling, 'EPIII', 11)
# Early P III
df_ews_tree_felling_Late_PIII = create_ews_df_tree_felling(df_ews_paleoclimate_tree_felling, 'LPIII', 12)

# concat df
df_ews = pd.concat([df_ews_anoxia_MS21_S1,
                    df_ews_anoxia_MS66_S1,
                    df_ews_anoxia_MS66_S3,
                    df_ews_anoxia_64PE_S3,
                    df_ews_anoxia_64PE_S4,
                    df_ews_anoxia_64PE_S5,
                    df_ews_anoxia_64PE_S6,
                    df_ews_paleoclimate_I,
                    df_ews_paleoclimate_II,
                    df_ews_tree_felling_PII,
                    df_ews_tree_felling_Early_PIII,
                    df_ews_tree_felling_Late_PIII,
                    ], axis=0, ignore_index=True)

# # Print the DataFrame results
# print(df_ews)
df_ews.to_csv('output_data/df_ews_all.csv', index=False)



# # ------------------------
# # 03 Load SDML prediction data
# # ------------------------

# MS21-S1
df_ml_ms21_s1 = pd.read_csv('input_data/ml_preds/MS21/prediction_probability_surrogate_AAFT_simples_10000_test_S1_CNN_model.csv')

# MS66-S1
df_ml_ms66_s1 = pd.read_csv('input_data/ml_preds/MS66/prediction_probability_surrogate_AAFT_simples_1000_test_S1_SVM_model.csv')
# MS66-S3
df_ml_ms66_s3 = pd.read_csv('input_data/ml_preds/MS66/prediction_probability_surrogate_AAFT_simples_1000_test_S3_SVM_model.csv')

# 64PE-S3
df_ml_64pe_s3 = pd.read_csv('input_data/ml_preds/64PE/prediction_probability_surrogate_AAFT_simples_1000_test_S3_LSTM_model.csv')
# 64PE-S4
df_ml_64pe_s4 = pd.read_csv('input_data/ml_preds/64PE/prediction_probability_surrogate_AAFT_simples_1000_test_S4_LSTM_model.csv')
# 64PE-S5
df_ml_64pe_s5 = pd.read_csv('input_data/ml_preds/64PE/prediction_probability_surrogate_AAFT_simples_10000_test_S5_LSTM_model.csv')
# 64PE-S6
df_ml_64pe_s6 = pd.read_csv('input_data/ml_preds/64PE/prediction_probability_surrogate_AAFT_simples_1000_test_S6_LSTM_model.csv')

# End of glaciation I
df_ml_paleoclimate_I = pd.read_csv('input_data/ml_preds/paleoclimate/prediction_probability_surrogate_AAFT_simples_10000_test_I_SVM_model.csv')
# End of glaciation II
df_ml_paleoclimate_II = pd.read_csv('input_data/ml_preds/paleoclimate/prediction_probability_surrogate_AAFT_simples_1000_test_II_SVM_model.csv')

# PII
df_ml_tree_felling_PII = pd.read_csv('input_data/ml_preds/tree_felling/prediction_probability_surrogate_AAFT_simples_1000_test_PII_three_head_CNN_model.csv')
# Early PII
df_ml_tree_felling_Early_PIII = pd.read_csv('input_data/ml_preds/tree_felling/prediction_probability_surrogate_AAFT_simples_1000_test_EPIII_three_head_CNN_model.csv')
# Late PIII
df_ml_tree_felling_Late_PIII = pd.read_csv('input_data/ml_preds/tree_felling/prediction_probability_surrogate_AAFT_simples_1000_test_LPIII_three_head_CNN_model.csv')

# concat df_ml
df_ml = pd.concat([df_ml_ms21_s1, df_ml_ms66_s1, df_ml_ms66_s3,
               df_ml_64pe_s3, df_ml_64pe_s4, df_ml_64pe_s5,
               df_ml_64pe_s6, df_ml_paleoclimate_I, df_ml_paleoclimate_II,
               df_ml_tree_felling_PII, df_ml_tree_felling_Early_PIII,
               df_ml_tree_felling_Late_PIII], axis=0, ignore_index=True)

# Create new DataFrame
df_ews_ml = pd.DataFrame(df_ews)

# Merging new DataFrames with the original DataFrame
df_ews_ml = df_ews_ml.merge(df_ml, on='Age', how='left', indicator=True, validate='many_to_many')

# # Print the DataFrame results
# print(df_ews_ml)
df_ml.to_csv('output_data/df_sdml_all.csv', index=False)
df_ews_ml.to_csv('output_data/df_ews_sdml_all.csv', index=False)

# # Colour scheme
cols_D3 = px.colors.qualitative.D3              # blue, orange, green, red, purple, brown
cols_plotly = px.colors.qualitative.Plotly      # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray

dic_colours = {
        'state': '#1A6FDF',
        'smoothing': 'gray',
        'variance': cols_D3[1],
        'ac1': cols_D3[2],
        'ml_1': cols_plotly[3],

     }

# dic_colours = {
#         'state': '#1A6FDF',
#         'smoothing': 'gray',
#         'ml_1': '#c64328',
#         'variance': '#eda247',
#         'ac1': '#56bba5',
#
#      }

# dic_colours = {
#         'state': '#1A6FDF',
#         'smoothing': col_grays[2],
#         'variance': '#FB6501',
#         'ac1': '#37AD6B',
#         'ml_1': 'red',
#      }

# dic_colours = {
#         'state': 'blue',
#         # 'smoothing': 'gray',
#         'smoothing': 'black',
#         'variance': 'Orange',
#         'ac1': 'Green',
#         'ml_1': 'red',
#      }

# Scale up factor on image export
scale = 4

def make_grid_figure(tsid_new, x_axis_label, y_axis_label, letter_label, title, df_transitions, df_ews, df_ews_ml):
    # Setting parameters
    linewidth = 2.0
    opacity = 0.5

    #---------------
    # Build figure
    #--------------

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0,
    )

    #----------------
    # Panel 1: Trajectory including transition
    #------------------

    df_transitions_plot = df_transitions.loc[(df_transitions['tsid_new'] == tsid_new)]

    # Trace for trajectory
    fig.add_trace(
        go.Scatter(x=df_transitions_plot['Age'],
                   y=df_transitions_plot['Proxy'],
                   marker_color=dic_colours['state'],
                   showlegend=False,
                   line={'width': linewidth},
                   ),
        row=1,
        col=1,
    )


    # Trace for smoothing
    df_ews_plot = df_ews[(df_ews['tsid_new'] == tsid_new)]

    fig.add_trace(
        go.Scatter(x=df_ews_plot['Age'],
                   y=df_ews_plot['smoothing'],
                   marker_color=dic_colours['smoothing'],
                   showlegend=False,
                   line={'width': linewidth},
                   ),
        row=1,
        col=1,
    )


    #-------------------
    # Panel 2: Variance
    #--------------------

    fig.add_trace(
        go.Scatter(x=df_ews_plot['Age'],
                   y=df_ews_plot['variance'],
                   marker_color=dic_colours['variance'],
                   showlegend=False,
                   line={'width': linewidth},
                   ),
        row=2,
        col=1,
    )


    #-------------------
    # Panel 3: Lag-1 AC
    #--------------------

    fig.add_trace(
        go.Scatter(x=df_ews_plot['Age'],
                   y=df_ews_plot['ac1'],
                   marker_color=dic_colours['ac1'],
                   showlegend=False,
                   line={'width': linewidth},
                   ),
        row=3,
        col=1,
    )


    #-------------------
    # Panel 4: ML weight for 1 transition
    #--------------------
    df_ews_ml_plot = df_ews_ml[(df_ews_ml['tsid_new'] == tsid_new)]
    # print(df_ews_ml_plot)

    df_ews_ml_plot_both = df_ews_ml[(df_ews_ml['tsid_new'] == tsid_new) & (df_ews_ml['_merge'] == 'both')]
    # print('ax=', df_ews_ml_plot_both['Age'].min())

    # Weight for 1 transition
    fig.add_trace(
        go.Scatter(x=df_ews_ml_plot['Age'],
                   y=df_ews_ml_plot['SD_probability_mean'],
                   mode='lines',
                   marker_color=dic_colours['ml_1'],
                   showlegend=False,
                   line={'width': linewidth},
                   ),
        row=4,
        col=1,
    )

    # Upper bound of the error
    fig.add_trace(
        go.Scatter(x=df_ews_ml_plot['Age'],
                   y=df_ews_ml_plot['SD_probability_mean'] + df_ews_ml_plot['SD_probability_error'],
                   mode='lines',
                   fill=None,
                   line=dict(color='rgba(0,0,0,0)'),
                   showlegend=False,
                   ),
        row=4,
        col=1,
    )

    # Convert the color to RGBA for transparency
    # Assuming 20% opacity (alpha = 0.2)
    from matplotlib.colors import to_rgba
    error_band_color_rgba = to_rgba(dic_colours['ml_1'], alpha=0.2)
    # Lower bound of the error, filling between this and the upper bound
    fig.add_trace(
        go.Scatter(x=df_ews_ml_plot['Age'],
                   y=df_ews_ml_plot['SD_probability_mean'] - df_ews_ml_plot['SD_probability_error'],
                   mode='lines',
                   line=dict(color='rgba(0,0,0,0)'),
                   fill='tonexty',
                   fillcolor='rgba' + str(error_band_color_rgba),
                   showlegend=False,
                   ),
        row=4,
        col=1,
    )

    #--------------
    # Add vertical line where transition occurs
    #--------------

    # Add vertical lines where transitions occur
    list_shapes = []

    # df_transitions_plot = df_transitions[df_transitions['Tsid'] == tsid]

    # Get transtiion interval
    t_transition_start = df_transitions_plot['Transition_start'].iloc[0]
    t_transition_end = df_transitions_plot['Transition_end'].iloc[0]

    # #  Make line for start of transition transition
    # shape = {'type': 'line',
    #           'x0': t_transition_start,
    #           'y0': 0,
    #           'x1': t_transition_start,
    #           'y1': 1,
    #           'xref': 'x',
    #           'yref': 'paper',
    #           'line': {'width':2,'dash':'dot'},
    #           }


    #  Make shaded box to show transition
    shape = {'type': 'rect',
             'x0': t_transition_start,
             'y0': 0,
             'x1': t_transition_end,
             'y1': 1,
             'xref': 'x',
             'yref': 'paper',
             'fillcolor': 'gray',
             'opacity': opacity,
             'line_width': 0,
             # 'line': {'width':2,'dash':'dot'},
             }

    # Add shape to list
    list_shapes.append(shape)

    fig['layout'].update(shapes=list_shapes)


    #--------------
    # Add labels and titles
    #----------------------

    list_annotations = []

    # Add label the first subgraph = 给第一个子图添加标签
    label_size = 16
    label_annotation = dict(
            x=0.02,
            y=1,
            text='<b>{}</b>'.format(letter_label),
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(color="black", size=label_size),
    )
    list_annotations.append(label_annotation)

    # # Add label all subgraphs = 给所有子图添加标签
    # label_size = 16
    # axes_numbers = [1, 2, 3, 4]
    # for idx, axis_number in enumerate(axes_numbers):
    #     label_annotation = dict(
    #         x=0.02,  # x 坐标，基于整个图形布局
    #         y=1.0 - (idx * (1.0 / len(axes_numbers))),  # 计算每个子图上方外部的 y 坐标
    #         text="<b>{}{}</b>".format(letter_label, axis_number),  # 标签文本
    #         xref="paper",  # 基于整个图形布局
    #         yref="paper",  # 基于整个图形布局
    #         showarrow=False,
    #         font=dict(color="black", size=label_size),
    #         xanchor='left',     # 保证标签在x坐标的左侧
    #         yanchor='top'       # 保证标签在y坐标的顶部
    #     )
    #     list_annotations.append(label_annotation)

    title_size = 14
    title_annotation = dict(
            # x=sum(xrange)/2,
            x=0.5,
            y=1,
            text=title,
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(color="black", size=title_size),
    )
    list_annotations.append(title_annotation)

    lable_n_size = 12
    # Label for N of data points
    n_label = 'N={}'.format(len(df_ews_plot))
    n_annotation = dict(
            # x=sum(xrange)/2,
            x=0.0,
            y=0.84,
            text=n_label,
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(color="black", size=lable_n_size)
    )
    list_annotations.append(n_annotation)

    # fig['layout'].update(annotations=list_annotations)


    # ---------
    # Arrows to indiciate rolling window
    # ---------

    rw = 0.5
    axes_numbers = [2, 3, 4]
    arrowhead = 3       # 箭头样式
    arrowsize = 1.0     # 箭头大小
    arrowwidth = 1.2    # 箭杆宽度

    for axis_number in axes_numbers:
        # Make left-pointing arrow
        annotation_arrow_left = dict(
            x=df_transitions_plot['Age'].min(),  # arrows' head
            #x=-20.48787,
            y=0.05,  # arrows' head

            ax=df_ews_ml_plot_both['Age'].min(),
            # ax=df_transitions_plot['Age'].min() + ((t_transition_start - df_transitions_plot['Age'].min()) * rw),  # arrows' tail
            # ax=-15.493935,
            ay=0.05,  # arrows' tail
            xref="x{}".format(axis_number),
            yref="y{} domain".format(axis_number),
            axref="x{}".format(axis_number),
            ayref="y{} domain".format(axis_number),
            text="",  # if you want only the arrow
            showarrow=True,
            arrowhead=arrowhead,
            arrowsize=arrowsize,
            arrowwidth=arrowwidth,
            arrowcolor="black",
        )
        list_annotations.append(annotation_arrow_left)

        # Make right-pointing arrow
        annotation_arrow_right = dict(
            ax=df_transitions_plot['Age'].min(),  # arrows' head
            # ax=-20.48787,
            ay=0.05,  # arrows' tail

            x=df_ews_ml_plot_both['Age'].min(),
            # x=df_transitions_plot['Age'].min() + ((t_transition_start - df_transitions_plot['Age'].min()) * rw),  # arrows' tail
            # x=-15.493935,
            y=0.05,  # arrows' head
            xref="x{}".format(axis_number),
            yref="y{} domain".format(axis_number),
            axref="x{}".format(axis_number),
            ayref="y{} domain".format(axis_number),
            text="",  # if you want only the arrow
            showarrow=True,
            arrowhead=arrowhead,
            arrowsize=arrowsize,
            arrowwidth=arrowwidth,
            arrowcolor="black",
        )
        # Append to annotations
        list_annotations.append(annotation_arrow_right)

    fig['layout'].update(annotations=list_annotations)


    # -------
    # Axes properties
    # ---------

    Layout_linewidth = 1.0
    # Layout properties
    fig.update_xaxes(title={'text': '{}'.format(x_axis_label), 'standoff': 5},
                     ticks="outside",
                     showline=True,
                     linewidth=Layout_linewidth,
                     linecolor='black',
                     mirror=True,
                     row=4,
                     col=1,
                     )

    # Global y axis properties
    fig.update_yaxes(showline=True,
                     ticks="outside",
                     linecolor='black',
                     mirror=True,
                     showgrid=False,
                     automargin=False,
                     )


    # Global x axis properties
    fig.update_xaxes(showline=True,
                     linecolor='black',
                     mirror=False,
                     showgrid=False,
                     tickangle=0,
                     tickmode='auto',
                     nticks=4,
                     )
    fig.update_xaxes(mirror=True, row=1, col=1)

    fig.update_yaxes(title={'text': '{}'.format(y_axis_label), 'standoff': 50}, row=1, col=1)
    fig.update_yaxes(title={'text': 'Variance', 'standoff': 50}, row=2, col=1)
    fig.update_yaxes(title={'text':  'Lag-1 AC', 'standoff': 50}, tickformat=".2f", row=3, col=1)
    fig.update_yaxes(tickmode='auto', nticks=4, row=3, col=1)
    fig.update_yaxes(title={'text': 'SDML probability', 'standoff': 50}, range=[-0.05, 1.07], tickformat=".2f", row=4, col=1)

    layout_size = 11
    fig.update_layout(height=400,
                      width=200,
                      margin={'l': 50, 'r': 10, 'b': 20, 't': 10},
                      font=dict(size=layout_size, family='Times New Roman'),
                      paper_bgcolor='rgba(255,255,255,1)',
                      plot_bgcolor='rgba(255,255,255,1)'
                      )

    fig.update_traces(mode="lines")

    return fig



# # make single fig
# fig = make_grid_figure(1, 'Mo','a','S1')
# fig.write_html('temp.html')
# # fig.write_image('temp.png',scale=2)


#---------- Loop over all tsid-------

import string
list_letter_labels = string.ascii_lowercase[:14]


# Make ind figures for Mo

i = 0
list_tsid = np.arange(1, 13)
for tsid_new in list_tsid:
    # Make figure
    letter_label = list_letter_labels[i]
    i += 1

    # Get sapropel ID and core name
    x_axis_label = df_transitions[df_transitions['tsid_new'] == tsid_new]['X_axis_label'].iloc[0]
    y_axis_label = df_transitions[df_transitions['tsid_new'] == tsid_new]['Y_axis_label'].iloc[0]
    record = df_transitions[df_transitions['tsid_new'] == tsid_new]['Record'].iloc[0]
    title = '{}'.format(record)

    # fig = make_grid_figure(tsid_new, 'Mo', letter_label, title)
    # fig = make_grid_figure(tsid_new, var_label, df_transitions, df_ews, merged_df_ml, letter_label, title)
    fig = make_grid_figure(tsid_new, x_axis_label, y_axis_label, letter_label, title, df_transitions, df_ews, df_ews_ml)

    # Export as png
    fig.write_image('output_figures/single_figs/img_{}.png'.format(tsid_new), scale=scale)
    print('Exported for tsid = {}'.format(tsid_new))


# -------------------------------
# # Combine plots into single png
# -------------------------------


# Import PIL for image tools
from PIL import Image
import os
import numpy as np

# –------------
# Combine plots for anoxia Mo forced
# –--------------

filepath = 'output_figures/single_figs/'

list_img = []
list_tsid = np.arange(1, 13)

for tsid_new in list_tsid:
    # print('tsid', tsid)
    img = Image.open(filepath + 'img_{}.png'.format(tsid_new))
    list_img.append(img)

# Get heght and width of individlau panels
ind_height = list_img[0].height
ind_width = list_img[0].width
# Creat frame
dst = Image.new('RGB', (6 * ind_width, 2 * ind_height), (255, 255, 255))

# Pasete in images
dst.paste(list_img[0], (0, 0))
dst.paste(list_img[1], (1 * ind_width, 0))
dst.paste(list_img[2], (2 * ind_width, 0))
dst.paste(list_img[3], (3 * ind_width, 0))
dst.paste(list_img[4], (4 * ind_width, 0))
dst.paste(list_img[5], (5 * ind_width, 0))

dst.paste(list_img[6], (0, ind_height))
dst.paste(list_img[7], (1 * ind_width, ind_height))
dst.paste(list_img[8], (2 * ind_width, ind_height))
dst.paste(list_img[9], (3 * ind_width, ind_height))
dst.paste(list_img[10], (4 * ind_width, ind_height))
dst.paste(list_img[11], (5 * ind_width, ind_height))

dst.save(filepath + '../figure_3_aaft.png')


# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print("Ran in {:.2f}s".format(time_taken))

print("--------- 03 Successful Make Figure 3---------")