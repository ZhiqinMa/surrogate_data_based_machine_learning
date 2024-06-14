# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 19:03:01 2021

Make fig of ROC curves with inset showing histogram of highest DL probability

@author: Thoams M. Bury

"""

import os
import time
start_time = time.time()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt

# Import PIL for image tools
from PIL import Image


try:
    os.mkdir('output_figures')
except:
    print('output_figures directory already exists!\n')



# -----------
# General fig params
# ------------

# Colour scheme
cols_D3 = px.colors.qualitative.D3                  # blue, orange, green, red, purple, brown
cols_plotly = px.colors.qualitative.Plotly          # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray

dic_colours = {
    "state": "gray",
    "smoothing": col_grays[2],
    "RP": cols_plotly[3],
    "FT": cols_D3[0],
    "AAFT": cols_D3[1],
    "IAAFT1": cols_D3[2],
    "IAAFT2": cols_D3[3],
}

# dic_colours = {
#     "state": "gray",
#     "smoothing": col_grays[2],
#     "SDML": cols_D3[4],
#     "variance": cols_D3[1],
#     "ac": cols_D3[2],
# }


# Pixels to mm
mm_to_pixel = 96 / 25.4  # 96 dpi, 25.4mm in an inch

# Nature width of single col fig : 89mm
# Nature width of double col fig : 183mm

# Get width of single panel in pixels
fig_width = 183 * mm_to_pixel / 3  # 3 panels wide
fig_height = fig_width


font_size = 10
font_family = "Times New Roman"
font_size_letter_label = 14
font_size_auc_text = 10

# AUC annotations
x_auc = 0.98
# y_auc = 0.6
y_auc = 0.65
x_N = 0.18
y_N = 0.05
# y_auc_sep = 0.065
y_auc_sep = 0.060

# linewidth = 0.7
linewidth = 1.4
linewidth_axes = 0.5
tickwidth = 0.5
linewidth_axes_inset = 0.5

axes_standoff = 0


# Scale up factor on image export
scale = 8  # default dpi=72 - nature=300-600

def make_roc_figure(df_roc, letter_label, title="", text_N=""):
    """Make ROC figure (no inset)"""

    fig = go.Figure()

    # plot SDML prediction for RP
    df_trace = df_roc[df_roc["surr_type"] == "RP"]
    auc_rp = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["RP"],
            ),
        )
    )

    # plot SDML prediction for FT
    df_trace = df_roc[df_roc["surr_type"] == "FT"]
    auc_ft = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["FT"],
            ),
        )
    )

    # plot SDML prediction for AAFT
    df_trace = df_roc[df_roc["surr_type"] == "AAFT"]
    auc_aaft = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["AAFT"],
            ),
        )
    )

    # plot SDML prediction for IAAFT1
    df_trace = df_roc[df_roc["surr_type"] == "IAAFT1"]
    auc_iaaft1 = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["IAAFT1"],
            ),
        )
    )

    # plot SDML prediction for IAAFT2
    df_trace = df_roc[df_roc["surr_type"] == "IAAFT2"]
    auc_iaaft2 = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["IAAFT2"],
            ),
        )
    )

    # Line y=x
    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, 1, 100),
            y=np.linspace(0, 1, 100),
            showlegend=False,
            line=dict(
                color="black",
                dash="dot",
                width=linewidth,
            ),
        )
    )

    # --------------
    # Add labels and titles
    # ----------------------

    list_annotations = []

    label_annotation = dict(
        # x=sum(xrange)/2,
        x=0.02,
        y=1.05,
        text="<b>{}</b>".format(letter_label),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color="black",
            size=font_size_letter_label,
        ),
    )

    annotation_auc_rp = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc,
        text="A<sub>rp</sub>={:.2f}".format(auc_rp),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["RP"],
            size=font_size_auc_text,
        ),
    )

    annotation_auc_ft = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc - y_auc_sep,
        text="A<sub>ft</sub>={:.2f}".format(auc_ft),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["FT"],
            size=font_size_auc_text,
        ),
    )

    annotation_auc_aaft = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc - (2 * y_auc_sep),
        text="A<sub>aaft</sub>={:.2f}".format(auc_aaft),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["AAFT"],
            size=font_size_auc_text,
        ),
    )

    annotation_auc_iaaft1 = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc - (3 * y_auc_sep),
        text="A<sub>iaaft1</sub>={:.2f}".format(auc_iaaft1),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["IAAFT1"],
            size=font_size_auc_text,
        ),
    )

    annotation_auc_iaaft2 = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc - (4 * y_auc_sep),
        text="A<sub>iaaft2</sub>={:.2f}".format(auc_iaaft2),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["IAAFT2"],
            size=font_size_auc_text,
        ),
    )

    annotation_N = dict(
        # x=sum(xrange)/2,
        x=x_N,
        y=y_N,
        text=text_N,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color="black",
            size=font_size_auc_text,
        ),
    )
    title_annotation = dict(
        # x=sum(xrange)/2,
        x=0.5,
        y=1,
        text=title,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color="black", size=font_size),
    )

    list_annotations.append(label_annotation)
    list_annotations.append(annotation_auc_rp)
    list_annotations.append(annotation_auc_ft)
    list_annotations.append(annotation_auc_aaft)
    list_annotations.append(annotation_auc_iaaft1)
    list_annotations.append(annotation_auc_iaaft2)
    list_annotations.append(annotation_N)
    # list_annotations.append(title_annotation)

    fig["layout"].update(annotations=list_annotations)

    # -------------
    # General layout properties
    # --------------

    # X axes properties
    fig.update_xaxes(
        title=dict(
            text="False positive",
            standoff=axes_standoff,
        ),
        range=[-0.04, 1.04],
        ticks="outside",
        tickwidth=tickwidth,
        tickvals=np.arange(0, 1.1, 0.2),
        showline=True,
        linewidth=linewidth_axes,
        linecolor="black",
        mirror=False,
    )

    # Y axes properties
    fig.update_yaxes(
        title=dict(
            text="True positive",
            standoff=axes_standoff,
        ),
        range=[-0.04, 1.04],
        ticks="outside",
        tickvals=np.arange(0, 1.1, 0.2),
        tickwidth=tickwidth,
        showline=True,
        linewidth=linewidth_axes,
        linecolor="black",
        mirror=False,
    )

    # Overall properties
    fig.update_layout(
        legend=dict(x=0.6, y=0),
        width=fig_width,
        height=fig_height,
        margin=dict(l=30, r=5, b=15, t=5),
        font=dict(size=font_size, family=font_family),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
    )

    return fig


import seaborn as sns

def counts_forced(df_roc, df_predict_forced):
    # set columus
    labels = ["Transition", "Neutral"]

    # get SDML probability column names
    columu_SDML_probability = df_predict_forced.columns[0]
    # print(columu_SDML_probability)

    # Choose the best threshold = 选择最佳阈值
    optimal_idx = np.argmax(df_roc['tpr'] - df_roc['fpr'])
    optimal_threshold = df_roc['thresholds'][optimal_idx]
    # print('optimal_threshold=', optimal_threshold)

    # Use conditionals and categorize = 使用条件语句判断并归类
    df_predict_forced['type'] = df_predict_forced[columu_SDML_probability].apply(lambda x: labels[0] if x >= optimal_threshold else labels[1])

    # Count the number of 'Forced' and 'Neutral'
    counts = df_predict_forced['type'].value_counts()
    # print(counts)

    # Ensure both labels are in the counts, default to 0 if not present
    forced_count = counts.get('Transition', 0)
    Neutral_count = counts.get('Neutral', 0)

    # # Print counts
    # print(f"Count of Forced: {forced_count}")
    # print(f"Count of Neutral: {Neutral_count}")

    # If you need to calculate and use the probabilities
    total_count = len(df_predict_forced)
    forced_probability = forced_count / total_count
    Neutral_probability = Neutral_count / total_count

    # create DataFrame
    data = {
        "sdml_prob": [forced_probability, Neutral_probability],
        "label": labels
    }
    df_counts = pd.DataFrame(data)
    return df_counts


def make_inset_barplot(df_plot, target_bif, save_dir):
    """
    Make inset boxplot that shows the value of the
    DL weights where the predictions are made

    """

    sns.set(
        style="ticks",
        rc={
            "figure.figsize": (2.5 * 1.05, 1.5 * 1.05),
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#333F4B",
            "xtick.color": "#333F4B",
            "xtick.major.width": 1.0,
            "xtick.major.size": 3,
            "text.color": "#333F4B",
            "font.family": "Times New Roman",
            # 'font.size':20,
        },
        font_scale=1.2,
    )

    plt.figure()
    # set columus
    labels = ["Transition", "Neutral"]

    color_main = "#A9A9A9"
    color_target = "#FFA15A"
    col_palette = {bif: color_main for bif in labels}
    col_palette[target_bif] = color_target

    # Calculate mean for each label
    df_plot = df_plot.groupby('label')['sdml_prob'].mean().reset_index()

    # Sort by the value of the 'sdml_prob' column in descending order
    df_plot = df_plot.sort_values(by="sdml_prob", ascending=False)

    b = sns.barplot(
        df_plot,
        orient="h",
        x="sdml_prob",
        y="label",
        palette=col_palette,
        linewidth=0.8,
        ci=None,
    )

    for i, value in enumerate(df_plot['sdml_prob']):
        if value > 0.8:
            b.text(value, i, f'{value:.2f}', ha='right', va='center')
        else:
            b.text(value, i, f'{value:.2f}', ha='left', va='center')


    # Set visual elements of the plot
    b.set(xlabel=None)
    b.set(ylabel=None)
    b.set_xticks([0, 0.5, 1])
    b.set_xticklabels(["0", "0.5", "1"])
    # sns.despine(offset=0, trim=False)             # 移除上和右边框
    b.tick_params(left=False, bottom=True)          # 设置刻度样式

    # Save the figure
    fig = b.get_figure()
    fig.savefig(save_dir, dpi=330, bbox_inches="tight", pad_inches=0)


def combine_roc_inset(path_roc, path_inset, path_out):
    """
    Combine ROC plot and inset, and export to path_out
    """

    # Import image
    img_roc = Image.open(path_roc)
    img_inset = Image.open(path_inset)

    # Get height and width of frame (in pixels)
    height = img_roc.height
    width = img_roc.width

    # Create frame
    dst = Image.new("RGB", (width, height), (255, 255, 255))

    # Pasete in images
    dst.paste(img_roc, (0, 0))
    # dst.paste(img_inset, (width - img_inset.width - 60, 1050))
    dst.paste(img_inset, (width - img_inset.width - 30, 1050))

    dpi = 96 * scale  # (default dpi) * (scaling factor)
    dst.save(path_out, dpi=(dpi, dpi))

    return



# -------
# 01 chick heart data
# --------
folder = "01_chick_heart"
simples = "10000"
model = "three_head_CNN_model"
ID_test = '[14]'

surr_rp = 'RP'
surr_ft = 'FT'
surr_aaft = 'AAFT'
surr_iaaft1 = 'IAAFT1'
surr_iaaft2 = 'IAAFT2'

# load roc DataFrame data
df_roc_rp = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_rp}/ROC/df_roc_surrogate_{surr_rp}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_rp['surr_type'] = surr_rp

df_roc_ft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_ft}/ROC/df_roc_surrogate_{surr_ft}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_ft['surr_type'] = surr_ft

df_roc_aaft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_aaft}/ROC/df_roc_surrogate_{surr_aaft}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_aaft['surr_type'] = surr_aaft

df_roc_iaaft1 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft1}/ROC/df_roc_surrogate_{surr_iaaft1}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_iaaft1['surr_type'] = surr_iaaft1

df_roc_iaaft2 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft2}/ROC/df_roc_surrogate_{surr_iaaft2}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_iaaft2['surr_type'] = surr_iaaft2

# Concat DataFrame by row
df_roc = pd.concat([df_roc_rp, df_roc_ft, df_roc_aaft, df_roc_iaaft1, df_roc_iaaft2], axis=0, ignore_index=True)
# # Print the DataFrame results
# print(df_roc)
# df_roc.to_csv('df_roc_all.csv', index=False)

# # load SDML predict forced DataFrame data
df_predict_forced_rp = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_rp}/df_forced_fixed_surrogate_{surr_rp}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_rp = counts_forced(df_roc_rp, df_predict_forced_rp)

df_predict_forced_ft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_ft}/df_forced_fixed_surrogate_{surr_ft}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_ft = counts_forced(df_roc_ft, df_predict_forced_ft)

df_predict_forced_aaft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_aaft}/df_forced_fixed_surrogate_{surr_aaft}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_aaft = counts_forced(df_roc_aaft, df_predict_forced_aaft)

df_predict_forced_iaaft1 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft1}/df_forced_fixed_surrogate_{surr_iaaft1}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_iaaft1 = counts_forced(df_roc_iaaft1, df_predict_forced_iaaft1)

df_predict_forced_iaaft2 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft2}/df_forced_fixed_surrogate_{surr_iaaft2}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_iaaft2 = counts_forced(df_roc_iaaft2, df_predict_forced_iaaft2)

# Concat DataFrame by row
df_plot = pd.concat([df_plot_rp, df_plot_ft, df_plot_aaft, df_plot_iaaft1, df_plot_iaaft2], axis=0, ignore_index=True)
# print(df_plot)

fig_roc = make_roc_figure(df_roc, "a", text_N="N={}".format(len(df_predict_forced_rp) * 2))
fig_roc.write_image("temp_roc.png", scale=scale)

make_inset_barplot(df_plot, "Transition", "temp_inset.png")

# Combine figs and export
path_roc = "temp_roc.png"
path_inset = "temp_inset.png"
path_out = "output_figures/roc_heart.png"

combine_roc_inset(path_roc, path_inset, path_out)


# -------
# 02 MS21 data
# --------
folder = "02_MS21"
simples = "10000"
model = "CNN_model"
ID_test = '[1]'

surr_rp = 'RP'
surr_ft = 'FT'
surr_aaft = 'AAFT'
surr_iaaft1 = 'IAAFT1'
surr_iaaft2 = 'IAAFT2'

# load roc DataFrame data
df_roc_rp = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_rp}/ROC/df_roc_surrogate_{surr_rp}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_rp['surr_type'] = surr_rp

df_roc_ft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_ft}/ROC/df_roc_surrogate_{surr_ft}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_ft['surr_type'] = surr_ft

df_roc_aaft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_aaft}/ROC/df_roc_surrogate_{surr_aaft}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_aaft['surr_type'] = surr_aaft

df_roc_iaaft1 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft1}/ROC/df_roc_surrogate_{surr_iaaft1}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_iaaft1['surr_type'] = surr_iaaft1

df_roc_iaaft2 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft2}/ROC/df_roc_surrogate_{surr_iaaft2}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_iaaft2['surr_type'] = surr_iaaft2

# Concat DataFrame by row
df_roc = pd.concat([df_roc_rp, df_roc_ft, df_roc_aaft, df_roc_iaaft1, df_roc_iaaft2], axis=0, ignore_index=True)
# # Print the DataFrame results
# print(df_roc)
# df_roc.to_csv('df_roc_all.csv', index=False)

# # load SDML predict forced DataFrame data
df_predict_forced_rp = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_rp}/df_forced_fixed_surrogate_{surr_rp}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_rp = counts_forced(df_roc_rp, df_predict_forced_rp)

df_predict_forced_ft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_ft}/df_forced_fixed_surrogate_{surr_ft}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_ft = counts_forced(df_roc_ft, df_predict_forced_ft)

df_predict_forced_aaft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_aaft}/df_forced_fixed_surrogate_{surr_aaft}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_aaft = counts_forced(df_roc_aaft, df_predict_forced_aaft)

df_predict_forced_iaaft1 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft1}/df_forced_fixed_surrogate_{surr_iaaft1}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_iaaft1 = counts_forced(df_roc_iaaft1, df_predict_forced_iaaft1)

df_predict_forced_iaaft2 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft2}/df_forced_fixed_surrogate_{surr_iaaft2}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_iaaft2 = counts_forced(df_roc_iaaft2, df_predict_forced_iaaft2)

# Concat DataFrame by row
df_plot = pd.concat([df_plot_rp, df_plot_ft, df_plot_aaft, df_plot_iaaft1, df_plot_iaaft2], axis=0, ignore_index=True)
# print(df_plot)

fig_roc = make_roc_figure(df_roc, "b", text_N="N={}".format(len(df_predict_forced_rp) * 2))
fig_roc.write_image("temp_roc.png", scale=scale)

make_inset_barplot(df_plot, "Transition", "temp_inset.png")

# Combine figs and export
path_roc = "temp_roc.png"
path_inset = "temp_inset.png"
path_out = "output_figures/roc_ms21.png"

combine_roc_inset(path_roc, path_inset, path_out)


# -------
# 03 MS66 data
# --------
folder = "03_MS66"
simples = "1000"
model = "SVM_model"
ID_test = '[4,2]'

surr_rp = 'RP'
surr_ft = 'FT'
surr_aaft = 'AAFT'
surr_iaaft1 = 'IAAFT1'
surr_iaaft2 = 'IAAFT2'

# load roc DataFrame data
df_roc_rp = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_rp}/ROC/df_roc_surrogate_{surr_rp}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_rp['surr_type'] = surr_rp

df_roc_ft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_ft}/ROC/df_roc_surrogate_{surr_ft}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_ft['surr_type'] = surr_ft

df_roc_aaft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_aaft}/ROC/df_roc_surrogate_{surr_aaft}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_aaft['surr_type'] = surr_aaft

df_roc_iaaft1 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft1}/ROC/df_roc_surrogate_{surr_iaaft1}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_iaaft1['surr_type'] = surr_iaaft1

df_roc_iaaft2 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft2}/ROC/df_roc_surrogate_{surr_iaaft2}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_iaaft2['surr_type'] = surr_iaaft2

# Concat DataFrame by row
df_roc = pd.concat([df_roc_rp, df_roc_ft, df_roc_aaft, df_roc_iaaft1, df_roc_iaaft2], axis=0, ignore_index=True)
# # Print the DataFrame results
# print(df_roc)
# df_roc.to_csv('df_roc_all.csv', index=False)

# # load SDML predict forced DataFrame data
df_predict_forced_rp = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_rp}/df_forced_fixed_surrogate_{surr_rp}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_rp = counts_forced(df_roc_rp, df_predict_forced_rp)

df_predict_forced_ft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_ft}/df_forced_fixed_surrogate_{surr_ft}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_ft = counts_forced(df_roc_ft, df_predict_forced_ft)

df_predict_forced_aaft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_aaft}/df_forced_fixed_surrogate_{surr_aaft}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_aaft = counts_forced(df_roc_aaft, df_predict_forced_aaft)

df_predict_forced_iaaft1 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft1}/df_forced_fixed_surrogate_{surr_iaaft1}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_iaaft1 = counts_forced(df_roc_iaaft1, df_predict_forced_iaaft1)

df_predict_forced_iaaft2 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft2}/df_forced_fixed_surrogate_{surr_iaaft2}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_iaaft2 = counts_forced(df_roc_iaaft2, df_predict_forced_iaaft2)

# Concat DataFrame by row
df_plot = pd.concat([df_plot_rp, df_plot_ft, df_plot_aaft, df_plot_iaaft1, df_plot_iaaft2], axis=0, ignore_index=True)
# print(df_plot)

fig_roc = make_roc_figure(df_roc, "c", text_N="N={}".format(len(df_predict_forced_rp) * 2))
fig_roc.write_image("temp_roc.png", scale=scale)

make_inset_barplot(df_plot, "Transition", "temp_inset.png")

# Combine figs and export
path_roc = "temp_roc.png"
path_inset = "temp_inset.png"
path_out = "output_figures/roc_ms66.png"

combine_roc_inset(path_roc, path_inset, path_out)


# -------
# 04 64PE data
# --------
folder = "04_64PE"
simples = "10000"
model = "LSTM_model"
ID_test = '[10,9,7,5]'

surr_rp = 'RP'
surr_ft = 'FT'
surr_aaft = 'AAFT'
surr_iaaft1 = 'IAAFT1'
surr_iaaft2 = 'IAAFT2'

# load roc DataFrame data
df_roc_rp = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_rp}/ROC/df_roc_surrogate_{surr_rp}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_rp['surr_type'] = surr_rp

df_roc_ft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_ft}/ROC/df_roc_surrogate_{surr_ft}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_ft['surr_type'] = surr_ft

df_roc_aaft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_aaft}/ROC/df_roc_surrogate_{surr_aaft}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_aaft['surr_type'] = surr_aaft

df_roc_iaaft1 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft1}/ROC/df_roc_surrogate_{surr_iaaft1}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_iaaft1['surr_type'] = surr_iaaft1

df_roc_iaaft2 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft2}/ROC/df_roc_surrogate_{surr_iaaft2}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_iaaft2['surr_type'] = surr_iaaft2

# Concat DataFrame by row
df_roc = pd.concat([df_roc_rp, df_roc_ft, df_roc_aaft, df_roc_iaaft1, df_roc_iaaft2], axis=0, ignore_index=True)
# # Print the DataFrame results
# print(df_roc)
# df_roc.to_csv('df_roc_all.csv', index=False)

# # load SDML predict forced DataFrame data
df_predict_forced_rp = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_rp}/df_forced_fixed_surrogate_{surr_rp}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_rp = counts_forced(df_roc_rp, df_predict_forced_rp)

df_predict_forced_ft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_ft}/df_forced_fixed_surrogate_{surr_ft}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_ft = counts_forced(df_roc_ft, df_predict_forced_ft)

df_predict_forced_aaft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_aaft}/df_forced_fixed_surrogate_{surr_aaft}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_aaft = counts_forced(df_roc_aaft, df_predict_forced_aaft)

df_predict_forced_iaaft1 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft1}/df_forced_fixed_surrogate_{surr_iaaft1}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_iaaft1 = counts_forced(df_roc_iaaft1, df_predict_forced_iaaft1)

df_predict_forced_iaaft2 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft2}/df_forced_fixed_surrogate_{surr_iaaft2}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_iaaft2 = counts_forced(df_roc_iaaft2, df_predict_forced_iaaft2)

# Concat DataFrame by row
df_plot = pd.concat([df_plot_rp, df_plot_ft, df_plot_aaft, df_plot_iaaft1, df_plot_iaaft2], axis=0, ignore_index=True)
# print(df_plot)

fig_roc = make_roc_figure(df_roc, "d", text_N="N={}".format(len(df_predict_forced_rp) * 2))
fig_roc.write_image("temp_roc.png", scale=scale)

make_inset_barplot(df_plot, "Transition", "temp_inset.png")

# Combine figs and export
path_roc = "temp_roc.png"
path_inset = "temp_inset.png"
path_out = "output_figures/roc_64pe.png"

combine_roc_inset(path_roc, path_inset, path_out)



# -------
# 05 paleoclimate data
# --------
folder = "05_paleoclimate"
simples = "10000"
model = "SVM_model"
ID_test = '[2,1]'

surr_rp = 'RP'
surr_ft = 'FT'
surr_aaft = 'AAFT'
surr_iaaft1 = 'IAAFT1'
surr_iaaft2 = 'IAAFT2'

# load roc DataFrame data
df_roc_rp = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_rp}/ROC/df_roc_surrogate_{surr_rp}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_rp['surr_type'] = surr_rp

df_roc_ft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_ft}/ROC/df_roc_surrogate_{surr_ft}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_ft['surr_type'] = surr_ft

df_roc_aaft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_aaft}/ROC/df_roc_surrogate_{surr_aaft}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_aaft['surr_type'] = surr_aaft

df_roc_iaaft1 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft1}/ROC/df_roc_surrogate_{surr_iaaft1}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_iaaft1['surr_type'] = surr_iaaft1

df_roc_iaaft2 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft2}/ROC/df_roc_surrogate_{surr_iaaft2}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_iaaft2['surr_type'] = surr_iaaft2

# Concat DataFrame by row
df_roc = pd.concat([df_roc_rp, df_roc_ft, df_roc_aaft, df_roc_iaaft1, df_roc_iaaft2], axis=0, ignore_index=True)
# # Print the DataFrame results
# print(df_roc)
# df_roc.to_csv('df_roc_all.csv', index=False)

# # load SDML predict forced DataFrame data
df_predict_forced_rp = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_rp}/df_forced_fixed_surrogate_{surr_rp}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_rp = counts_forced(df_roc_rp, df_predict_forced_rp)

df_predict_forced_ft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_ft}/df_forced_fixed_surrogate_{surr_ft}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_ft = counts_forced(df_roc_ft, df_predict_forced_ft)

df_predict_forced_aaft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_aaft}/df_forced_fixed_surrogate_{surr_aaft}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_aaft = counts_forced(df_roc_aaft, df_predict_forced_aaft)

df_predict_forced_iaaft1 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft1}/df_forced_fixed_surrogate_{surr_iaaft1}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_iaaft1 = counts_forced(df_roc_iaaft1, df_predict_forced_iaaft1)

df_predict_forced_iaaft2 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft2}/df_forced_fixed_surrogate_{surr_iaaft2}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_iaaft2 = counts_forced(df_roc_iaaft2, df_predict_forced_iaaft2)

# Concat DataFrame by row
df_plot = pd.concat([df_plot_rp, df_plot_ft, df_plot_aaft, df_plot_iaaft1, df_plot_iaaft2], axis=0, ignore_index=True)
# print(df_plot)

fig_roc = make_roc_figure(df_roc, "e", text_N="N={}".format(len(df_predict_forced_rp) * 2))
fig_roc.write_image("temp_roc.png", scale=scale)

make_inset_barplot(df_plot, "Transition", "temp_inset.png")

# Combine figs and export
path_roc = "temp_roc.png"
path_inset = "temp_inset.png"
path_out = "output_figures/roc_paleoclimate.png"

combine_roc_inset(path_roc, path_inset, path_out)


# -------
# 06 tree_felling data
# --------
folder = "06_tree_felling"
simples = "10000"
model = "CNN_LSTM_model"
ID_test = '[2,3,4]'

surr_rp = 'RP'
surr_ft = 'FT'
surr_aaft = 'AAFT'
surr_iaaft1 = 'IAAFT1'
surr_iaaft2 = 'IAAFT2'

# load roc DataFrame data
df_roc_rp = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_rp}/ROC/df_roc_surrogate_{surr_rp}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_rp['surr_type'] = surr_rp

df_roc_ft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_ft}/ROC/df_roc_surrogate_{surr_ft}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_ft['surr_type'] = surr_ft

df_roc_aaft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_aaft}/ROC/df_roc_surrogate_{surr_aaft}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_aaft['surr_type'] = surr_aaft

df_roc_iaaft1 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft1}/ROC/df_roc_surrogate_{surr_iaaft1}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_iaaft1['surr_type'] = surr_iaaft1

df_roc_iaaft2 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft2}/ROC/df_roc_surrogate_{surr_iaaft2}_simples_{simples}_{model}_{ID_test}.csv")
df_roc_iaaft2['surr_type'] = surr_iaaft2

# Concat DataFrame by row
df_roc = pd.concat([df_roc_rp, df_roc_ft, df_roc_aaft, df_roc_iaaft1, df_roc_iaaft2], axis=0, ignore_index=True)
# # Print the DataFrame results
# print(df_roc)
# df_roc.to_csv('df_roc_all.csv', index=False)

# # load SDML predict forced DataFrame data
df_predict_forced_rp = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_rp}/df_forced_fixed_surrogate_{surr_rp}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_rp = counts_forced(df_roc_rp, df_predict_forced_rp)

df_predict_forced_ft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_ft}/df_forced_fixed_surrogate_{surr_ft}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_ft = counts_forced(df_roc_ft, df_predict_forced_ft)

df_predict_forced_aaft = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_aaft}/df_forced_fixed_surrogate_{surr_aaft}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_aaft = counts_forced(df_roc_aaft, df_predict_forced_aaft)

df_predict_forced_iaaft1 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft1}/df_forced_fixed_surrogate_{surr_iaaft1}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_iaaft1 = counts_forced(df_roc_iaaft1, df_predict_forced_iaaft1)

df_predict_forced_iaaft2 = pd.read_csv(f"input_data/{folder}/07_roc_data_and_figures_sdml/data/{surr_iaaft2}/df_forced_fixed_surrogate_{surr_iaaft2}_simples_{simples}_{model}_{ID_test}.csv")
df_plot_iaaft2 = counts_forced(df_roc_iaaft2, df_predict_forced_iaaft2)

# Concat DataFrame by row
df_plot = pd.concat([df_plot_rp, df_plot_ft, df_plot_aaft, df_plot_iaaft1, df_plot_iaaft2], axis=0, ignore_index=True)
# print(df_plot)

fig_roc = make_roc_figure(df_roc, "f", text_N="N={}".format(len(df_predict_forced_rp) * 2))
fig_roc.write_image("temp_roc.png", scale=scale)

make_inset_barplot(df_plot, "Transition", "temp_inset.png")

# Combine figs and export
path_roc = "temp_roc.png"
path_inset = "temp_inset.png"
path_out = "output_figures/roc_tree_felling.png"

combine_roc_inset(path_roc, path_inset, path_out)



# ------------
# Combine ROC plots
# ------------

# -----------------
# Fig 4 of manuscript: 8-panel figure for all models and empirical data
# -----------------

# # Early or late predictions
# timing = 'late'

list_filenames = [
    "roc_heart",
    "roc_ms21",
    "roc_ms66",
    "roc_64pe",
    "roc_paleoclimate",
    "roc_tree_felling",

]
list_filenames = ["output_figures/{}.png".format(s) for s in list_filenames]

list_img = []
for filename in list_filenames:
    img = Image.open(filename)
    list_img.append(img)

# Get heght and width of individual panels
ind_height = list_img[0].height
ind_width = list_img[0].width


# Create frame
dst = Image.new("RGB", (3 * ind_width, 2 * ind_height), (255, 255, 255))

# Paste in images
i = 0
for y in np.arange(2) * ind_height:
    for x in np.arange(3) * ind_width:
        dst.paste(list_img[i], (x, y))
        i += 1


dpi = 96 * 8  # (default dpi) * (scaling factor)
dst.save("output_figures/figure_s7.png", dpi=(dpi, dpi))

# Remove temporary images
import os

for filename in list_filenames + ["temp_inset.png", "temp_roc.png"]:
    try:
        os.remove(filename)
    except:
        pass

# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print("Ran in {:.2f}s".format(time_taken))


print("--------- 04 Successful Make Figure 4 ROC curves ---------")