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
    "variance": cols_D3[1],
    "ac": cols_D3[2],
    "SDML": cols_plotly[3],
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
y_auc = 0.6
x_N = 0.18
y_N = 0.05
y_auc_sep = 0.065

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

    # DL prediction any bif
    df_trace = df_roc[df_roc["ews"] == "SDML"]
    auc_dl = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["SDML"],
            ),
        )
    )

    # Variance plot
    df_trace = df_roc[df_roc["ews"] == "Variance"]
    auc_var = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["variance"],
            ),
        )
    )

    # Lag-1  AC plot
    df_trace = df_roc[df_roc["ews"] == "Lag-1 AC"]
    auc_ac = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["ac"],
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

    annotation_auc_dl = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc,
        text="A<sub>SDML</sub>={:.2f}".format(auc_dl),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=cols_plotly[3],
            size=font_size_auc_text,
        ),
    )

    annotation_auc_var = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc - y_auc_sep,
        text="A<sub>Var</sub>={:.2f}".format(auc_var),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=cols_D3[1],
            size=font_size_auc_text,
        ),
    )

    annotation_auc_ac = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc - 2 * y_auc_sep,
        text="A<sub>AC</sub>={:.2f}".format(auc_ac),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=cols_D3[2],
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
    list_annotations.append(annotation_auc_dl)
    list_annotations.append(annotation_auc_var)
    list_annotations.append(annotation_auc_ac)
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

def make_inset_barplot(df_roc_sdml, df_predict_forced, target_bif, save_dir):
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
    # labels = ["Forced", "Neutral"]
    labels = ["Transition", "Neutral"]

    # get SDML probability column names
    columu_SDML_probability = df_predict_forced.columns[0]
    # print(columu_SDML_probability)

    # Choose the best threshold = 选择最佳阈值
    optimal_idx = np.argmax(df_roc_sdml['tpr'] - df_roc_sdml['fpr'])
    optimal_threshold = df_roc_sdml['thresholds'][optimal_idx]
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
    df_plot = pd.DataFrame(data)

    color_main = "#A9A9A9"
    color_target = "#FFA15A"
    col_palette = {bif: color_main for bif in labels}
    col_palette[target_bif] = color_target

    b = sns.barplot(
        df_plot,
        orient="h",
        x="sdml_prob",
        y="label",
        palette=col_palette,
        linewidth=0.8,
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



surr_type = 'AAFT'
# --------------------
# 01 chick heart data
# --------------------
df_roc_sdml = pd.read_csv(f"input_data/01_chick_heart/df_roc_surrogate_{surr_type}_simples_10000_CNN_model_[14].csv")
df_roc_var_corr = pd.read_csv(f"input_data/01_chick_heart/df_roc_var_corr_[14].csv")
# Concat DataFrame by row
df_roc = pd.concat([df_roc_sdml, df_roc_var_corr], axis=0, ignore_index=True)
# # Print the DataFrame results
# print(df_roc)
# df_roc.to_csv('output_data/df_roc_all.csv', index=False)

df_predict_forced = pd.read_csv(f"input_data/01_chick_heart/df_forced_fixed_surrogate_{surr_type}_simples_10000_CNN_model_[14].csv")
# # rename columns
# df_predict_forced.rename(columns={'SDML probability': '1'}, inplace=True)
# # add new columns: 0
# df_predict_forced['0'] = (1.0 - df_dl_forced['1'])
# print(df_predict_forced)

fig_roc = make_roc_figure(df_roc, "a", text_N="N={}".format(len(df_predict_forced) * 2))
fig_roc.write_image("temp_roc.png", scale=scale)

make_inset_barplot(df_roc_sdml, df_predict_forced, "Transition", "temp_inset.png")

# Combine figs and export
path_roc = "temp_roc.png"
path_inset = "temp_inset.png"
path_out = "output_figures/roc_heart.png"

combine_roc_inset(path_roc, path_inset, path_out)


# --------------------
# 02 MS21 data
# --------------------
df_roc_sdml = pd.read_csv(f"input_data/02_MS21/df_roc_surrogate_{surr_type}_simples_10000_CNN_model_[1].csv")
df_roc_var_corr = pd.read_csv(f"input_data/02_MS21/df_roc_var_corr_[1].csv")
# Concat DataFrame by row
df_roc = pd.concat([df_roc_sdml, df_roc_var_corr], axis=0, ignore_index=True)
# print(df_roc)

df_predict_forced = pd.read_csv(f"input_data/02_MS21/df_forced_fixed_surrogate_{surr_type}_simples_10000_CNN_model_[1].csv")
# print(df_predict_forced)

fig_roc = make_roc_figure(df_roc, "b", text_N="N={}".format(len(df_predict_forced) * 2))
fig_roc.write_image("temp_roc.png", scale=scale)

make_inset_barplot(df_roc_sdml, df_predict_forced, "Transition", "temp_inset.png")

# Combine figs and export
path_roc = "temp_roc.png"
path_inset = "temp_inset.png"
path_out = "output_figures/roc_ms21.png"

combine_roc_inset(path_roc, path_inset, path_out)


# --------------------
# 03 MS66 data
# --------------------
df_roc_sdml = pd.read_csv(f"input_data/03_MS66/df_roc_surrogate_{surr_type}_simples_1000_SVM_model_[4,2].csv")
df_roc_var_corr = pd.read_csv(f"input_data/03_MS66/df_roc_var_corr_[4,2].csv")
# Concat DataFrame by row
df_roc = pd.concat([df_roc_sdml, df_roc_var_corr], axis=0, ignore_index=True)
# print(df_roc)

# df_dl_forced = pd.read_csv("data/03_MS66/df_forced_fixed_surrogate_FT_simples_1000_SVM_model_[4,2].csv")
df_predict_forced = pd.read_csv(f"input_data/03_MS66/df_forced_fixed_surrogate_{surr_type}_simples_1000_SVM_model_[4,2].csv")
# print(df_predict_forced)

fig_roc = make_roc_figure(df_roc, "c", text_N="N={}".format(len(df_predict_forced) * 2))

# Add text to indicate which ROC curve belongs to each EWS
fig_roc.add_annotation(
    x=0.15,
    y=0.93,
    text="SDML",
    showarrow=False,
    font=dict(family="Times New Roman", size=12, color=cols_plotly[3]),
)
fig_roc.add_annotation(
    x=0.31,
    y=0.70,
    text="Var",
    showarrow=False,
    font=dict(family="Times New Roman", size=12, color=cols_D3[1]),
)
fig_roc.add_annotation(
    x=0.43,
    y=0.57,
    text="AC",
    showarrow=False,
    font=dict(family="Times New Roman", size=12, color=cols_D3[2]),
)

fig_roc.write_image("temp_roc.png", scale=scale)

make_inset_barplot(df_roc_sdml, df_predict_forced, "Transition", "temp_inset.png")

# Combine figs and export
path_roc = "temp_roc.png"
path_inset = "temp_inset.png"
path_out = "output_figures/roc_ms66.png"

combine_roc_inset(path_roc, path_inset, path_out)


# --------------------
# 04 64PE data
# --------------------
df_roc_sdml = pd.read_csv(f"input_data/04_64PE/df_roc_surrogate_{surr_type}_simples_1000_LSTM_model_[10,9,7,5].csv")
df_roc_var_corr = pd.read_csv(f"input_data/04_64PE/df_roc_var_corr_[10,9,7,5].csv")
# Concat DataFrame by row
df_roc = pd.concat([df_roc_sdml, df_roc_var_corr], axis=0, ignore_index=True)
# print(df_roc)

df_predict_forced = pd.read_csv(f"input_data/04_64PE/df_forced_fixed_surrogate_{surr_type}_simples_1000_LSTM_model_[10,9,7,5].csv")
# print(df_predict_forced)

fig_roc = make_roc_figure(df_roc, "d", text_N="N={}".format(len(df_predict_forced) * 2))
fig_roc.write_image("temp_roc.png", scale=scale)

make_inset_barplot(df_roc_sdml, df_predict_forced, "Transition", "temp_inset.png")

# Combine figs and export
path_roc = "temp_roc.png"
path_inset = "temp_inset.png"
path_out = "output_figures/roc_64pe.png"

combine_roc_inset(path_roc, path_inset, path_out)



# --------------------
# 05 paleoclimate data
# --------------------
df_roc_sdml = pd.read_csv(f"input_data/05_paleoclimate/df_roc_surrogate_{surr_type}_simples_10000_SVM_model_[2,1].csv")
df_roc_var_corr = pd.read_csv(f"input_data/05_paleoclimate/df_roc_var_corr_[2,1].csv")
# Concat DataFrame by row
df_roc = pd.concat([df_roc_sdml, df_roc_var_corr], axis=0, ignore_index=True)
# print(df_roc)

df_predict_forced = pd.read_csv(f"input_data/05_paleoclimate/df_forced_fixed_surrogate_{surr_type}_simples_10000_SVM_model_[2,1].csv")
# print(df_predict_forced)

fig_roc = make_roc_figure(df_roc, "e", text_N="N={}".format(len(df_predict_forced) * 2))
fig_roc.write_image("temp_roc.png", scale=scale)

make_inset_barplot(df_roc_sdml, df_predict_forced, "Transition", "temp_inset.png")

# Combine figs and export
path_roc = "temp_roc.png"
path_inset = "temp_inset.png"
path_out = "output_figures/roc_paleoclimate.png"

combine_roc_inset(path_roc, path_inset, path_out)


# --------------------
# 06 tree_felling data
# --------------------
df_roc_sdml = pd.read_csv(f"input_data/06_tree_felling/df_roc_surrogate_{surr_type}_simples_1000_CNN_LSTM_model_[2,3,4].csv")
df_roc_var_corr = pd.read_csv(f"input_data/06_tree_felling/df_roc_var_corr_[2,3,4].csv")
# Concat DataFrame by row
df_roc = pd.concat([df_roc_sdml, df_roc_var_corr], axis=0, ignore_index=True)
# print(df_roc)

df_predict_forced = pd.read_csv(f"input_data/06_tree_felling/df_forced_fixed_surrogate_{surr_type}_simples_1000_CNN_LSTM_model_[2,3,4].csv")
# print(df_predict_forced)

fig_roc = make_roc_figure(df_roc, "f", text_N="N={}".format(len(df_predict_forced) * 2))
fig_roc.write_image("temp_roc.png", scale=scale)

make_inset_barplot(df_roc_sdml, df_predict_forced, "Transition", "temp_inset.png")

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
dst.save("output_figures/figure_4_{}.png".format(surr_type), dpi=(dpi, dpi))

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