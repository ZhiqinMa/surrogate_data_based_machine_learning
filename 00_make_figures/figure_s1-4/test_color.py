import plotly.graph_objects as go
import plotly.express as px

# 获取 Plotly 预定义的颜色
cols = px.colors.qualitative.Plotly     # 蓝色（Blue）, 红色（Red）, 绿色（Green）,紫色（Purple）,橙色（Orange）,青色（Cyan）,粉色（Pink）,浅绿色（Light Green）
# cols = px.colors.qualitative.D3           # blue, orange, green, red, purple, brown
# cols = px.colors.qualitative.G10
# cols = px.colors.qualitative.T10
# cols = px.colors.qualitative.Alphabet

# 创建条形图
fig = go.Figure()

# 添加每种颜色的条形
for i, col in enumerate(cols):
    fig.add_trace(go.Bar(x=[i], y=[1], marker_color=col, name=col))

# 更新图表布局
fig.update_layout(
    title="Plotly 预定义颜色序列：px.colors.qualitative.Plotly",
    xaxis=dict(title="颜色索引", tickvals=list(range(len(cols))), ticktext=[f"颜色 {i+1}" for i in range(len(cols))]),
    yaxis=dict(title="值（示例）"),
    barmode='group'
)

# 显示图表
fig.show()
