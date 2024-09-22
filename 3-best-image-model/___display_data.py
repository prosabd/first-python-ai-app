import __read_data as rd
import plotly.express as px
w, h = 1000, 800

def show_all(df, title, size):
    return px.scatter(df, width=w, height=h, size=df[size]**2, title=title, x='secs', y='top1', log_x=True, color='family', hover_name='model_org', hover_data=[size])

show_all(rd.df, 'Inference', 'infer_img_size').show()