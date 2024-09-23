import __read_data as rd
import plotly.express as px

## Filter convnext models who have been pretrained 

# Define constants
w, h = 1000, 800

# Define the filter
subs = 'levit|resnetd?|regnetx|vgg|convnext.*|efficientnetv2|beit|swin'

# Function to show filtered data
def show_subs(df, title, size):
    df_subs = df[df.family.str.fullmatch(subs)]
    return px.scatter(df_subs, width=w, height=h, size=df_subs[size]**2, title=title, trendline="ols", trendline_options={'log_x':True}, 
                      x='secs', y='top1', log_x=True, color='family', hover_name='model_org', hover_data=[size])
    
show_subs(rd.df, 'Inference', 'infer_img_size').show()

# Try displaying by speed vs parameter count with infer_img_size as color
px.scatter(rd.df, width=w, height=h,
    x='param_count_x',  y='secs', log_x=True, log_y=True, color='infer_img_size',
    hover_name='model_org', hover_data=['infer_samples_per_sec', 'family']
).show()