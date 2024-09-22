import __read_data as rd
import plotly.express as px

# Define constants
w, h = 1000, 800

# Define the function to display data
def show_all(df, title, size):
    # Create a scatter method
    return px.scatter(df, width=w, height=h, size=df[size]**2, title=title, x='secs', y='top1', log_x=True, color='family', hover_name='model_org', hover_data=[size])

# Call the function to display the data, and show it on default browser
show_all(rd.df, 'Inference', 'infer_img_size').show()