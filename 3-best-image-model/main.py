import pandas as pd

df_results = pd.read_csv('pytorch-image-models/results/results-imagenet.csv')

df_results['model_org'] = df_results['model']
df_results['model'] = df_results['model'].str.split('_').str[0]

def get_data(part, col):
    df = pd.read_csv(f'benchmark-{part}-amp-nhwc-pt111-cu113-rtx3090.csv').merge(df_results, on='model')
    df['secs'] = 1. / df[col]
    df['family'] = df.model.str.extract('^([a-z]+?(?:v2)?)(?:\d|_|$)')
    df = df[~df.model.str.contains('in22'), 'family'] = df.loc[df.model.str.contains('in22'), 'family'] + '_in22'
    