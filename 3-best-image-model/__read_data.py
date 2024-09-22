import pandas as pd

# Load ImageNet results
df_results = pd.read_csv('pytorch-image-models/results/results-imagenet.csv')

# Create a copy of the original 'model' column
df_results['model_org'] = df_results['model']
# Simplify model names
df_results['model'] = df_results['model'].str.split('_').str[0]

def get_data(part, col):
    # Load and merge benchmark data with ImageNet results
    df = pd.read_csv(f'pytorch-image-models/results/benchmark-{part}-amp-nhwc-pt240-cu124-rtx3090.csv').merge(df_results, on='model')
    
    # Calculate time (in seconds)
    df['secs'] = 1. / df[col]
    
    # Extract model 'family' from the name
    df['family'] = df.model.str.extract(r'^([a-z]+?(?:v2)?)(?:\d|_|$)')
    
    # Remove models ending with 'gn'
    df = df[~df.model.str.endswith('gn')]
    
    # Add '_in22' to the family for models containing 'in22'
    df.loc[df.model.str.contains('in22'), 'family'] = df.loc[df.model.str.contains('in22'), 'family'] + '_in22'
    
    # Add 'd' to the family for ResNet models
    df.loc[df.model.str.contains('resnet.*d'),'family'] = df.loc[df.model.str.contains('resnet.*d'),'family'] + 'd'
    
    # Filter the DataFrame to include only specific families
    return df[df.family.str.contains('^re[sg]netd?|beit|convnext|levit|efficient|vit|vgg|swin')]

# Get data for inference
df = get_data('infer', 'infer_samples_per_sec')
print(df)