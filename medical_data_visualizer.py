import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Import data
df = pd.read_csv('medical_examination.csv', index_col=0)

# Add 'overweight' column
# calculate their BMI by dividing their weight in kilograms by the square of their height in meters.
df['BMI'] = df['weight']/(((df['height'])/100)**2)
df['overweight'] = df['BMI'].apply(lambda x: 0 if x <= 25 else 1)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 1 else 0)
df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 1 else 0)


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.

    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the collumns for the catplot to work correctly.
    df_cat = pd.melt(df[["cardio", "cholesterol", "gluc", "smoke", "alco", "active", "overweight"]], id_vars="cardio")
    df_cat = pd.DataFrame(df_cat.groupby(['cardio', 'variable','value'])['value'].count()).rename(columns={'value': 'total'}).reset_index()

    # Draw the catplot with 'sns.catplot()'
    graf = sns.catplot( x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')
    fig = graf.fig
    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.loc[
                (df['ap_lo'] <= df['ap_hi'])
                & (df['height'] >= df['height'].quantile(0.025))
                & (df['height'] <= df['height'].quantile(0.975))
                & (df['weight'] >= df['weight'].quantile(0.025))
                & (df['weight'] <= df['weight'].quantile(0.975))
                 ]
    # diastolic pressure is higher then systolic
    # height is less than the 2.5th percentile
    # height is more than the 97.5th percentile
    # weight is less then the 2.5th percentile
    # weight is more than the 97.5th percentile

    df_heat.drop(["BMI"], axis=1, inplace=True)
    df_heat.reset_index(inplace=True)

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    # Use a mask to plot only part of a matrix
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True



    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(14, 14))

    # Draw the heatmap with 'sns.heatmap()'
    ax = sns.heatmap(
        corr,
        linewidths=0.5,  # Add Lines between the correlation cells
        annot=True,  # annot parameter which will add correlation numbers to each cell in the visuals.
        fmt='0.1f',  # String formatting code to use when adding annotations.
        mask=mask,
        square=True,  # If True, set the Axes aspect to “equal” so each cell will be square-shaped
        center=0,  # Plot a heatmap for data centered on 0 with a diverging colormap
        vmin=-0.1,  # Change the limits of the colormap
        vmax=0.2,  # Change the limits of the colormap
        cmap='rocket',  # Use a different colormap
        cbar_kws={   # Use different axes for the colorbar
            'shrink': .45,
            'format': '%.2f'
        })


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
