import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from typing import Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

def value_counts_plot(
        data: pd.Series,
        color: str = "orange", 
        figsize: tuple = (10, 8), 
        ax: plt.Axes = None,
        alpha:float  = 1,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data.value_counts() / len(data) * 100,
                orient='h', 
                ax=ax, 
                color=color,
                alpha=alpha)
    plt.xlabel("Percentage Frequancy, %")
    plt.grid(True, axis="x")
    if ax is None:
        plt.show()

def word_count_plot(
        data: pd.Series,
        color: str = "orange", 
        figsize: tuple = (8, 6),
        count_symbols: bool = False,
        return_result: bool = False,
        ax: plt.Axes = None,
):
    if count_symbols:
        counted_data = data.str.len()
    else:
        counted_data = data.str.split().str.len()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(counted_data, ax=ax, binwidth=1, color=color)
    plt.title('Length in words', fontsize=14, fontweight='bold')
    plt.xlabel('Number of words', fontsize=12)
    plt.ylabel(f'Amount of {data.name}', fontsize=12)
    print(counted_data.describe())    
    if ax is None:
        plt.show()

    if return_result:
        return counted_data
    
def draw_wordcloud(
        frequencies: Any,
        *,
        width: int = 800,
        height: int = 400,
        max_words: int = 200, 
        colormap: Any =None,
        background_color: str = 'white',
        random_state:int = 42,
):
    word_cloud = WordCloud(
        width=width, 
        height=height, 
        max_words=max_words, 
        colormap=colormap, 
        background_color=background_color, 
        random_state=random_state
        ).generate_from_frequencies(frequencies)
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def barplot(
        data:pd.Series, 
        title: str = "Top 50 popular tokens",   
        y_label: str = "Tokens",
        figsize: tuple[int, int] = (6, 10),    
        top_n: int = 50,
        color_palette: str = 'viridis_r'
    ):
    data = data[:top_n]
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=data.values,
                y=data.index, 
                hue=data.index,
                orient='h', 
                ax=ax, 
                palette=color_palette, 
                legend=False,)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(y_label, fontsize=12)
    plt.xlabel("Frequency", fontsize=12)

    plt.show()

def plot_feature_importance(
    pipeline: Any,
    X: Any,
    y: Any,
    n_features: int = 20,
    model_name: str = 'model'
    ):

    if isinstance(pipeline, Pipeline):
            model = pipeline.named_steps['model']
    else:
            model = pipeline 
    
    pipeline.fit(X, y)

    if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):                
            if hasattr(model, "feature_importances_"):
                    feature_importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                    feature_importance = model.coef_
                    feature_importance = abs(feature_importance).sum(axis=0)
    else:
        print(f"Model {model_name} doesn't have attribute feature_importances_")
        return()

    feature_name = []

    if 'vectorizer' in pipeline.named_steps:
         vectorizer = pipeline.named_steps['vectorizer']
         feature_name.extend([name.replace("_vectorizer", "") 
            for name in vectorizer.get_feature_names_out()])
                            
    if 'svd' in pipeline.named_steps:
         svd = pipeline.named_steps['svd']
         feature_importance = np.abs(svd.components_).T @ feature_importance

    series_importance = pd.Series(data=feature_importance, index=feature_name).sort_values(ascending=False).iloc[:n_features]

    sns.barplot(series_importance, orient='h', color='firebrick')
    plt.title(f"Feature importance for {model_name}", fontsize=14, fontweight='bold')
    plt.grid(True, axis='x')
    plt.xlabel("Feature importance")
    plt.show()

if __name__ == "__main__":
    print('ok')