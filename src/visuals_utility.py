from colorama import Style, Fore, Back
import logger_setup
import pandas as pd
from matplotlib_venn import venn2
from conf.config import PATH_OUT_VISUALS, MODEL_VERSION

# importing visualisation libraries & stylesheets
import matplotlib.pyplot as plt
from conf.config import MPL_STYLE_FILE
plt.style.use(MPL_STYLE_FILE)

class ColourStyling(object):
    blk = Style.BRIGHT + Fore.BLACK
    gld = Style.BRIGHT + Fore.YELLOW
    grn = Style.BRIGHT + Fore.GREEN
    red = Style.BRIGHT + Fore.RED
    blu = Style.BRIGHT + Fore.BLUE
    mgt = Style.BRIGHT + Fore.MAGENTA
    res = Style.RESET_ALL

custColour = ColourStyling()

# function to render colour coded print statements
def beautify(str_to_print: str, format_type: int = 0) -> str:
    if format_type == 0:
        return custColour.mgt + str_to_print + custColour.res
    if format_type == 1:
        return custColour.grn + str_to_print + custColour.res
    if format_type == 2:
        return custColour.gld + str_to_print + custColour.res
    if format_type == 3:
        return custColour.red + str_to_print + custColour.res

def plot_line(list_of_df: list, list_of_labels: list, x_col, y_col, color='teal', figsize: tuple = (8, 6), dpi: int = 130):
    logger_setup.logger.debug("START ...")
    if list_of_labels is None:
        labels = [f'Line {i + 1}' for i in range(len(list_of_df))]

    for idx, df in enumerate(list_of_df):
        plt.plot(df[x_col], df[y_col], label=list_of_labels[idx], marker='o')

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Multiple Line Plots')
    plt.legend()

    # Saving the plot as an image file
    plt.savefig(f'{PATH_OUT_VISUALS}optuna_model_perf_{MODEL_VERSION}.png')
    logger_setup.logger.debug("... FINISH")

def plot_filled_values_percent(df: pd.DataFrame, color='teal', figsize: tuple = (8, 6), dpi: int = 130):
    logger_setup.logger.info('START ...')
    filled_values_percent = (df.notnull().sum() / len(df) * 100).sort_values()
    fig, axes = plt.subplots(figsize=figsize, dpi=dpi)
    # Create bars representing total values (set to 100)
    axes.barh(filled_values_percent.index, [100] * len(df.columns), color='#f5f5f5')
    # Create bars representing filled values
    axes.barh(filled_values_percent.index, filled_values_percent, color='turquoise')

    axes.set_xlim([0, 100])
    axes.set_xlabel('Percentage (%) filled')
    axes.set_title('Percentage of filled values in each column')
    plt.show()
    logger_setup.logger.info('... FINISH')

def plot_cat_col_cardinality(df: pd.DataFrame, color='turquoise', height=0.75,
                             figsize: tuple = (6, 9), dpi: int = 150):
    logger_setup.logger.info('START ...')
    cardinality = df.nunique().sort_values(ascending=False)
    fig, axes = plt.subplots(figsize=figsize, dpi=dpi)
    axes.barh(cardinality.index, cardinality, color=color, height=height)
    for i in range(len(df.columns)):
        axes.text(cardinality.iloc[i] + 0.5, i,
                  f'{str(cardinality.iloc[i])} among {df[cardinality.index[i]].count()}', va='center',
                  fontsize=7)
    axes.set_xlim([0, cardinality.iloc[0] + 5])
    axes.set_xlabel('Unique values (aka cardinality) among total non null values')
    axes.set_title('Cardinality of categorical columns')
    plt.show()
    logger_setup.logger.info('... FINISH')

def plot_venn_diagram(df1, df1_display_name, df2, df2_display_name, join_column, join_column_display_name, figsize: tuple = (8, 6), dpi: int = 150):
    logger_setup.logger.info('START ...')
    # Convert the joining column to a set for each dataframe
    set1 = set(df1[join_column])
    set2 = set(df2[join_column])

    common_values = set1.intersection(set2)
    # Create the venn diagram
    fig, axes = plt.subplots(figsize=figsize, dpi=dpi)
    venn = venn2([set1, set2], (df1_display_name, df2_display_name))

    # Display the plot
    plt.title(f"Venn Diagram for {join_column_display_name}")
    plt.show()
    logger_setup.logger.info('... FINISH')
    return list(common_values)