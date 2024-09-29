from colorama import Style, Fore, Back
import logger_setup
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
        plt.plot(df[x_col], df[y_col], label=list_of_labels[idx])

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Multiple Line Plots')
    plt.legend()

    # Saving the plot as an image file
    plt.savefig(f'{PATH_OUT_VISUALS}optuna_model_perf_{MODEL_VERSION}.png')
    logger_setup.logger.debug("... FINISH")