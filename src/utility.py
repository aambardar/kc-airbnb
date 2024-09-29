from colorama import Style, Fore, Back
import logging
import inspect
import configparser
import os
import conf.config

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
