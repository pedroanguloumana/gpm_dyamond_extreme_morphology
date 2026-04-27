# Functions for saving figures and output

from src.paths import *
import os

def save_figure(fig, name):
    path = FIGURE_DIR / name
    print(f'Saving {path} as PDF ...')
    fig.savefig(path, format='pdf', bbox_inches='tight')