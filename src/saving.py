# Functions for saving figures and output

from src.paths import FIGURE_DIR, TABLE_DIR
import os

def save_figure(fig, name):
    # Leading slashes would make `name` absolute and discard FIGURE_DIR
    path = FIGURE_DIR / str(name).lstrip('/')
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving {path} as PDF ...')
    fig.savefig(path, format='pdf', bbox_inches='tight')

def save_table(df, name):
    path = TABLE_DIR / str(name).lstrip('/')
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving {path} as CSV ...')
    df.write_csv(path)