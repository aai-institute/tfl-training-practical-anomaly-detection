from nbformat import read, write
from glob import glob
import click

@click.command()
def normalize():
    for file in glob('notebooks/*.ipynb', recursive=False):
        print(file)
        with open(file, 'r', encoding='utf-8') as f:
            nb = read(f, 4)
        with open(file, 'w', encoding='utf-8') as f:
            write(nb, f, 4)