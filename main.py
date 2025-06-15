import click
from loguru import logger
import sys
import lib




@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    level = "DEBUG" if debug else "INFO"
    logger.remove()
    logger.add(
        "logs/world_map.log",
        level="DEBUG",
        rotation="1 MB",
        compression="zip",
    )
    logger.add(sys.stdout, format="[{time:YYYY-MM-DD HH:mm:ss}] [{level}] - {file}:{line} - {function} - {message}", level=level)


@cli.command()
def render():
    map = lib.WorldMap()
    map.render_plots()

@cli.command()
def show():
    map = lib.WorldMap()
    map.render_plots()
    map.show_plots()

if __name__ == '__main__':
    #delaney()
    #main()
    cli()