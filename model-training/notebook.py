import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    from pathlib import Path
    return Path, pl


@app.cell
def _(Path, pl):
    DATASET_PATH = Path("~/Playground/tiles/dataset-combined-512-256-jpg/")

    split_df = pl.read_csv(DATASET_PATH / "metadata" / "split_info.csv")
    slide_df = pl.read_csv(DATASET_PATH / "metadata" / "slide_info.csv")
    tiles_df = pl.read_csv(DATASET_PATH / "metadata" / "tile_info.csv")
    return slide_df, split_df, tiles_df


@app.cell
def _(split_df):
    split_df.glimpse()
    return


@app.cell
def _(slide_df):
    slide_df.glimpse()
    return


@app.cell
def _(tiles_df):
    tiles_df.glimpse()
    return


if __name__ == "__main__":
    app.run()
