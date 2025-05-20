import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    from pathlib import Path
    import marimo as mo
    return Path, mo, pl


@app.cell
def _(Path, pl):
    DATASET_PATH = Path("~/Playground/tiles/dataset-combined-512-256-jpg/")

    split_df = pl.read_csv(DATASET_PATH / "metadata" / "split_info.csv")
    slide_df = pl.read_csv(DATASET_PATH / "metadata" / "slide_info.csv")
    tiles_df = pl.read_csv(DATASET_PATH / "metadata" / "tile_info.csv")
    return slide_df, split_df, tiles_df


@app.cell
def _(mo):
    mo.md(r"""# General Information""")
    return


@app.cell
def _(split_df):
    split_df.glimpse(max_items_per_column=3)
    return


@app.cell
def _(split_df):
    split_df.head(5)
    return


@app.cell
def _(slide_df):
    slide_df.glimpse(max_items_per_column=3)
    return


@app.cell
def _(slide_df):
    slide_df.head(5)
    return


@app.cell
def _(tiles_df):
    tiles_df.glimpse(max_items_per_column=3)
    return


@app.cell
def _(tiles_df):
    tiles_df.head(5)
    return


@app.cell
def _(mo):
    mo.md(r"""# Slide Information""")
    return


@app.cell
def _(slide_df, split_df):
    slide_split_join_df = slide_df.join(split_df, on="slide_name")

    slide_split_join_df.head(5)
    return (slide_split_join_df,)


@app.cell
def _(pl, slide_split_join_df):
    # Overall Information

    stats = slide_split_join_df["tile_count"]
    overall = pl.DataFrame({
        "set": ["ALL"],
        "s_count": [len(slide_split_join_df)],
        "s_pct": [100.0],
        "t_min": [stats.min()],
        "t_max": [stats.max()],
        "t_avg": [round(stats.mean(), 0)],
        "t_std": [round(stats.std(), 0)],
        "t_median": [stats.median()],
        "t_count": [stats.sum()],
        "t_pct": [100.0],
    })

    overall = overall.cast({"s_count": pl.UInt32})

    overall
    return (overall,)


@app.cell
def _(overall, pl, slide_split_join_df):
    # Category-wise Information

    per_category = (
        slide_split_join_df
        .group_by("category")
        .agg([
            pl.len().alias("s_count"),
            pl.col("tile_count").min().alias("t_min"),
            pl.col("tile_count").max().alias("t_max"),
            pl.col("tile_count").mean().round(0).alias("t_avg"),
            pl.col("tile_count").std().round(0).alias("t_std"),
            pl.col("tile_count").median().alias("t_median"),
            pl.col("tile_count").sum().alias("t_count"),
        ])
        .with_columns(
            (pl.col("s_count") / pl.col("s_count").sum() * 100).round(2).alias("s_pct"),
            (pl.col("t_count") / pl.col("t_count").sum() * 100).round(2).alias("t_pct")
        )
        .sort(by="t_count", descending=True)
    )

    per_category = per_category.rename({"category": "set"})
    per_category = per_category.select(overall.columns)

    per_category
    return (per_category,)


@app.cell
def _(overall, pl, slide_split_join_df):
    # Split-wise Information

    per_split = (
        slide_split_join_df
        .group_by("split")
        .agg([
            pl.len().alias("s_count"),
            pl.col("tile_count").min().alias("t_min"),
            pl.col("tile_count").max().alias("t_max"),
            pl.col("tile_count").mean().round(0).alias("t_avg"),
            pl.col("tile_count").std().round(0).alias("t_std"),
            pl.col("tile_count").median().alias("t_median"),
            pl.col("tile_count").sum().alias("t_count"),
        ])
        .with_columns(
            (pl.col("s_count") / pl.col("s_count").sum() * 100).round(2).alias("s_pct"),
            (pl.col("t_count") / pl.col("t_count").sum() * 100).round(2).alias("t_pct")
        )
        .sort(by=["t_count"], descending=True)
    )

    per_split = per_split.rename({"split": "set"})
    per_split = per_split.select(overall.columns)

    per_split
    return (per_split,)


@app.cell
def _(overall, pl, slide_split_join_df):
    # Category-wise & Split-wise Information

    per_category_split = (
        slide_split_join_df
        .group_by(["category", "split"])
        .agg([
            pl.len().alias("s_count"),
            pl.col("tile_count").min().alias("t_min"),
            pl.col("tile_count").max().alias("t_max"),
            pl.col("tile_count").mean().round(0).alias("t_avg"),
            pl.col("tile_count").std().round(0).alias("t_std"),
            pl.col("tile_count").median().alias("t_median"),
            pl.col("tile_count").sum().alias("t_count"),
        ])
        .with_columns(
            (pl.col("s_count") / pl.col("s_count").sum() * 100).round(2).alias("s_pct"),
            (pl.col("t_count") / pl.col("t_count").sum() * 100).round(2).alias("t_pct"),
            pl.format("{}_{}", pl.col("category"), pl.col("split")).alias("set")
        )
        .sort(by=["t_count"], descending=True)
    )

    per_category_split.drop_in_place("category")
    per_category_split.drop_in_place("split")
    per_category_split = per_category_split.select(overall.columns)

    per_category_split
    return (per_category_split,)


@app.cell
def _(overall, per_category, per_category_split, per_split, pl):
    # All Slide Type Information

    all_slide_metrics_df = pl.concat([overall, per_category, per_split, per_category_split])

    all_slide_metrics_df
    return


@app.cell
def _(mo):
    mo.md(r"""# Tile Information""")
    return


@app.cell
def _(slide_split_join_df, tiles_df):
    combined_df = slide_split_join_df.join(tiles_df, on="slide_name")

    combined_df.head(5)
    return (combined_df,)


@app.cell
def _(combined_df, pl):
    # Overall Tile Tumor Percentage Stats

    tile_stats = combined_df["tumor_percentage"]
    overall_tile = pl.DataFrame({
        "set": ["ALL"],
        "min": [round(tile_stats.min(), 2)],
        "max": [round(tile_stats.max(), 2)],
        "avg": [round(tile_stats.mean(), 2)],
        "std": [round(tile_stats.std(), 2)],
        "median": [round(tile_stats.median(), 2)],
    })


    overall_tile
    return


@app.cell
def _(combined_df, pl):
    # Per Split Set Tile Tumor Percentage Stats

    per_split_tile = (
        combined_df
        .group_by("split")
        .agg([
            pl.col("tumor_percentage").min().round(2).alias("min"),
            pl.col("tumor_percentage").max().round(2).alias("max"),
            pl.col("tumor_percentage").mean().round(2).alias("avg"),
            pl.col("tumor_percentage").std().round(2).alias("std"),
            pl.col("tumor_percentage").median().round(2).alias("median"),
        ])
        .with_columns(
        )
        .sort(by=["split"], descending=True)
    )

    per_split_tile = per_split_tile.rename({"split": "set"})

    per_split_tile
    return


if __name__ == "__main__":
    app.run()
