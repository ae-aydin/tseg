import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

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
    overall = pl.DataFrame(
        {
            "set": ["ALL"],
            "slide_count": [len(slide_split_join_df)],
            "slide_pct": [100.0],
            "tile_min": [stats.min()],
            "tile_max": [stats.max()],
            "tile_avg": [round(stats.mean(), 0)],
            "tile_std": [round(stats.std(), 0)],
            "tile_median": [stats.median()],
            "tile_count": [stats.sum()],
            "tile_pct": [100.0],
        }
    )

    overall = overall.cast({"slide_count": pl.UInt32})

    overall
    return (overall,)


@app.cell
def _(overall, pl, slide_split_join_df):
    # Category-wise Information

    per_category = (
        slide_split_join_df.group_by("category")
        .agg(
            [
                pl.len().alias("slide_count"),
                pl.col("tile_count").min().alias("tile_min"),
                pl.col("tile_count").max().alias("tile_max"),
                pl.col("tile_count").mean().round(0).alias("tile_avg"),
                pl.col("tile_count").std().round(0).alias("tile_std"),
                pl.col("tile_count").median().alias("tile_median"),
                pl.col("tile_count").sum().alias("tile_count"),
            ]
        )
        .with_columns(
            (pl.col("slide_count") / pl.col("slide_count").sum() * 100)
            .round(2)
            .alias("slide_pct"),
            (pl.col("tile_count") / pl.col("tile_count").sum() * 100)
            .round(2)
            .alias("tile_pct"),
        )
        .sort(by="tile_count", descending=True)
    )

    per_category = per_category.rename({"category": "set"})
    per_category = per_category.select(overall.columns)

    per_category
    return (per_category,)


@app.cell
def _(overall, pl, slide_split_join_df):
    # Split-wise Information

    per_split = (
        slide_split_join_df.group_by("split")
        .agg(
            [
                pl.len().alias("slide_count"),
                pl.col("tile_count").min().alias("tile_min"),
                pl.col("tile_count").max().alias("tile_max"),
                pl.col("tile_count").mean().round(0).alias("tile_avg"),
                pl.col("tile_count").std().round(0).alias("tile_std"),
                pl.col("tile_count").median().alias("tile_median"),
                pl.col("tile_count").sum().alias("tile_count"),
            ]
        )
        .with_columns(
            (pl.col("slide_count") / pl.col("slide_count").sum() * 100)
            .round(2)
            .alias("slide_pct"),
            (pl.col("tile_count") / pl.col("tile_count").sum() * 100)
            .round(2)
            .alias("tile_pct"),
        )
        .sort(by=["tile_count"], descending=True)
    )

    per_split = per_split.rename({"split": "set"})
    per_split = per_split.select(overall.columns)

    per_split
    return (per_split,)


@app.cell
def _(overall, pl, slide_split_join_df):
    # Category-wise & Split-wise Information

    per_category_split = (
        slide_split_join_df.group_by(["category", "split"])
        .agg(
            [
                pl.len().alias("slide_count"),
                pl.col("tile_count").min().alias("tile_min"),
                pl.col("tile_count").max().alias("tile_max"),
                pl.col("tile_count").mean().round(0).alias("tile_avg"),
                pl.col("tile_count").std().round(0).alias("tile_std"),
                pl.col("tile_count").median().alias("tile_median"),
                pl.col("tile_count").sum().alias("tile_count"),
            ]
        )
        .with_columns(
            (pl.col("slide_count") / pl.col("slide_count").sum() * 100)
            .round(2)
            .alias("slide_pct"),
            (pl.col("tile_count") / pl.col("tile_count").sum() * 100)
            .round(2)
            .alias("tile_pct"),
            pl.format("{}_{}", pl.col("category"), pl.col("split")).alias("set"),
        )
        .sort(by=["tile_count"], descending=True)
    )

    per_category_split.drop_in_place("category")
    per_category_split.drop_in_place("split")
    per_category_split = per_category_split.select(overall.columns)

    per_category_split
    return (per_category_split,)


@app.cell
def _(overall, per_category, per_category_split, per_split, pl):
    # All Slide Tile Information

    all_slide_metrics_df = pl.concat(
        [overall, per_category, per_split, per_category_split]
    )

    all_slide_metrics_df
    return (all_slide_metrics_df,)


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
    # Overall Tile Tumor Fraction Stats

    tile_stats = combined_df["tumor_frac"]
    overall_tumor = pl.DataFrame(
        {
            "set": ["ALL"],
            "tumor_min": [round(tile_stats.min(), 2)],
            "tumor_max": [round(tile_stats.max(), 2)],
            "tumor_avg": [round(tile_stats.mean(), 2)],
            "tumor_std": [round(tile_stats.std(), 2)],
            "tumor_median": [round(tile_stats.median(), 2)],
        }
    )

    overall_tumor
    return (overall_tumor,)


@app.cell
def _(combined_df, overall_tumor, pl):
    # Per Category Tile Tumor Fraction Stats

    per_category_tumor = (
        combined_df.group_by("category")
        .agg(
            [
                pl.col("tumor_frac").min().round(2).alias("tumor_min"),
                pl.col("tumor_frac").max().round(2).alias("tumor_max"),
                pl.col("tumor_frac").mean().round(2).alias("tumor_avg"),
                pl.col("tumor_frac").std().round(2).alias("tumor_std"),
                pl.col("tumor_frac").median().round(2).alias("tumor_median"),
            ]
        )
        .with_columns()
        .sort(by=["category"], descending=True)
    )

    per_category_tumor = per_category_tumor.rename({"category": "set"})
    per_category_tumor = per_category_tumor.select(overall_tumor.columns)

    per_category_tumor
    return (per_category_tumor,)


@app.cell
def _(combined_df, overall_tumor, pl):
    # Per Split Set Tile Tumor Fraction Stats

    per_split_tumor = (
        combined_df.group_by("split")
        .agg(
            [
                pl.col("tumor_frac").min().round(2).alias("tumor_min"),
                pl.col("tumor_frac").max().round(2).alias("tumor_max"),
                pl.col("tumor_frac").mean().round(2).alias("tumor_avg"),
                pl.col("tumor_frac").std().round(2).alias("tumor_std"),
                pl.col("tumor_frac").median().round(2).alias("tumor_median"),
            ]
        )
        .sort(by=["split"], descending=True)
    )

    per_split_tumor = per_split_tumor.rename({"split": "set"})
    per_split_tumor = per_split_tumor.select(overall_tumor.columns)

    per_split_tumor
    return (per_split_tumor,)


@app.cell
def _(combined_df, overall_tumor, pl):
    # Per Category & Split Set Tile Tumor Fraction Stats

    per_category_split_tumor = (
        combined_df.group_by(["category", "split"])
        .agg(
            [
                pl.col("tumor_frac").min().round(2).alias("tumor_min"),
                pl.col("tumor_frac").max().round(2).alias("tumor_max"),
                pl.col("tumor_frac").mean().round(2).alias("tumor_avg"),
                pl.col("tumor_frac").std().round(2).alias("tumor_std"),
                pl.col("tumor_frac").median().round(2).alias("tumor_median"),
            ]
        )
        .with_columns(
            pl.format("{}_{}", pl.col("category"), pl.col("split")).alias("set")
        )
        .sort(by=["split"], descending=True)
    )

    per_category_split_tumor = per_category_split_tumor.select(overall_tumor.columns)

    per_category_split_tumor
    return (per_category_split_tumor,)


@app.cell
def _(
    overall_tumor,
    per_category_split_tumor,
    per_category_tumor,
    per_split_tumor,
    pl,
):
    # All Tumor Tile Information

    all_tumor_metrics_df = pl.concat(
        [overall_tumor, per_category_tumor, per_split_tumor, per_category_split_tumor]
    )

    all_tumor_metrics_df
    return (all_tumor_metrics_df,)


@app.cell
def _(mo):
    mo.md(r"""# Combined Metrics""")
    return


@app.cell
def _(all_slide_metrics_df, all_tumor_metrics_df):
    final_df = all_slide_metrics_df.join(all_tumor_metrics_df, on="set")

    final_df
    return


if __name__ == "__main__":
    app.run()
