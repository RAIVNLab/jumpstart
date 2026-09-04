from jumpstart.utils.d3rl_data import get_minari
from jumpstart.utils.data_stats import DatasetStats
import tyro


def print_stats(
    env: str,
    val_split: float = 0.0,
    val_percentile: int = 50,
    download: bool = True,
):
    data, val_data, environment = get_minari(
        env_name=env,
        download=download,
        val_split=val_split,
        val_percentile=val_percentile,
    )
    stats = DatasetStats.from_dataset(data)
    print(stats)


if __name__ == "__main__":
    tyro.cli(print_stats)
