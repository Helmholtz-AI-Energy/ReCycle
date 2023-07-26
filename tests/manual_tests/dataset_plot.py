from data import ResidualDataset
from utils.visualisation import plot_sample
from matplotlib import pyplot as plt

if __name__ == '__main__':
    dataset_name = 'entsoe_de'
    train_set, valid_set, tests_set = ResidualDataset.from_spec(
        spec_name=dataset_name,
        historic_window=2,
        forecast_window=1,
        features_per_step=24,
    )

    historic_data, historic_pslp, historic_metadata, pslp_forecast, forecast_metadata, cat_index, reference = train_set[0]
    fig, ax = plot_sample(historic_data.flatten(), reference.flatten())
    plt.show()
