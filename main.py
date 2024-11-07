import torch
from torch.utils.data import DataLoader
from data.database import DatabaseHandler
import os
from data.dataloader import Dataset_ETT_hour
from models.llmbased.timeseries.timellm.timellm import TimeLLM


def get_dataloader(db):
    df = db.fetch_table_as_dataframe("ETTH1")
    dataset = Dataset_ETT_hour(df)

    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=1,
        drop_last=True
    )
    return data_loader


def main():
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "timeseries.db")
    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "schema.sql")
    db = DatabaseHandler(database_path, schema_path, "timeseries")

    model = TimeLLM()
    model.set_backbone_as_llama()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.move_to_device(device)

    data_loader = get_dataloader(db)
    model.set_data(data_loader)
    model.train_time_llm()


if __name__ == "__main__":
    main()
