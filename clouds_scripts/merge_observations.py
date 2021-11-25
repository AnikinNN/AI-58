# read table with observations
import os

import pandas as pd

from clouds_scripts import path
from clouds_scripts.path import observations_table_dir
from clouds_scripts.service_defs import find_files


observations = pd.DataFrame()
for file in find_files(observations_table_dir, "AI58_Clouds*.xlsx"):
    observations = observations.append(pd.read_excel(file))

# drop rows that doesnt contain SDS, because SDS is easiest feature to determine
# drop rows that doesnt contain Date_Time, because they doesnt contain anything
observations = observations.dropna(how="any", axis="index", subset=["SDS", "Date_Time"])
# delete headers
observations = observations[observations["TCC"] != "TCC"]
# keep rows with clouds by TCC
observations = observations[observations["TCC"] != "-"]


# capitalize "п" to achieve sameness
def capitalize_sds(row: pd.Series):
    if type(row["SDS"]) == str:
        row["SDS"] = row["SDS"].capitalize()
    return row


observations = observations.apply(capitalize_sds, axis=1)

# cast to datetime
observations["Date_Time"] = pd.to_datetime(observations["Date_Time"])

observations.to_csv(os.path.join(path.observations_table_dir, "AI58-observations.csv"), index=False)

print()