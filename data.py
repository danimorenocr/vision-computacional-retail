import pandas as pd
df = pd.read_parquet("tracking_cam_01.parquet")
print(df.groupby("track_id")["timestamp_s"].agg(["min", "max", "count"]))