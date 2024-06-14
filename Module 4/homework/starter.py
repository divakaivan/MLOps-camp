import pickle
import pandas as pd
import numpy as np
import argparse

def read_data(filename, categorical=['PULocationID', 'DOLocationID']):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def parse_args():
    parser = argparse.ArgumentParser(description='Choose year and date for NYC dataset')
    parser.add_argument('--year', type=int, required=True, help='Input year')
    parser.add_argument('--month', type=int, required=True, help='Input month')
    return parser.parse_args()

def main():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    args = parse_args()
    df = read_data(f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year}-{args.month:02d}.parquet")
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    
    print('Mean trip duration:', np.mean(y_pred))

if __name__ == '__main__':
    main()
