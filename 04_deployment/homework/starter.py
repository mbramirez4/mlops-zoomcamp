#!/usr/bin/env python
# coding: utf-8

import os
import pickle
from argparse import ArgumentParser

import pandas as pd
from flask import Flask, request

def read_data(filename, year, month):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")
    
    return df

def prepare_dicts(df, categorical):
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    dicts = df[categorical].to_dict(orient='records')
    
    return dicts
    

def get_paths(year, month):
    url_src =  f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"output/yelow_taxi_{year:04d}-{month:02d}.parquet"
    
    return url_src, output_file

def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"].copy()
    df_result["prediction"] = y_pred

    df_result.to_parquet(
        output_file,
        engine="pyarrow",
        compression=None,
        index=False
    )

def apply_model(year, month, save_files=False):
    categorical = ['PULocationID', 'DOLocationID']

    url_src, output_file = get_paths(year, month)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Reading data from {url_src}...")
    df = read_data(url_src, year, month)
    
    print("Preparing data for predictions...")
    dicts = prepare_dicts(df, categorical)

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    X_val = dv.transform(dicts)

    print("Making predictions...")
    y_pred = model.predict(X_val)

    if save_files:
        print(f"Saving results to {output_file}...")
        save_results(df, y_pred, output_file)
    
    return y_pred

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    args = parser.parse_args()
    
    y_pred = apply_model(args.year, args.month)
    print(f"The mean predicted duration for {args.year:04d}/{args.month:02d} "+\
          f"is: {y_pred.mean():.2f}")