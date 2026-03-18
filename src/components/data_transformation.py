import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils.common import save_object


class DataTransformation:

    def _transform_dataframe(self, df):
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        
        df = df[df['fare_amount'] >= 0]
        
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        
        df['pickup_hours'] = df['tpep_pickup_datetime'].dt.hour
        df['pickup_day'] = df['tpep_pickup_datetime'].dt.day
        df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
        df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
        
        df['trip_duration(min)'] = ((df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60).astype(int)
        
        selected_cols = [
            'VendorID','passenger_count','trip_distance','RatecodeID','store_and_fwd_flag',
            'PULocationID','DOLocationID','payment_type','fare_amount','extra','mta_tax',
            'tip_amount','tolls_amount','improvement_surcharge','total_amount',
            'congestion_surcharge','trip_duration(min)'
        ]
        
        df = df[selected_cols]
        df.columns = [
            'vendor','passengers','distance','rate_id','flag','pickup_id','dropoff_id',
            'payment_type','fare','extras','tax','tip','tolls','improvement','total',
            'congestion','duration'
        ]
        
        df['flag'] = df['flag'].map({'N': 0, 'Y': 1})
        df['fare'] = df['fare'].astype('int64')
        df['payment_type'] = df['payment_type'].astype('int64')
        
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def initiate_data_transformation(self, train_path, test_path):
        train_df = pd.read_csv(train_path, low_memory=False)
        test_df = pd.read_csv(test_path, low_memory=False)

        train_df = self._transform_dataframe(train_df)
        test_df = self._transform_dataframe(test_df)

        cat_cols = ['vendor', 'rate_id', 'flag', 'pickup_id', 'dropoff_id', 'payment_type']
        num_cols = ['passengers', 'distance', 'fare', 'extras', 'tax', 'tip', 'tolls', 'improvement', 'congestion', 'duration']

        X_train = train_df[cat_cols + num_cols].copy()
        y_train = train_df['total']
        
        X_test = test_df[cat_cols + num_cols].copy()
        y_test = test_df['total']

        scaler = StandardScaler()
        
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

        save_object("artifacts/scaler.pkl", scaler)

        return X_train, X_test, y_train, y_test
