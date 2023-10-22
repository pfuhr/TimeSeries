import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Test
#DataPreprocessing working on the basis of the pd dataframe
#handle missing values in a pd df in a column with a strategy
#handling duplicate data
#scaling numerical features to have similar scales
#handling categorical data

#TimeSeriesPreprocessing
#extracting relevant information, such as day of the year

import pandas as pd



class DataPreprocessing: #this object can be understood as a process
     
    def __init__(self, df):
        self.df = df

    def get_processed_data(self):
        return self.df
    
    #possible abstraction: function to add rows, where they are missing
    #for now: function to add dates, where they are missing
    #note: this creates new Nan values in the other columns
    def fill_missing_dates(self, date_column):
        #construct a pandas datetime object
        self.df[date_column] = pd.to_datetime(self.df[date_column]) #merger only possible on same type
        # Generate a list of all dates within the date range
        start_date = min(self.df[date_column])
        end_date = max(self.df[date_column])
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        #new df with dates
        data = {date_column: all_dates}
        new_df = pd.DataFrame(data)
        #merger
        self.df = pd.merge(new_df, self.df, how="left", on=date_column)
        
        

# 2 approaches to fill missing values (where we have None)
# filling them globally
# filling them locally in one column with a specific strategy

    def fill_missing_values(self, strategy='mean'): #causes errors for non numerical values
        
        if strategy == 'mean':
            self.df.fillna(self.df.mean(), inplace=True)
        elif strategy == 'median':
            self.df.fillna(self.df.median(), inplace=True)
        elif strategy == 'mode':
            self.df.fillna(self.df.mode().iloc[0], inplace=True)
        elif strategy == 'zeros':
            self.df.fillna(0.0, inplace=True)
    
        else:
            raise ValueError("Invalid fill strategy. Supported strategies: 'mean', 'median', 'mode'")
        
    # Custom function to fill missing values with a windowed mean on one pandas series
    def fill_missing_with_windowed_strategy(self, series, strategy='mean', window_size=3):
        
        result = series.copy()  # Create a copy of the series to avoid modifying the original
        for i in range(len(result)):
            if pd.isna(result[i]):
                start = max(0, i - window_size)
                end = min(len(result), i + window_size + 1)
                valid_values = [x for x in result[start:end] if not pd.isna(x)]
                if valid_values:
                    if strategy == 'mean':
                        result[i] = sum(valid_values) / len(valid_values)
                    #maybe continue with other strategies?                       
                else:
                    raise ValueError("No valid values in window")        
        return result
        
    def fill_missing_values_column(self, column, strategy = 'mean', window_size = 2):
        if strategy == 'mean':
            series = self.df[column]
            self.df[column] = self.fill_missing_with_windowed_strategy(series, strategy, window_size)
        elif strategy == 'zeros':
            series = self.df[column].copy()
            self.df[column] = series.fillna(0.0)
        else:
            raise ValueError("Invalid fill strategy. Supported strategies: 'mean', 'zeros' ")
            
    
#handling categorical data by one hot encoding 
# we want to eliminate categorical columns such that we only have numerical columns
# works!
    def one_hot_encode_column(self, column):
        # Extract the specified column
        series = self.df[column]
        
        # Perform one-hot encoding
        one_hot = pd.get_dummies(series, prefix=column)
        
        # Concatenate the one-hot encoded columns with the original DataFrame
        self.df = pd.concat([self.df, one_hot], axis=1)
       
        # Drop the original column, as it's now one-hot encoded
        del self.df[column]

    #only relevant for Time series with date timesteps    
    def weekdays(self, date_column): #date_column is a string indicating the key to the date column
        
        # Convert the 'date' column to a Pandas datetime object
        date_series = pd.to_datetime(self.df[date_column])

        # Extract the day of the week and one-hot encode it
        day_series = date_series.dt.day_name()
        one_hot = pd.get_dummies(day_series)

        # Concatenate the one-hot encoded columns with the original DataFrame
        self.df = pd.concat([self.df, one_hot], axis=1)

    #scalers: work!
    #Min-Max-Scaler: scales linearly between x_new = x_old-min/max-min
    #Standardization: assumes features to follow a normal distribution x_new = (x_old-mü)/sigma

    #cur_scaler takes arguments MinMaxScaler, StandardScaler
    def scaler(self, CurScaler = MinMaxScaler, include = 'float'): 
        
        cur_scaler = CurScaler()
        numerical_df = self.df.select_dtypes(include = include)
        
        numerical_df_scaled = pd.DataFrame(cur_scaler.fit_transform(numerical_df), columns=numerical_df.columns) #scal.fit_transform returns an array
        
        self.df.update(numerical_df_scaled)
        
        
    def scaler_column(self, column, CurScaler = MinMaxScaler): #maybe catch errors if non numerical
        cur_scaler = CurScaler()
        data = self.df[column].values.reshape(-1,1)
        #print('data:', data)
        column_scaled = cur_scaler.fit_transform(data) 
        self.df[column] = column_scaled

    def scaler_columns(self, columns, CurScaler = MinMaxScaler):
        for column in columns:
            self.scaler_column(column, CurScaler = CurScaler)
    """
    Sci-Kit-Learn Syntax for Scaling: can be applied to columns as well as to dataframes
    # Min-Max Scaling
    min_max_scaler = MinMaxScaler()
    df_min_max_scaled = min_max_scaler.fit_transform(df)

    # Standardization
    standard_scaler = StandardScaler()
    df_standardized = standard_scaler.fit_transform(df)

    # Convert the scaled arrays back to DataFrames if needed
    df_min_max_scaled = pd.DataFrame(df_min_max_scaled, columns=df.columns)
    df_standardized = pd.DataFrame(df_standardized, columns=df.columns)
    """   
    


''''
# Example usage:
data = {'A': [1, 2, None, 4, 5],
        'B': [None, 2, 3, None, 5],
        'C': [1, 2, 3, 4, 5]}
df = pd.DataFrame(data)
preprocessor = DataPreprocessor(df)

# Fill missing values with the mean of each column
preprocessor.fill_missing_values(strategy='mean')

# Or drop rows with missing values
# preprocessor.drop_missing_values()

# Get the preprocessed DataFrame
processed_data = preprocessor.get_processed_data()
print(processed_data)
'''