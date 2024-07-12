import os 
import sys
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import warnings
warnings.filterwarnings('ignore')

ema = 55
tp = 2
sl = 1
slope = 0.00005

i = 'gRlkn6fkjj7rD4i7eOtYIm4qDA0jhXT6uIjAEqR2EgLCI4jweygYlrtYi9vbuzhc'
s = 'RN7Ye5lGUvsQwfNMdbLUr5Va0qwszFhZJnRQylb5HMSU6A4kgJ1AbwBNWaql021Z'

client = Client(i,s)

start = True
position = ''
    
while start == True:    

    try:
        Coin = client.get_historical_klines(symbol = 'WIFUSDT', interval = '5m', start_str = '1d ago UTC')
        df = pd.DataFrame(data = Coin)
    except BinanceAPIException as e:
        print("ERROR: DataReader.getData:", e.message)
    except AttributeError as e:
        print("ERROR: DataReader.getData: Invalid start-time\n", e)
    else:
        #cleaning data
        df.columns = ["Date", "Open", "High", "Low", "Close", "volume", "close time", "quote", "trade", "1", "2", "3"]
        df.drop(["1","2","3","quote","trade","close time", "volume"],axis = 1,inplace = True)
        df["Open"]= df.Open.astype(float) 
        df["Close"] = df.Close.astype(float)
        df["Low"] = df.Low.astype(float)
        df["High"] = df.High.astype(float)
            
    # Assuming df is your DataFrame with OHLC data
    # Calculate 200 EMA
    df['EMA'] = df['Close'].ewm(span=ema, min_periods=0, adjust=False).mean() #21

    # Add EMA slope column to stock_data
    df['EMA_Slope'] = 0

    # Calculate the slope of EMA
    for i in range(len(df)):
        df['EMA_Slope'].iloc[i] = (df['EMA'].iloc[i] - df['EMA'].iloc[i-1]) / (2 - 1)

    # Fix Date column
    for i in range(len(df["Date"])):
        df["Date"].iloc[i] = datetime.datetime.fromtimestamp(df["Date"].iloc[i]/1000)

    # Add 'Signal' column based on slope conditions
    df['Signal'] = ''

    # Initialize variables
    last_position = ''
    in_position = False
    entry_price = 0
    target_price = 0
    stop_loss = 0
    upper_limit = tp # 10%
    lower_limit = sl # 5%

    # Search for and records entry and exit conditions
    for i in range(len(df)):

        # Search for entry conditions
        if in_position == False:
                    
            # Set initial position. Check whether entering long or short.
            if last_position == '':

                # Set the initial "last position" equal to opposite of current position (If we're about to enter a long, set it to short)
                last_position = 'Long' if df['EMA_Slope'].iloc[i] <= 0 else 'Short'

            # Check if previous signal was Short 
            if last_position == 'Short':

                # Enrty condition: if the  value of the slope exceeds a threshold (slope variable), it triggers an entry signal
                if df['EMA_Slope'].iloc[i-1] < slope and df['EMA_Slope'].iloc[i] >= slope:

                    # Update current position
                    df['Signal'].iloc[i] = 'Long Entry'
                    last_position = 'Long'

                    # Entry price 
                    entry_price = df['Close'].iloc[i]

                    # Calculate target price and stop loss
                    target_price = entry_price + (upper_limit * entry_price) 
                    stop_loss = entry_price - (lower_limit * entry_price) 

                    # Update variable
                    in_position = True 

            # Check if previous signal was Long      
            elif last_position == 'Long':

                # Enrty condition: if the  value of the slope exceeds a threshold (slope variable), it triggers an entry signal
                if df['EMA_Slope'].iloc[i-1] > slope * -1 and df['EMA_Slope'].iloc[i] <= slope * -1:

                    # Update current position
                    df['Signal'].iloc[i] = 'Short Entry'
                    last_position = 'Short'

                    # Entry price
                    entry_price = df['Close'].iloc[i] 
                        
                    # Calculate target price and stop loss
                    target_price = entry_price - (upper_limit * entry_price) 
                    stop_loss = entry_price + (lower_limit * entry_price)

                    # Update variable
                    in_position = True

        # Search for exit conditions
        elif in_position == True:

            current_price = df['Close'].iloc[i]
            
            # Check if previous signal was Long 
            if last_position == 'Long':
                
                # Checks if position should be closed
                if current_price >= target_price or current_price <= stop_loss: 
                    
                    # Update current position
                    df['Signal'].iloc[i] = 'Long Exit'
                    in_position = False
            
            # Check if previous signal was Short 
            elif last_position == 'Short':

                # Checks if position should be closed
                if current_price <= target_price or current_price >= stop_loss:
                    
                    # Update current position
                    df['Signal'].iloc[i] = 'Short Exit'
                    in_position = False
        
    # Enter long position
    if df['Signal'].iloc[-2] == 'Long Entry' and position == '':
        position = 'Long'
        print('Long Entry @: ', df['Close'].iloc[-2])
    
    # Enter short position
    elif df['Signal'].iloc[-2] == 'Short Entry' and position == '':
        position = 'Short'
        print('Short Entry @: ', df['Close'].iloc[-2])
    
    # Exit long position
    elif df['Signal'].iloc[-2] == 'Long Exit' and position == 'Long':
        position = ''
        print('Long Exit @: ', df['Close'].iloc[-2])
    
    # Exit short position
    elif df['Signal'].iloc[-2] == 'Short Exit' and position == 'Short':
        position = ''
        print('Short Exit @: ', df['Close'].iloc[-2])
   
    # If no position is entered
    else:
        pass

    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(df)

    time.sleep(300)

