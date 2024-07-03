import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Define the ticker symbol
tickerSymbol = 'NVDA'

# Get data for the last 2 years
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

# Fetch the data directly
df = yf.download(tickerSymbol, start=start_date, end=end_date)

# Fix date column
df.reset_index(inplace=True)
df['Date'] = df['Date'].dt.strftime('%Y/%m/%d')

# Assuming df is your DataFrame with OHLC data
# Calculate 200 EMA
df['EMA_200'] = df['Close'].ewm(span=21, min_periods=0, adjust=False).mean() #21

# Assuming df is your DataFrame with OHLC data and 'EMA_200' already calculated
# Convert 'Date' column to datetime if it's not already
df['Date'] = pd.to_datetime(df['Date'])

# Calculate daily EMA slope
df['EMA_Slope'] = df['EMA_200'].diff() / df['Date'].diff().dt.days

# Calculate month-wise average slope of EMA
df['Month'] = df['Date'].dt.to_period('M')  # Create a column for month
avg_slope_by_month = df.groupby('Month')['EMA_Slope'].mean()

# Add 'Avg Slope' column to the original DataFrame
df['Avg Slope'] = df['Month'].map(avg_slope_by_month)

# Add 'Signal' column based on slope conditions
df['Signal'] = ''

# Initialize variables
last_position = ''
in_position = False
entry_price = 0
target_price = 0
stop_loss = 0
upper_limit = 0.1 # 10%
lower_limit = 0.05 # 5%

# Absolute value of slope
slope = 0.1 #2

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

                print('Long')
                print('Slope: ', df['EMA_Slope'].iloc[i])
                print('Entry Price: ', entry_price)
                print('Target Price:' , target_price)
                print('Stop-Loss:' , stop_loss)

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

                print('Short')
                print('Slope: ', df['EMA_Slope'].iloc[i])
                print('Entry Price: ', entry_price)
                print('Target Price:' , target_price)
                print('Stop-Loss:' , stop_loss)

                # Update variable
                in_position = True

    # Search for exit conditions
    elif in_position == True:

        current_price = df['Close'].iloc[i]
        
        # Check if previous signal was Long 
        if last_position == 'Long':
            
            # Checks if position should be closed
            if current_price >= target_price or current_price <= stop_loss: 

                print('Current Price: ', current_price, '\n')
                
                # Update current position
                df['Signal'].iloc[i] = 'Long Exit'
                in_position = False
        
        # Check if previous signal was Short 
        elif last_position == 'Short':

            # Checks if position should be closed
            if current_price <= target_price or current_price >= stop_loss:

                print('Current Price: ', current_price, '\n')
                
                # Update current position
                df['Signal'].iloc[i] = 'Short Exit'
                in_position = False
                
# Backtesting

# Lists to create profit (%) and profit ($) records
profit_p = []
profit_d = []

# Initialize variables 
initial_capital = 10000
capital = initial_capital
entry_price = 0
exit_price = 0
profit = 0
position = ''

# Lambda functions calculate gain/loss
calc_profit = lambda x, y, p: (x - y) / y if p == 'l' else (y - x) / y
calc_percentage = lambda x: round(x * 100, 2)
calc_dollar = lambda x, y: round(x * y, 2)

# Check enrty and exit signals
for i in range(len(df)):

    # Look for an entry signal, records entry price
    if position == '':
        if df['Signal'].iloc[i] == 'Long Entry':
            position = 'Long'
        elif df['Signal'].iloc[i] == 'Short Entry':
            position = 'Short'
        entry_price = df['Close'].iloc[i]                                              
    
    # If in a position
    elif position != '': 
        exit_price = df['Close'].iloc[i]

        # If Long Exit, calculate proift or loss
        if df['Signal'].iloc[i] == 'Long Exit':
            profit = calc_profit(exit_price, entry_price, 'l')
            capital += (capital * profit)
            position = ''
    
            # Appending data to list so it can be added to records dataframe
            profit_p.append(calc_percentage(profit))
            profit_d.append(calc_dollar(profit, capital))

        # If Short Exit, calculate proift or loss
        elif df['Signal'].iloc[i] == 'Short Exit':
            profit = calc_profit(exit_price, entry_price, 's')
            capital += (capital * profit)
            position = ''

            # Appending data to list so it can be added to records dataframe
            profit_p.append(calc_percentage(profit))
            profit_d.append(calc_dollar(profit, capital))

# Print ending funds
print('Ending funds: $', round(capital, 2))
print('Profit: +', round(((capital / initial_capital)) * 10, 2), '%') if capital >= initial_capital else print('Profit: -', round(((capital / initial_capital)) * 10, 3), '%')
print('Profit: + $', round(((capital-initial_capital)), 2), '\n') if capital >= initial_capital else print('Profit: - $', round(((initial_capital - capital)), 2), '%', '\n')


# This code block is just for vizualization
df['Position'] = np.nan
position = False
for i in range(len(df)-1):
    if position == False:
        if df['Signal'].iloc[i] == 'Long Entry' or df['Signal'].iloc[i] == 'Short Entry':
            df['Position'].iloc[i] = df['Close'].iloc[i]
            position = True
    elif position == True:
        if df['Signal'].iloc[i] == 'Long Exit' or df['Signal'].iloc[i] == 'Short Exit':
            df['Position'].iloc[i] = df['Close'].iloc[i]
            position = False
        else:
            df['Position'].iloc[i] = df['Close'].iloc[i]

# Trade records dataframe
records = pd.DataFrame(columns=['Position','Entry','Exit','Profit (%)','Profit ($)','Win/Loss'])

# Adding Enrty column
temp = df[df['Signal'].isin(['Long Entry', 'Short Entry'])].reset_index(drop=True)
records['Entry'] = temp['Close']

# Adding Exit column
temp = df[df['Signal'].isin(['Long Exit', 'Short Exit'])].reset_index(drop=True) # Reset index so new columns are joined even when value is at diff index
records['Exit'] = temp['Close']

# Drop last row incase position hasn't exited yet
records.drop(records.index[-1], inplace=True)

# Adding position column
for i in range(len(records)):
    if temp['Signal'].iloc[i] == 'Long Exit':
        records['Position'].iloc[i] = 'Long'
    elif temp['Signal'].iloc[i] == 'Short Exit':
        records['Position'].iloc[i] = 'Short'

# Adding Profit columns
for i in range(len(records)):
    records['Profit (%)'].iloc[i] = profit_p[i]
    records['Profit ($)'].iloc[i] = profit_d[i]

for i in range(len(records)):
    if records['Profit ($)'].iloc[i] <= 0:
        records['Win/Loss'].iloc[i] = 'L'
    else:
        records['Win/Loss'].iloc[i] = 'W'

# Getting statistics
print('W/L Ratio: ', sum(records['Win/Loss'] == 'W'), '-', sum(records['Win/Loss'] == 'L'), ' | ', round((records['Win/Loss'] == 'W').mean() / (records['Win/Loss'] == 'L').mean(), 2), '%')
print('Mean Wins: ', round( records.loc[records['Win/Loss'] == 'W', 'Profit (%)'].mean(), 2))
print('Mean Losses: ', round(records.loc[records['Win/Loss'] == 'L', 'Profit (%)'].mean(), 2), '\n')

# Printing records df
print(records)

# Plotting
plt.figure(figsize=(14,7))
plt.plot(df['Close'], label='Close Price')
plt.plot(df['EMA_200'], label='200 EMA', color='red')
plt.plot(df['Position'], color='yellow')

for i in range(1, len(df)):
    if df['Signal'].iloc[i] == 'Long Entry':
        plt.scatter(i, df['Close'].iloc[i], marker='^', color='lime', s=100)
    elif df['Signal'].iloc[i] == 'Short Entry':
        plt.scatter(i, df['Close'].iloc[i], marker='v', color='red', s=100)
    elif df['Signal'].iloc[i] == 'Short Exit':
        plt.scatter(i, df['Close'].iloc[i], marker='^', color='red', s=100)
    elif df['Signal'].iloc[i] == 'Long Exit':
        plt.scatter(i, df['Close'].iloc[i], marker='v', color='lime', s=100)

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price with 200 EMA')
plt.legend()
plt.grid(True)
plt.show()