import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

data = pd.read_csv('TSLA DATA - Sheet1.csv')

data['Date'] = pd.to_datetime(data['Date'])

data = data.sort_values('Date')

data = data.tail(30).reset_index(drop=True)

data['Date_ordinal'] = data['Date'].map(datetime.toordinal)

X = data[['Date_ordinal']]  
y = data['Close']           

model = LinearRegression()
model.fit(X, y)

data['Predicted_Close'] = model.predict(X)

us_business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())

future_dates = []
current_date = data['Date'].iloc[-1]

while len(future_dates) < 20:
    current_date += us_business_day
    future_dates.append(current_date)

future_data = pd.DataFrame({'Date': future_dates})
future_data['Date_ordinal'] = future_data['Date'].map(datetime.toordinal)

future_data['Predicted_Close'] = model.predict(future_data[['Date_ordinal']])

print("Predicted Closing Prices for the Next 20 Trading Days:")
print(future_data[['Date', 'Predicted_Close']])

combined_data = pd.concat(
    [data[['Date', 'Close', 'Predicted_Close']], future_data[['Date', 'Predicted_Close']]],
    ignore_index=True
)

plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='Actual Closing Price')
plt.plot(combined_data['Date'], combined_data['Predicted_Close'], label='Predicted Closing Price', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Stock Closing Price Regression and Future Predictions')
plt.legend()
plt.show()
