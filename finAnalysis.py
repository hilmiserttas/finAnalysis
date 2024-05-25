import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
# Retrieve Bitcoin prices from Yahoo Finance API
btc = yf.download('BTC-USD', start='2017-01-01')

import yfinance as yf

def get_closing_prices(symbol, start_date):
    data = yf.download(symbol, start=start_date)
    return data['Close']

def calculate_returns(closing_prices):
    returns = closing_prices.pct_change() * 100
    returns = returns.dropna()
    return returns

def main():
    btc_closing_prices = get_closing_prices('BTC-USD', '2017-01-01')
    btc_returns = calculate_returns(btc_closing_prices)
    print("Bitcoin Aritmetik Getirileri:")
    print(btc_returns)

if __name__ == "__main__":
    main()

#Q2:
import yfinance as yf
import pandas as pd

# Retrieve Bitcoin prices
btc = yf.download('BTC-USD', start='2017-01-01')

# Calculate daily returns
btc['Daily Returns'] = btc['Close'].pct_change()

# Resample the data to get weekly and monthly returns
btc['Weekly Returns'] = btc['Close'].resample('W').ffill().pct_change()
btc['Monthly Returns'] = btc['Close'].resample('ME').ffill().pct_change()

# Define a function to calculate the statistics
def calculate_statistics(data):
    statistics = {
        'Mean': data.mean(),
        'Standard Deviation': data.std(),
        'Skewness': data.skew(),
        'Excess Kurtosis': data.kurtosis()
    }
    return statistics

# Calculate statistics for daily, weekly, and monthly returns
daily_stats = calculate_statistics(btc['Daily Returns'].dropna())
weekly_stats = calculate_statistics(btc['Weekly Returns'].dropna())
monthly_stats = calculate_statistics(btc['Monthly Returns'].dropna())

# Print the results
print("Daily Returns Statistics:")
print(daily_stats)
print("\nWeekly Returns Statistics:")
print(weekly_stats)
print("\nMonthly Returns Statistics:")
print(monthly_stats)

#Q3:
import yfinance as yf
import pandas as pd

# Verileri indir
nasdaq = yf.download('^IXIC', start='2017-01-01', end='2024-01-01')  # NASDAQ
apple = yf.download('AAPL', start='2017-01-01', end='2024-01-01')  # Apple
bist100 = yf.download('XU100.IS', start='2017-01-01', end='2024-01-01')  # BIST 100
thy = yf.download('THYAO.IS', start='2017-01-01', end='2024-01-01')  # Türk Hava Yolları

# BIST100 verisindeki redenominasyonu düzelt
rebase_date = '2020-07-27'
bist100.loc[bist100.index < rebase_date, 'Close'] /= 100

# Günlük, haftalık ve aylık getirileri hesapla
freqs = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}
returns = {}
for freq in freqs:
    for stock, data in {'NASDAQ': nasdaq, 'AAPL': apple, 'BIST100': bist100, 'THY': thy}.items():
        returns[f'{stock} {freq} Returns'] = data['Close'].resample(freqs[freq]).ffill().pct_change()

# istatistikleri hesapla ve tabloyu oluştur
stats = pd.DataFrame()
for name, data in returns.items():
    stats[name] = data.describe()

print(stats)
import matplotlib.pyplot as plt

# NASDAQ ve Apple için grafik
fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(nasdaq.index, nasdaq['Close'], label='NASDAQ', color='blue')
ax1.set_xlabel('Date')
ax1.set_ylabel('NASDAQ Closing Price', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Apple için ikinci bir y ekseni
ax2 = ax1.twinx()
ax2.plot(apple.index, apple['Close'], label='Apple', color='orange')
ax2.set_ylabel('Apple Closing Price', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Başlık ve etiketler
plt.title('NASDAQ vs Apple Closing Prices')
fig.tight_layout()
plt.show()
# BIST100 ve THY için zaman serisi grafiği
fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(bist100.index, bist100['Close'], label='BIST100', color='blue')
ax1.set_xlabel('Date')
ax1.set_ylabel('BIST100 Closing Price', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# THY için ikinci bir y ekseni
ax2 = ax1.twinx()
ax2.plot(thy.index, thy['Close'], label='THY', color='orange')
ax2.set_ylabel('THY Closing Price', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Başlık ve Etiketler
plt.title('Adjusted BIST100 vs THY Closing Prices')
fig.tight_layout()
plt.show()

#Q4:
import yfinance as yf
import matplotlib.pyplot as plt

# Retrieve data from Yahoo Finance
nasdaq = yf.download('^IXIC', start='2017-01-01')  # NASDAQ Index
btc = yf.download('BTC-USD', start='2017-01-01')  # Bitcoin
bist100_data = yf.download('XU100.IS', start='2017-01-01')

# Calculate daily returns
nasdaq['Daily Return'] = nasdaq['Close'].pct_change()
btc['Daily Return'] = btc['Close'].pct_change()
bist100_data['Daily Return'] = bist100_data['Close'].pct_change()

# Drop the NaN values from the returns series
nasdaq_daily_returns = nasdaq['Daily Return'].dropna()
btc_daily_returns = btc['Daily Return'].dropna()
bist100_daily_returns = bist100_data['Daily Return'].dropna()

# Plot histograms
plt.figure(figsize=(14, 7))

# NASDAQ Histogram
plt.subplot(1, 2, 1)
plt.hist(nasdaq_daily_returns, bins=100, color='blue', alpha=0.7)
plt.title('NASDAQ Daily Returns')
plt.xlabel('Returns')
plt.ylabel('Frequency')

# Bitcoin Histogram
plt.subplot(1, 2, 2)
plt.hist(btc_daily_returns, bins=100, color='orange', alpha=0.7)
plt.title('Bitcoin Daily Returns')
plt.xlabel('Returns')

plt.tight_layout()
plt.show()

# Compare the expected return and risk
nasdaq_expected_return = nasdaq_daily_returns.mean()
nasdaq_risk = nasdaq_daily_returns.std()
btc_expected_return = btc_daily_returns.mean()
btc_risk = btc_daily_returns.std()
bist100_expected_return = bist100_daily_returns.mean()
bist100_risk = bist100_daily_returns.std()
print(f"NASDAQ Expected Return (Mean): {nasdaq_expected_return:.4f}, Risk (Std Dev): {nasdaq_risk:.4f}")
print(f"Bitcoin Expected Return (Mean): {btc_expected_return:.4f}, Risk (Std Dev): {btc_risk:.4f}")
print(f"BIST100 Expected Return (Mean): {bist100_expected_return:.4f}, Risk (Std Dev): {bist100_risk:.4f}")

print("""
Based on these data, we can evaluate the relationship between expected return and risk (standard deviation):

- NASDAQ: Offers a moderate average return (0.0007) with a relatively low level of risk (standard deviation 0.0144). This reflects that NASDAQ is typically more stable and predictable, as it includes many large and mature companies.

- Bitcoin: We see that Bitcoin offers a significantly high average return (0.0024) but comes with much higher risk (standard deviation 0.0384). This indicates that Bitcoin's market value can fluctuate significantly, showcasing high volatility.

- BIST100: The Turkish stock market, BIST100, provides higher average returns (0.0010) and risk (standard deviation 0.0286) than NASDAQ. This indicates that BIST100 is a more volatile market, thus offering higher potential returns and risks.

When looking at the risk-return relationship of these three assets, we can say that Bitcoin has the highest risk and return potential, NASDAQ offers lower risk and return, and BIST100 is positioned somewhere in between. Investors generally seek higher potential returns by taking on higher risks, but this also comes with the possibility of greater losses.

The risk and return profile of each asset class should be evaluated in the context of an investor's own risk tolerance, investment objectives, and market outlook. For example, a risk-averse investor might prefer a lower-risk asset like NASDAQ, while an investor with a higher appetite for risk may lean towards assets with higher return potential, like Bitcoin.
""")


#PartB-Q1:
import yfinance as yf
import pandas as pd

# Verileri yükle
btc = yf.download('BTC-USD', start='2017-01-01')
nasdaq = yf.download('^IXIC', start='2017-01-01')

# Günlük getirileri hesapla
btc['Daily Return'] = btc['Close'].pct_change()
nasdaq['Daily Return'] = nasdaq['Close'].pct_change()

# Verileri birleştir
data = pd.DataFrame({'BTC': btc['Daily Return'], 'NASDAQ': nasdaq['Daily Return']})

# Eksik değerleri kaldır
data = data.dropna()

# Korelasyonu hesapla
correlation = data.corr()

print(correlation)

#PartB-Q2:
import yfinance as yf

# Verileri indir
nasdaq = yf.download('^IXIC', start='2017-01-01')
bist100 = yf.download('XU100.IS', start='2017-01-01')
btc = yf.download('BTC-USD', start='2017-01-01')

# Günlük, haftalık ve aylık getirileri hesapla ve Sharpe oranını bul
for asset, label in zip([nasdaq, bist100, btc], ['NASDAQ', 'BIST100', 'BTC']):
    asset['Daily Return'] = asset['Close'].pct_change()
    asset['Weekly Return'] = asset['Close'].resample('W').ffill().pct_change()
    asset['Monthly Return'] = asset['Close'].resample('ME').ffill().pct_change()

    for period in ['Daily', 'Weekly', 'Monthly']:
        mean_return = asset[f'{period} Return'].mean()
        std_dev = asset[f'{period} Return'].std()
        sharpe_ratio = mean_return / std_dev if std_dev != 0 else 0
        print(f"{label} {period} Sharpe Ratio: {sharpe_ratio}")

#Result
print("""
      Looking at the Sharpe ratios, I see that BIST100 is the best investment on a monthly basis. As someone who generally prefers long-term investments, I would allocate half of my portfolio to BIST100. I would invest 25% in Bitcoin because it has a high daily Sharpe ratio and could be beneficial for short-term trades. The remaining 25% would be invested in NASDAQ to diversify my portfolio and because it has almost the same Sharpe ratio as Bitcoin.
      """)


#PartB-Q3:
import yfinance as yf
import statsmodels.api as sm

# Verileri indir
apple = yf.download('AAPL', start='2017-01-01', interval='1mo')
nasdaq = yf.download('^IXIC', start='2017-01-01', interval='1mo')
thy = yf.download('THYAO.IS', start='2017-01-01', interval='1mo')
bist100 = yf.download('XU100.IS', start='2017-01-01', interval='1mo')

# Aylık getirileri hesapla ve eksik verileri temizle
apple['Monthly Return'] = apple['Adj Close'].pct_change()
nasdaq['Monthly Return'] = nasdaq['Adj Close'].pct_change()
apple = apple.dropna()
nasdaq = nasdaq.dropna()

thy['Monthly Return'] = thy['Adj Close'].pct_change()
bist100['Monthly Return'] = bist100['Adj Close'].pct_change()
thy = thy.dropna()
bist100 = bist100.dropna()

# Apple için OLS regresyonu
X_apple = sm.add_constant(nasdaq['Monthly Return'])  # Bağımsız değişken
y_apple = apple['Monthly Return']  # Bağımlı değişken
model_apple = sm.OLS(y_apple, X_apple).fit()
print("Apple OLS Regression Results")
print(model_apple.summary())
print(f"Apple Beta Değeri: {model_apple.params[1]}")

# THY için OLS regresyonu
X_thy = sm.add_constant(bist100['Monthly Return'])  # Bağımsız değişken
y_thy = thy['Monthly Return']  # Bağımlı değişken
model_thy = sm.OLS(y_thy, X_thy).fit()
print("\nTHY OLS Regression Results")
print(model_thy.summary())
print(f"THY Beta Değeri: {model_thy.params[1]}")

#PartB-Q4,Q5
import yfinance as yf
import statsmodels.api as sm
import pandas as pd

# Türk hisseleri ve BIST100 endeksi için veriler
symbols_tr = ['AKBNK.IS', 'TUPRS.IS', 'XU100.IS']
data_tr = yf.download(symbols_tr, start='2017-01-01', interval='1mo')['Adj Close']

# Aylık getirileri hesapla ve eksik verileri temizle
monthly_returns_tr = data_tr.pct_change().dropna()

beta_values = {}

# AKBNK ve TUPRS için beta değerlerini hesapla
for stock in ['AKBNK.IS', 'TUPRS.IS']:
    Y = monthly_returns_tr[stock]
    X = monthly_returns_tr['XU100.IS']
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    beta_values[stock] = model.params[1]
    print(f"{stock} Beta: {beta_values[stock]}")

# En yüksek beta değerine sahip hisseyi bul
highest_beta_stock = max(beta_values, key=beta_values.get)
print(f"{highest_beta_stock} has the higher beta value with Beta: {beta_values[highest_beta_stock]}")

import yfinance as yf
import statsmodels.api as sm
import pandas as pd

# ABD hisseleri ve NASDAQ endeksi için verileri indir
symbols_us = ['AAPL', 'GOOGL', 'NVDA', 'AMZN', '^IXIC']
data_us = yf.download(symbols_us, start='2017-01-01', interval='1mo')['Adj Close']

# Aylık getirileri hesapla ve eksik verileri temizle
monthly_returns_us = data_us.pct_change().dropna()

beta_values_us = {}

# ABD hisseleri için beta değerlerini hesapla
for stock in ['AAPL', 'GOOGL', 'NVDA', 'AMZN']:
    Y = monthly_returns_us[stock]
    X = monthly_returns_us['^IXIC']
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    beta_values_us[stock] = model.params[1]
    print(f"{stock} Beta: {beta_values_us[stock]}")

# En yüksek beta değerine sahip hisseyi bul
highest_beta_stock_us = max(beta_values_us, key=beta_values_us.get)
print(f"{highest_beta_stock_us} has the highest beta value with Beta: {beta_values_us[highest_beta_stock_us]}")

print("""
Based on the provided beta values, let's evaluate the stocks in Turkey and the US separately:

Turkish Stocks:
Akbank (AKBNK.IS): The beta value is 0.5649. This indicates that Akbank is less volatile compared to the BIST100 index. It responds to market movements partially but not fully.
Tüpraş (TUPRS.IS): The beta value is 0.4426. Tüpraş has a lower beta value than Akbank, indicating it's less sensitive to market movements and possesses lower volatility.

US Stocks:
Apple (AAPL): The beta value is 1.1556. Apple is more volatile relative to the NASDAQ index, indicating a strong response to market movements.
Google (GOOGL): The beta value is 0.9655. Google responds to market movements less than Apple but still significantly.
Nvidia (NVDA): The beta value is 1.8156. This is the highest beta value in this group, showing that Nvidia is the most volatile stock compared to the NASDAQ index.
Amazon (AMZN): The beta value is 1.2936. Amazon responds strongly to the market but is not as volatile as Nvidia.

Evaluation:
Nvidia (NVDA) has the highest beta value, indicating the potential for high returns when the market rises but also significant losses when the market falls.

If a sharp drop in the US economy is expected, preferring stocks with lower beta values might be sensible as they are less sensitive to market downturns. In this case, Google (GOOGL) with its lower beta value offers less risk and might exhibit more stable performance during economic downturns.

Beta values are an essential tool for investors in risk management and portfolio diversification. Investors can use this information to make informed investment decisions based on their risk tolerance and market outlook.
""")


print("""
Based on the provided OLS regression results, let's evaluate the significance of the beta values for Apple and THY, and examine how well the models fit based on their R-squared values:

### Apple OLS Regression Results:
- **Beta Value (coef for Monthly Return):** 1.1556
- **Standard Error (std err):** 0.105
- **t-statistic:** 11.049
- **P-value (P>|t|):** 0.000

The beta coefficient for Apple (1.1556) is positive and statistically significant, as the p-value is 0.000, which is typically considered significant at a level of 0.05 or lower. This indicates that Apple's beta is statistically significant.

- **R-squared:** 0.592
- **Adjusted R-squared:** 0.588

The R-squared value indicates that the model explains 59.2% of the total variance, which signifies a good fit.

### THY OLS Regression Results:
- **Beta Value (coef for Monthly Return):** 0.6063
- **Standard Error (std err):** 0.096
- **t-statistic:** 6.321
- **P-value (P>|t|):** 0.000

The beta coefficient for THY (0.6063) is positive and statistically significant, as the p-value is 0.000, indicating statistical significance.

- **R-squared:** 0.322
- **Adjusted R-squared:** 0.314

The R-squared value for THY is 32.2%, indicating that the model explains about one-third of the total variance, which is lower than Apple but still provides a useful model.

### Evaluation:
- The beta values for both stocks are statistically significant.
- The model for Apple has a higher R-squared value compared to THY, indicating a better fit in relating market returns to the stock returns for Apple.
- If low growth is expected in the US or Turkish markets, stocks with lower beta values (in this case, THY) carry less risk and might be preferred.
""")


