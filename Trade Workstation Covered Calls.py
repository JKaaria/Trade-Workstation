import pandas_datareader.data as web
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import opstrat as op
from scipy.stats import norm


ticker = "MRK"

## Risk Free Rate
input = "^TNX"
output = yf.Ticker(input)
latest_price2 = output.history(period='1d')['Close'][0]
rate = round(latest_price2, 2)

def get_beta(stock_symbol, market_index_symbol):
    stock_data = yf.Ticker(stock_symbol).history(period='max')
    market_index_data = yf.Ticker(market_index_symbol).history(period='max')

    # Get the overlapping dates in the historical data
    start_date = max(stock_data.index.min(), market_index_data.index.min())
    end_date = min(stock_data.index.max(), market_index_data.index.max())

    # Slice the historical data to only include the overlapping dates
    stock_data = stock_data.loc[start_date:end_date]
    market_index_data = market_index_data.loc[start_date:end_date]

    # Calculate the daily returns for the stock and market index
    stock_returns = stock_data['Close'].pct_change()[1:]
    market_returns = market_index_data['Close'].pct_change()[1:]

    # Calculate the beta using the covariance and volatility of the returns
    covariance = stock_returns.cov(market_returns)
    market_volatility = market_returns.var()
    return covariance / market_volatility

beta = get_beta(ticker, 'SPY')

stock = yf.Ticker(ticker)
latest_price = stock.history(period='1d')['Close'][0]
price = round(latest_price, 2) 

input2 = "^VIX"
output2 = yf.Ticker(input2)
latest_price3 = output2.history(period='1d')['Close'][0]
vol = round(latest_price3, 2) 

print (ticker,"Price:",price)
print (ticker,"Beta", beta)
print ("Risk Free Rate:",rate)
print ("Volatility:",vol*beta)

St = price
K = price
r = rate
T = 90
v= (vol*beta)

call1=op.black_scholes(K=K, St=St*0.95, r=r, t=T, v=v, type='c')
print('3M Call 2 Price:',(call1['value']['option value']))

call2=op.black_scholes(K=K, St=St*1.05, r=r, t=T, v=v, type='c')
print('3M Call 2 Price:',(call2['value']['option value']))
print('Strategy Cost:',float((call2['value']['option value']))-(call1['value']['option value']))

op_1 = {'op_type':'c','strike':price*1.05,'tr_type':'s','op_pr':(call1['value']['option value'])}
op_2 = {'op_type':'c','strike':price*0.95,'tr_type':'b','op_pr':(call2['value']['option value'])}
op.multi_plotter(spot=price, op_list=[op_1,op_2],save=True,file='Covered Call Pay Off')


print('Call 1 Delta is:',call1['greeks']['delta'])
print('Call 2 Delta is:',call2['greeks']['delta'])

df = yf.download(ticker)
returns = np.log(1+df['Adj Close'].pct_change())
mu, sigma = returns.mean(), returns.std()
sim_rets = np.random.normal(mu,sigma,252)
initial = df['Adj Close'].iloc[-1]
sim_prices = initial * (sim_rets + 1).cumprod()

for _ in range(100):
    sim_rets = np.random.normal(mu,sigma,252)
    sim_prices = initial * (sim_rets + 1).cumprod()
    plt.axhline(initial,c='k')
    plt.plot(sim_prices)
    plt.ylabel(price)
    plt.grid()
    plt.title(ticker)
    plt.savefig("Covered Call - Monte Carlo")