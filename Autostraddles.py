import pandas_datareader.data as web
import pandas_market_calendars as mcal
import yfinance as yf

# Get the S&P 500 tickers using pandas_market_calendars
nyse = mcal.get_calendar('NYSE')
sp500 = nyse.schedule(start_date='2000-01-01', end_date='2023-02-25')
sp500_tickers = sp500['market_open'].apply(lambda x: web.get_data_yahoo('^GSPC', start=x.date(), end=x.date()+pd.Timedelta(days=1)).index[0]).tolist()

# Loop through the tickers and perform the analysis
for ticker in sp500_tickers:
    # Your existing code here

    try:
        # Risk Free Rate
        risk_free_input = "^TNX"
        risk_free_output = yf.Ticker(risk_free_input)
        latest_price2 = risk_free_output.history(period='1d')['Close'][0]
        rate = round(latest_price2, 2)
        
        # Beta
        beta = op.get_beta(ticker, 'SPY')
        
        # Stock Price and Volatility
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

        # Calculate 3M ATM Call and Put prices
        St = price
        K = price
        r = rate
        T = 90
        v = vol*beta

        call = op.black_scholes(K=K, St=St, r=r, t=T, v=v, type='c')
        put = op.black_scholes(K=K, St=St, r=r, t=T, v=v, type='p')

        # Calculate the total cost of the straddle
        strategy_cost = call['value']['option value'] + put['value']['option value']
        print('Strategy Cost:', strategy_cost)

        # Append the results to the list
        results.append((ticker, strategy_cost))
        
    except:
        print("Error getting data for", ticker)
        continue

# Sort the results by cost and return the top 10
results.sort(key=lambda x: x[1])
top_10 = results[:10]

# Print the top 10 results
for ticker, cost in top_10:
    print(ticker, "Straddle Cost:", cost)
