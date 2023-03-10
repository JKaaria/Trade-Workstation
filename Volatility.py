import yfinance as yf
import pandas as pd
import numpy as np

tickers = ['AAPL',
    'MSFT',
    'AMZN',
    'TSLA',
    'GOOGL',
    'GOOG',
    'BRK-B',
    'UNH',
    'JNJ',
    'XOM',
    'JPM',
    'META',
    'PG',
    'NVDA',
    'V',
    'CVX',
    'HD',
    'MA',
    'PFE',
    'ABBV',
    'BAC',
    'LLY',
    'KO',
    'PEP',
    'COST',
    'MRK',
    'TMO',
    'AVGO',
    'DIS',
    'MCD',
    'WMT',
    'ABT',
    'DHR',
    'CSCO',
    'ACN',
    'VZ',
    'NEE',
    'WFC',
    'CRM',
    'BMY',
    'CMCSA',
    'TXN',
    'COP',
    'ADBE',
    'PM',
    'LIN',
    'QCOM',
    'UNP',
    'UPS',
    'NKE',
    'CVS',
    'RTX',
    'LOW',
    'AMD',
    'MS',
    'AMGN',
    'MDT',
    'SPGI',
    'HON',
    'T',
    'INTU',
    'INTC',
    'ELV',
    'GS',
    'AMT',
    'IBM',
    'PYPL',
    'ORCL',
    'SCHW',
    'SBUX',
    'NFLX',
    'DE',
    'LMT',
    'CAT',
    'ADP',
    'AXP',
    'C',
    'CI',
    'BLK',
    'NOW',
    'PLD',
    'BA',
    'TMUS',
    'MDLZ',
    'DUK',
    'SO',
    'CB',
    'GILD',
    'BKNG',
    'MMC',
    'AMAT',
    'ADI',
    'MO',
    'TGT',
    'REGN',
    'TJX',
    'SYK',
    'GE',
    'ISRG',
    'ZTS',
    'PGR',
    'VRTX',
    'BDX',
    'EOG',
    'NOC',
    'CCI',
    'CME',
    'PNC',
    'CSX',
    'WM',
    'D',
    'MMM',
    'TFC',
    'CL',
    'HUM',
    'USB',
    'FISV',
    'BSX',
    'ATVI',
    'AON',
    'GM',
    'EW',
    'MU',
    'F',
    'OXY',
    'LRCX',
    'NSC',
    'EL',
    'EQIX',
    'SLB',
    'ETN',
    'ITW',
    'APD',
    'DG',
    'PXD',
    'ICE',
    'SRE',
    'SHW',
    'GD',
    'FIS',
    'MPC',
    'AEP',
    'KLAC',
    'CNC',
    'MCK',
    'FDX',
    'SNPS',
    'ADM',
    'PSA',
    'EMR',
    'CMG',
    'HCA',
    'MRNA',
    'MCO',
    'MET',
    'ORLY',
    'GIS',
    'CTVA',
    'CDNS',
    'MAR',
    'LHX',
    'ADSK',
    'CHTR',
    'FCX',
    'APH',
    'AIG',
    'VLO',
    'DVN',
    'AZO',
    'NXPI',
    'EXC',
    'ENPH',
    'ROP',
    'KMB',
    'TEL',
    'COF',
    'SYY',
    'MSI',
    'A',
    'ECL',
    'XEL',
    'WMB',
    'TRV',
    'IQV',
    'PSX',
    'STZ',
    'JCI',
    'AJG',
    'O',
    'PAYX',
    'HLT',
    'GPN',
    'TT',
    'MSCI',
    'DXCM',
    'MCHP',
    'AFL',
    'KMI',
    'ALL',
    'CTAS',
    'EA',
    'PRU',
    'HES',
    'DOW',
    'CARR',
    'ED',
    'PH',
    'FTNT',
    'NEM',
    'SBAC',
    'PEG',
    'ALB',
    'MTB',
    'YUM',
    'WELL',
    'RMD',
    'BK',
    'MNST',
    'SPG',
    'CTSH',
    'VICI',
    'HSY',
    'ROST',
    'WEC',
    'DLR',
    'ILMN',
    'KR',
    'NUE',
    'DLTR',
    'TDG',
    'AMP',
    'RSG',
    'KEYS',
    'BIIB',
    'CMI',
    'ES',
    'BAX',
    'ON',
    'IDXX',
    'PCAR',
    'VRSK',
    'OTIS',
    'WBD',
    'MTD',
    'PPG',
    'HPQ',
    'DFS',
    'DD',
    'ROK',
    'CEG',
    'AVB',
    'AME',
    'OKE',
    'KDP',
    'FAST',
    'IFF',
    'AWK',
    'HAL',
    'FRC',
    'ANET',
    'APTV',
    'STT',
    'EIX',
    'TROW',
    'CBRE',
    'DTE',
    'CTRA',
    'EXR',
    'IT',
    'GLW',
    'EQR',
    'WBA',
    'KHC',
    'FITB',
    'ZBH',
    'ODFL',
    'EBAY',
    'CPRT',
    'ETR',
    'WY',
    'AEE',
    'EPAM',
    'GWW',
    'WTW',
    'FE',
    'CDW',
    'FTV',
    'FANG',
    'BKR',
    'EFX',
    'DHI',
    'ARE',
    'ULTA',
    'SIVB',
    'GPC',
    'HIG',
    'LUV',
    'PPL',
    'VMC',
    'TSCO',
    'RF',
    'NDAQ',
    'DAL',
    'TSN',
    'CF',
    'ABC',
    'RJF',
    'MLM',
    'ANSS',
    'LH',
    'URI',
    'LYB',
    'HBAN',
    'TTWO',
    'CNP',
    'WST',
    'MOH',
    'STE',
    'IR',
    'MOS',
    'NTRS',
    'PWR',
    'MKC',
    'LEN',
    'CMS',
    'MAA',
    'MRO',
    'BR',
    'VTR',
    'BALL',
    'CHD',
    'CFG',
    'CAH',
    'ALGN',
    'PFG',
    'MPWR',
    'WAT',
    'AMCR',
    'K',
    'AES',
    'DOV',
    'SEDG',
    'TDY',
    'CLX',
    'HOLX',
    'HPE',
    'VRSN',
    'XYL',
    'ESS',
    'FDS',
    'DRI',
    'PAYC',
    'KEY',
    'PKI',
    'WAB',
    'SWKS',
    'EXPD',
    'SYF',
    'MTCH',
    'EXPE',
    'CAG',
    'CTLT',
    'FLT',
    'ATO',
    'ZBRA',
    'IEX',
    'J',
    'IRM',
    'NTAP',
    'EVRG',
    'SJM',
    'LNT',
    'CINF',
    'TRMB',
    'AVY',
    'DGX',
    'TYL',
    'IP',
    'COO',
    'BBY',
    'JBHT',
    'OMC',
    'AKAM',
    'BRO',
    'ETSY',
    'WRB',
    'JKHY',
    'CHRW',
    'TXT',
    'APA',
    'PARA',
    'KMX',
    'PEAK',
    'CPT',
    'FMC',
    'LVS',
    'UDR',
    'TER',
    'LKQ',
    'POOL',
    'GNRC',
    'VFC',
    'GRMN',
    'HWM',
    'SWK',
    'HRL',
    'LDOS',
    'INCY',
    'DPZ',
    'HST',
    'BF-B',
    'KIM',
    'CBOE',
    'PKG',
    'STX',
    'UAL',
    'LYV',
    'NVR',
    'TECH',
    'MGM',
    'PTC',
    'NDSN',
    'WDC',
    'NI',
    'MAS',
    'BXP',
    'CE',
    'SNA',
    'VTRS',
    'RCL',
    'RE',
    'LW',
    'SBNY',
    'FOXA',
    'NRG',
    'IPG',
    'L',
    'TFX',
    'CCL',
    'CMA',
    'EMN',
    'AAP',
    'WRK',
    'CRL',
    'BIO',
    'HSIC',
    'HAS',
    'CZR',
    'MKTX',
    'TAP',
    'CPB',
    'GL',
    'HII',
    'QRVO',
    'PHM',
    'AAL',
    'JNPR',
    'FFIV',
    'ZION',
    'BBWI',
    'REG',
    'BWA',
    'AIZ',
    'TPR',
    'RHI',
    'LNC',
    'PNW',
    'ALLE',
    'LUMN',
    'WHR',
    'CDAY',
    'ROL',
    'SEE',
    'OGN',
    'PNR',
    'XRAY',
    'AOS',
    'FRT',
    'WYNN',
    'DXC',
    'NWSA',
    'BEN',
    'UHS',
    'NCLH',
    'NWL',
    'ALK',
    'DVA',
    'IVZ',
    'MHK',
    'PENN',
    'FOX',
    'RL',
    'DISH',
    'VNO',
    'PVH',
    'NWS',
    'EMBC'
]

# Define the time range for the historical stock prices (1 year)
start_date = pd.Timestamp.today() - pd.DateOffset(years=1)
end_date = pd.Timestamp.today()

# Retrieve the historical stock prices for each company and calculate the volatility
volatility = {}
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        prices = stock.history(start=start_date, end=end_date)['Close']
        log_returns = pd.Series(np.log(prices) - np.log(prices.shift(1)))
        volatility[ticker] = log_returns.std() * np.sqrt(252)  # Annualized volatility
    except yf.TickerError as e:
        if e.args[0] == f"No data found for symbol '{ticker}'":
            print(f"No data found for symbol '{ticker}', symbol may be delisted")
        elif e.args[0] == f"No timezone found for symbol '{ticker}'":
            print(f"No timezone found for symbol '{ticker}', symbol may be delisted")
        else:
            print(f"Error for ticker {ticker}: {e}")
        pass
    except KeyError as e:
        print(f"Error for ticker {ticker}: {e}")
        pass

# Sort the companies based on their volatility
sorted_volatility = sorted(volatility.items(), key=lambda x: x[1])

# Return the top 10 lowest and highest volatility stocks
lowest_volatility = sorted_volatility[:10]
highest_volatility = sorted_volatility[-10:][::-1]
print(f'Top 10 lowest volatility stocks: {lowest_volatility}')
print(f'Top 10 highest volatility stocks: {highest_volatility}')
