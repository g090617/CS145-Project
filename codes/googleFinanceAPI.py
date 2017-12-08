from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data

if __name__ == '__main__':
    param = {
        'q': "MSFT",  # Stock symbol (ex: "AAPL")
        'i': "3600",  # Interval size in seconds ("86400" = 1 day intervals)
        'x': "NASDAQ",  # Stock exchange symbol on which stock is traded (ex: "NASD")
        'p': "2M"  # Period (Ex: "1Y" = 1 year)
    }
    # get price data (return pandas dataframe)
    df = get_price_data(param)
    print(df)
