# Using IMF Currency Data

This trading strategy is designed for the [Quantiacs](https://quantiacs.com/contest) platform, which hosts competitions
for trading algorithms. Detailed information about the competitions is available on
the [official Quantiacs website](https://quantiacs.com/contest).

## How to Run the Strategy

### In an Online Environment

The strategy can be executed in an online environment using Jupiter or JupiterLab on
the [Quantiacs personal dashboard](https://quantiacs.com/personalpage/homepage). To do this, clone the template in your
personal account.

### In a Local Environment

To run the strategy locally, you need to install the [Quantiacs Toolbox](https://github.com/quantiacs/toolbox).

## Strategy Overview

The "Using IMF Currency Data" notebook introduces a trading strategy that utilizes International Monetary Fund (IMF)
currency data, specifically focusing on the Euro (EUR). This strategy leverages the predictive power of currency trends
to inform trading decisions on futures contracts. It involves calculating moving averages over different periods (10,
50, and 250 days) for the EUR spot rate and then devising a trading signal based on these averages. A long position is
taken when the short-term moving average is above the medium-term, and the medium-term is below the long-term average,
indicating a potential upward trend in currency strength that could influence futures prices. The strategy showcases the
integration of macroeconomic indicators with trading algorithms, emphasizing data-driven decision-making in futures
markets.

```python
import xarray as xr
import numpy as np
import pandas as pd

import qnt.ta as qnta
import qnt.backtester as qnbt
import qnt.data as qndata

# currencies listing
currency_list = qndata.imf_load_currency_list()
pd.DataFrame(currency_list)


def load_data(period):
    # load the AEX Index data and the spot EUR rate:
    futures = qndata.futures_load_data(assets=['F_AE'], tail=period, dims=('time', 'field', 'asset'))
    currency = qndata.imf_load_currency_data(assets=['EUR'], tail=period).isel(asset=0)
    return dict(currency=currency, futures=futures), futures.time.values


def window(data, max_date: np.datetime64, lookback_period: int):
    # build sliding window for rolling evaluation:
    min_date = max_date - np.timedelta64(lookback_period, 'D')
    return dict(
        futures=data['futures'].sel(time=slice(min_date, max_date)),
        currency=data['currency'].sel(time=slice(min_date, max_date))
    )


def strategy(data):
    # this strategy uses the currency data as predictors for the Futures contract:   
    close = data['futures'].sel(field='close')
    currency = data['currency']

    ma1 = qnta.lwma(currency, 10)
    ma2 = qnta.lwma(currency, 50)
    ma3 = qnta.lwma(currency, 250)

    if ma1.isel(time=-1) > ma2.isel(time=-1) and ma2.isel(time=-1) < ma3.isel(time=-1):
        return xr.ones_like(close.isel(time=-1))
    else:
        return xr.zeros_like(close.isel(time=-1))


weights = qnbt.backtest(
    competition_type='futures',
    load_data=load_data,
    window=window,
    lookback_period=365,
    start_date='2006-01-01',
    strategy=strategy,
    analyze=True,
    build_plots=True
)
```

More examples of use data from the International Monetary Fund (IMF)
in [documentation](https://quantiacs.com/documentation/en/data/imf.html#imf-currency-data).
