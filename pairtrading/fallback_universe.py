"""
fallback_universe.py
====================
A hard-coded snapshot of NSE F&O-eligible equity symbols.

This is ONLY used when the live NSE endpoints are unreachable (network block,
maintenance, etc.).  It keeps the pipeline runnable end-to-end without ever
asking the user for a manual list.  Company names are filled from Yahoo Finance
metadata downstream when this fallback is used.

The list intentionally over-provides (~215 liquid names) so that even after
data-cleaning drops a handful, we still have a large, tradable universe.
"""

FALLBACK_FNO_SYMBOLS = [
    # Nifty-50 / large caps
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT",
    "MARUTI", "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND",
    "ONGC", "NTPC", "POWERGRID", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ADANIENT",
    "ADANIPORTS", "COALINDIA", "GRASIM", "HINDALCO", "BAJAJFINSV", "BPCL", "BRITANNIA",
    "CIPLA", "DIVISLAB", "DRREDDY", "EICHERMOT", "HDFCLIFE", "HEROMOTOCO", "INDUSINDBK",
    "M&M", "SBILIFE", "SHRIRAMFIN", "TATACONSUM", "TECHM", "APOLLOHOSP", "LTIM",
    "BAJAJ-AUTO",
    # Banks / financials
    "BANKBARODA", "PNB", "CANBK", "FEDERALBNK", "IDFCFIRSTB", "AUBANK", "BANDHANBNK",
    "RBLBANK", "CHOLAFIN", "MUTHOOTFIN", "MANAPPURAM", "LICHSGFIN", "PFC", "RECLTD",
    "IEX", "BSE", "CDSL", "ANGELONE", "IIFL", "POONAWALLA", "ABCAPITAL", "SBICARD",
    "HDFCAMC", "ICICIGI", "ICICIPRULI", "LICI", "PAYTM", "POLICYBZR", "JIOFIN",
    # IT / tech / telecom
    "PERSISTENT", "COFORGE", "MPHASIS", "LTTS", "OFSS", "TATAELXSI", "KPITTECH",
    "IDEA", "INDUSTOWER", "NAUKRI",
    # Auto / ancillaries
    "TVSMOTOR", "ASHOKLEY", "BHARATFORG", "BOSCHLTD", "MOTHERSON", "BALKRISIND",
    "MRF", "APOLLOTYRE", "EXIDEIND", "TIINDIA", "ESCORTS", "SONACOMS",
    # Pharma / healthcare
    "AUROPHARMA", "LUPIN", "BIOCON", "TORNTPHARM", "ALKEM", "ZYDUSLIFE", "GLENMARK",
    "LAURUSLABS", "GRANULES", "IPCALAB", "SYNGENE", "MANKIND", "MAXHEALTH", "FORTIS",
    "PPLPHARMA", "ABBOTINDIA",
    # Metals / mining / energy
    "VEDL", "NMDC", "SAIL", "JINDALSTEL", "NATIONALUM", "HINDCOPPER", "GAIL",
    "IOC", "PETRONET", "IGL", "GUJGASLTD", "ATGL", "ADANIGREEN", "ADANIPOWER",
    "TATAPOWER", "TORNTPOWER", "NHPC", "SJVN", "OIL", "MGL", "CASTROLIND",
    # FMCG / consumer
    "DABUR", "MARICO", "GODREJCP", "COLPAL", "UBL", "UNITDSPR", "TATACONSUM",
    "VBL", "PGHH", "EMAMILTD", "RADICO", "PATANJALI", "JUBLFOOD", "DEVYANI",
    "TRENT", "PIDILITIND", "BERGEPAINT", "PAGEIND", "HAVELLS", "VOLTAS", "WHIRLPOOL",
    "CROMPTON", "DIXON", "BATAINDIA", "RELAXO", "TITAGARH",
    # Cement / infra / realty
    "SHREECEM", "AMBUJACEM", "ACC", "DALBHARAT", "RAMCOCEM", "JKCEMENT", "INDIACEM",
    "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "PHOENIXLTD", "LODHA", "NBCC",
    "GMRINFRA", "IRB", "NCC", "RVNL", "IRCON", "KEC", "CONCOR",
    # Chemicals / fertilisers / diversified
    "UPL", "PIIND", "SRF", "DEEPAKNTR", "AARTIIND", "NAVINFLUOR", "TATACHEM",
    "CHAMBLFERT", "COROMANDEL", "GNFC", "FACT", "ATUL", "VINATIORGA", "SUMICHEM",
    # PSU / capital goods / defence
    "BEL", "HAL", "BHEL", "SIEMENS", "ABB", "CUMMINSIND", "THERMAX", "POLYCAB",
    "KEI", "ASTRAL", "SUPREMEIND", "APLAPOLLO", "JSWENERGY", "BHARATELE", "MAZDOCK",
    "COCHINSHIP", "BEML", "IRFC", "IRCTC", "RAILTEL", "HUDCO", "MOIL",
    # Others / midcaps in F&O
    "INDHOTEL", "INDIGO", "ZOMATO", "NYKAA", "DELHIVERY", "GODREJIND", "CANFINHOME",
    "LTF", "BANKINDIA", "UNIONBANK", "YESBANK", "IDBI", "JSL", "WELCORP",
    "CGPOWER", "HONAUT", "SCHAEFFLER", "TIMKEN", "SKFINDIA", "GRINDWELL",
    "ABFRL", "PVRINOX", "SUNTV", "NETWORK18", "DMART", "BALRAMCHIN",
]

# de-dup while preserving order
FALLBACK_FNO_SYMBOLS = list(dict.fromkeys(FALLBACK_FNO_SYMBOLS))
