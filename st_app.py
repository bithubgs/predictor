import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import math
from scipy import stats
import warnings
import time # Added for auto-refresh in live mode
import pytz # Import pytz for timezone handling

warnings.filterwarnings('ignore')

# Georgian localization
st.set_page_config(
    page_title="კრიპტო ციკლების ანალიზი",
    page_icon="₿",
    layout="wide", # Keep wide layout for better use of space on larger screens
    initial_sidebar_state="expanded"
)

# Custom CSS for a more refined look and compactness
st.markdown("""
<style>
    /* General font size reduction for overall compactness */
    html, body, .stApp {
        font-size: 12px; /* Base font size */
    }
    /* Headers - slightly smaller for compactness */
    h1 {
        font-size: 2.0em; /* Smaller h1 */
        margin-bottom: 0.5em; /* Reduce space below headers */
    }
    h2 {
        font-size: 1.5em; /* Smaller h2 */
        margin-top: 1em; /* Adjust space above headers */
        margin-bottom: 0.5em;
    }
    h3 {
        font-size: 1.2em; /* Smaller h3 */
        margin-top: 0.8em;
        margin-bottom: 0.4em;
    }
    /* Streamlit specific elements for compactness */
    .st-emotion-cache-1r6n02x { /* Target for st.metric container */
        padding: 0.5em; /* Further reduced padding for metrics */
        margin-bottom: 0.5em; /* Further reduced margin below metrics */
        border-radius: 8px; /* Slightly rounded corners */
        background-color: white; /* Clean background */
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); /* Subtle shadow */
        min-width: 100px; /* Ensure a minimum width for metrics */
    }
    .st-emotion-cache-1r6n02x div[data-testid="stMetricValue"] {
        font-size: 1.3em; /* Adjusted metric value font size */
        font-weight: bold;
        color: #004d40; /* Dark teal for values */
    }
    .st-emotion-cache-1r6n02x div[data-testid="stMetricLabel"] {
        font-size: 0.7em; /* Adjusted metric label font size */
        color: #555555; /* Grey for labels */
    }
    .st-emotion-cache-1r6n02x div[data-testid="stMetricDelta"] {
        font-size: 0.8em; /* Delta font size */
    }
    /* Buttons styling for better fit and appearance */
    .stButton>button {
        background-color: #00796b; /* Teal button background */
        color: white;
        border-radius: 8px; /* Rounded corners */
        border: none;
        padding: 0.6em 1em; /* Reduced padding for buttons */
        font-size: 0.95em; /* Slightly smaller button text */
        transition: background-color 0.3s ease;
        width: 100%; /* Make buttons fill width */
        margin-bottom: 0.4em; /* Space between buttons */
    }
    .stButton>button:hover {
        background-color: #004d40; /* Darker teal on hover */
        color: white;
    }
    /* Input fields (selectbox, text_input, number_input) */
    .stTextInput>div>div>input, .stSelectbox>div>div>div>div, .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #cccccc;
        padding: 0.3em 0.8em; /* Adjusted padding */
        font-size: 0.7em; /* Smaller font in inputs */
    }
    /* Ensure selectbox selected value is clearly visible */
    div[data-testid="stSelectbox"] div[data-testid="stSelectboxSelectedValue"] {
        color: #333333 !important; /* Dark grey for better visibility */
        font-weight: bold; /* Make it bold */
        white-space: nowrap; /* Keep text on a single line */
        overflow: visible !important; /* Allow text to overflow if necessary */
        text-overflow: clip !important; /* Prevent ellipsis if overflow is visible */
        flex-grow: 1; /* Allow it to grow and take available space */
        min-width: 0; /* Allow it to shrink, but overflow: visible should prevent clipping */
        padding-right: 5px; /* Add some padding to prevent text from touching the arrow */
    }
    /* Adjust the button container of the selectbox to ensure space for the selected value and arrow */
    .stSelectbox > div > div > div[role="button"] {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 5px; /* Small gap between text and arrow */
        min-height: 2.5em; /* Ensure sufficient height */
        /* Remove overflow: hidden and text-overflow: ellipsis from here */
        /* Let the inner text element manage its own overflow if it's too long */
    }

    /* General text and markdown */
    .st-emotion-cache-13ln4gm { /* Streamlit markdown text */
        font-size: 0.9em;
    }
    /* Sidebar adjustments */
    .st-emotion-cache-vk330y { /* Sidebar header */
        font-size: 1.8em;
        color: #004d40;
    }
    .st-emotion-cache-10qj072 { /* Sidebar elements padding */
        padding-top: 0.4rem; /* Reduced padding */
        padding-bottom: 0.4rem;
    }
    /* Plotly chart container styling */
    .stPlotlyChart {
        padding: 10px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1em;
    }
    /* Adjust column padding for tighter layout */
    .st-emotion-cache-uf99v8 { /* Column container */
        padding-left: 0.4rem; /* Reduced horizontal padding */
        padding-right: 0.4rem;
    }
    /* Table/DataFrame styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden; /* Ensures rounded corners apply to content */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-size: 0.85em; /* Smaller font for table content */
    }
    .stDataFrame table th { /* Table headers */
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'live_mode' not in st.session_state:
    st.session_state.live_mode = False
if 'page' not in st.session_state: # Initialize page state for new button navigation
    st.session_state.page = "📈 ციკლების ანალიზი"

# Bitcoin Halving Dates
# Ensure halving dates are timezone-aware (UTC) from the start for consistency
HALVING_DATES = {
    'halving_1': datetime(2012, 11, 28, tzinfo=pytz.utc),
    'halving_2': datetime(2016, 7, 9, tzinfo=pytz.utc),
    'halving_3': datetime(2020, 5, 11, tzinfo=pytz.utc),
    'halving_4': datetime(2024, 4, 20, tzinfo=pytz.utc)  # Next estimated halving
}

# Helper functions
@st.cache_data(ttl=300)  # Cache for 5 minutes for live mode
def get_crypto_data(symbol, period="2y"):
    """Fetch cryptocurrency data from Yahoo Finance"""
    with st.spinner(f"მონაცემების ჩატვირთვა {symbol}-ისთვის..."):
        try:
            # Try with -USD suffix first
            ticker = yf.Ticker(f"{symbol}-USD")
            data = ticker.history(period=period)
            
            # If no data, try with -USDT
            if data.empty:
                ticker = yf.Ticker(f"{symbol}-USDT")
                data = ticker.history(period=period)
            
            # If still no data, try without suffix
            if data.empty:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
            # Ensure the index is timezone-aware (UTC) if it's not already
            if data is not None and not data.index.tz:
                data.index = data.index.tz_localize('UTC')
            return data if not data.empty else None
        except Exception as e:
            st.error(f"შეცდომა მონაცემების ჩატვირთვისას {symbol}: {e}")
            return None

@st.cache_data(ttl=60)  # Cache for 1 minute for live price
def get_current_price(symbol):
    """Get current price of cryptocurrency"""
    with st.spinner(f"მიმდინარე ფასის მიღება {symbol}-ისთვის..."):
        try:
            # Try with -USD suffix first
            ticker = yf.Ticker(f"{symbol}-USD")
            info = ticker.info
            price = info.get('regularMarketPrice', 0)
            
            # If no price, try with -USDT
            if price == 0:
                ticker = yf.Ticker(f"{symbol}-USDT")
                info = ticker.info
                price = info.get('regularMarketPrice', 0)
            
            # If still no price, try without suffix
            if price == 0:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                price = info.get('regularMarketPrice', 0)
                
            return price
        except:
            return 0

def get_halving_cycle_data(symbol):
    """Get data aligned to halving cycles"""
    data = get_crypto_data(symbol, "max")
    if data is None or symbol != 'BTC':
        return None
    
    cycles = {}
    for cycle_name, halving_date in HALVING_DATES.items():
        # Ensure start_date and end_date are timezone-aware (UTC)
        start_date = halving_date - timedelta(days=365)
        end_date = halving_date + timedelta(days=1095)  # 3 years after
        
        cycle_data = data[(data.index >= start_date) & (data.index <= end_date)]
        if len(cycle_data) > 0:
            # Normalize to days from halving
            cycle_data = cycle_data.copy()
            # Ensure halving_date is timezone-aware when calculating days_from_halving
            cycle_data['days_from_halving'] = (cycle_data.index - halving_date).days
            cycles[cycle_name] = cycle_data
    
    return cycles

def advanced_prediction_model(data, symbol, days_ahead, use_cycles=True):
    """
    Advanced prediction model using multiple techniques.
    Returns predicted price, min price, max price, confidence, and trend strength.
    """
    if data is None or len(data) < 30:
        # Return zeros for all outputs if data is insufficient
        return 0, 0, 0, 0, 0  # predicted_price, min_price, max_price, confidence, trend
    
    current_price = data['Close'].iloc[-1]
    
    # 1. Technical Analysis
    # Moving averages
    ma_7 = data['Close'].rolling(window=7).mean().iloc[-1]
    ma_21 = data['Close'].rolling(window=21).mean().iloc[-1]
    ma_50 = data['Close'].rolling(window=min(50, len(data))).mean().iloc[-1]
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    # Volatility (annualized)
    volatility = data['Close'].pct_change().std() * np.sqrt(252)
    
    # 2. Trend Analysis
    prices = data['Close'].values
    x = np.arange(len(prices))
    
    # Linear regression for trend (last 30 days for short-term trend)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[-30:], prices[-30:])
    
    # 3. Cycle Analysis for BTC
    cycle_factor = 1.0
    if symbol == 'BTC' and use_cycles:
        # Check position in halving cycle
        current_date = datetime.now(pytz.utc) # Make current_date timezone-aware
        last_halving = HALVING_DATES['halving_3']
        # next_halving = HALVING_DATES['halving_4'] # Not directly used in factor calculation
        
        days_since_halving = (current_date - last_halving).days
        cycle_position = days_since_halving / 1461  # Approximately 4-year cycle (365 * 4)
        
        # Cycle multiplier based on historical patterns (simplified)
        if cycle_position < 0.25:  # First year after halving (post-halving pump)
            cycle_factor = 1.2
        elif cycle_position < 0.5:  # Second year (peak bull)
            cycle_factor = 1.8
        elif cycle_position < 0.75:  # Third year (bear market)
            cycle_factor = 1.1 # Still some recovery/stability
        else:  # Fourth year (pre-halving accumulation)
            cycle_factor = 0.9 # Potential dip before next halving
    
    # 4. Prediction Calculation
    # Trend component
    trend_prediction = current_price + (slope * days_ahead)
    
    # Momentum component (influence of recent price change)
    momentum = (current_price - ma_21) / ma_21
    momentum_factor = 1 + (momentum * 0.1) # Small influence
    
    # RSI adjustment (mean reversion)
    rsi_factor = 1.0
    if rsi > 70:  # Overbought, likely to pull back
        rsi_factor = 0.95
    elif rsi < 30:  # Oversold, likely to bounce
        rsi_factor = 1.05
    
    # Time decay factor (uncertainty increases with time)
    time_factor = np.exp(-days_ahead / 365.0) # Exponential decay over a year
    
    # Final predicted price combining factors
    predicted_price = (trend_prediction * momentum_factor * rsi_factor * cycle_factor * time_factor +
                       current_price * (1 - time_factor))
    
    # Confidence calculation (higher volatility and longer days_ahead reduce confidence)
    confidence = max(0, min(100, 100 - (volatility * 100) - (days_ahead / 10)))
    
    # Trend strength (absolute slope relative to current price)
    trend_strength = abs(slope / current_price) * 100 if current_price != 0 else 0
    
    # Calculate Min and Max Price predictions based on confidence and volatility
    # A wider range for lower confidence or higher volatility
    # Use a base deviation, scaled by (100 - confidence) / 100
    base_deviation_factor = 0.05 # 5% base deviation for a 100% confident prediction
    
    # Adjust deviation based on confidence and volatility
    # Lower confidence or higher volatility means larger potential deviation
    # Max deviation could be capped to prevent unrealistic ranges
    deviation_percent = (base_deviation_factor + (volatility * 0.5)) * ((100 - confidence) / 100)
    
    min_price = predicted_price * (1 - deviation_percent)
    max_price = predicted_price * (1 + deviation_percent)

    # Ensure min_price is not negative
    min_price = max(0, min_price)
    
    return predicted_price, min_price, max_price, confidence, trend_strength

def create_cycle_overlay_chart(symbol):
    """Create Bitcoin-style cycle overlay chart"""
    if symbol != 'BTC':
        return create_enhanced_chart(symbol)
    
    cycles = get_halving_cycle_data(symbol)
    if not cycles:
        return create_enhanced_chart(symbol)
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (cycle_name, cycle_data) in enumerate(cycles.items()):
        if len(cycle_data) > 0:
            # Normalize price to halving day price
            # Ensure halving_date is timezone-aware for correct indexing
            halving_date_for_indexing = HALVING_DATES[cycle_name]
            halving_price_series = cycle_data[cycle_data['days_from_halving'] == 0]['Close']
            
            if len(halving_price_series) > 0:
                halving_price = halving_price_series.iloc[0]
                normalized_price = (cycle_data['Close'] / halving_price - 1) * 100
                
                fig.add_trace(go.Scatter(
                    x=cycle_data['days_from_halving'],
                    y=normalized_price,
                    mode='lines',
                    name=f'{cycle_name.replace("_", " ").title()}',
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.7
                ))
    
    # Add current cycle projection
    current_data = get_crypto_data(symbol, "2y")
    if current_data is not None:
        current_date = datetime.now(pytz.utc) # Make current_date timezone-aware
        last_halving = HALVING_DATES['halving_3']
        days_since_halving = (current_date - last_halving).days
        
        # Ensure current_data index is aligned for comparison
        halving_data = current_data[current_data.index <= last_halving]
        if len(halving_data) > 0:
            halving_price = halving_data['Close'].iloc[-1]
            current_change = (current_data['Close'].iloc[-1] / halving_price - 1) * 100
            
            fig.add_trace(go.Scatter(
                x=[days_since_halving],
                y=[current_change],
                mode='markers',
                name='Current Position',
                marker=dict(color='red', size=10, symbol='diamond'),
                text=f'Current: {current_change:.1f}%',
                textposition="top center"
            ))
    
    fig.update_layout(
        title=f'{symbol} - ჰალვინგის ციკლების შედარება',
        xaxis_title='დღეები ჰალვინგის შემდეგ',
        yaxis_title='ფასის ცვლილება ჰალვინგის ფასიდან (%)',
        height=600,
        showlegend=True,
        xaxis=dict(
            range=[-365, 1095],
            gridcolor='lightgray',
            zerolinecolor='gray'
        ),
        yaxis=dict(
            gridcolor='lightgray',
            zerolinecolor='gray'
        ),
        plot_bgcolor='white'
    )
    
    return fig

def create_enhanced_chart(symbol):
    """Create enhanced chart with predictions"""
    data = get_crypto_data(symbol, "6mo")
    if data is None:
        return None
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol}/USD - Enhanced Analysis', 'Volume & Indicators', 'Predictions'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price chart with candlesticks
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'].rolling(window=7).mean(),
            name='MA7',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'].rolling(window=21).mean(),
            name='MA21',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=rsi,
            name='RSI',
            line=dict(color='purple', width=1)
        ),
        row=2, col=1
    )
    
    # Predictions
    current_price = data['Close'].iloc[-1]
    future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=30, freq='D')
    predictions_prices = []
    
    for i, date in enumerate(future_dates):
        pred_price, min_price, max_price, confidence, trend = advanced_prediction_model(data, symbol, i+1)
        predictions_prices.append(pred_price)
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions_prices,
            name='Prediction',
            line=dict(color='red', dash='dash', width=2)
        ),
        row=1, col=1
    )
    
    # Confidence intervals (using a fixed 10% for chart visualization for simplicity)
    upper_bound = [p * 1.1 for p in predictions_prices]
    lower_bound = [p * 0.9 for p in predictions_prices]
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=upper_bound,
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence Interval',
            fillcolor='rgba(255,0,0,0.1)'
        ),
        row=1, col=1
    )
    
    fig.update_layout(
        title=f'{symbol}/USD - Advanced Trading Analysis',
        height=800,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def get_futures_signals(symbol):
    """Generate futures trading signals"""
    data = get_crypto_data(symbol, "1mo")
    if data is None:
        return []
    
    current_price = data['Close'].iloc[-1]
    signals = []
    
    # Predictions for different timeframes
    timeframes = [1, 3, 7, 14, 30]
    
    for days in timeframes:
        pred_price, min_price, max_price, confidence, trend = advanced_prediction_model(data, symbol, days)
        change_percent = (pred_price - current_price) / current_price * 100
        
        # Generate signal
        if change_percent > 2 and confidence > 60:
            signal = "LONG"
            color = "green"
        elif change_percent < -2 and confidence > 60:
            signal = "SHORT"
            color = "red"
        else:
            signal = "HOLD"
            color = "gray"
        
        signals.append({
            'timeframe': f'{days} დღე',
            'predicted_price': pred_price,
            'min_price': min_price,  # Added min_price
            'max_price': max_price,  # Added max_price
            'change_percent': change_percent,
            'confidence': confidence,
            'signal': signal,
            'color': color
        })
    
    return signals

# Sidebar navigation
with st.sidebar:
    st.header("🚀 ნავიგაცია")
    # Removed st.markdown(" ") for more compactness
    
    # Live mode toggle moved inside the sidebar block
    st.session_state.live_mode = st.checkbox("🔴 ლაივ რეჟიმი", value=st.session_state.live_mode)

    if st.session_state.live_mode:
        auto_refresh = st.slider("Auto Refresh (წამი)", 10, 300, 60)
        if st.button("🔄 განახლება"):
            st.rerun()
    
    st.markdown("---") # Separator
    
    # Navigation buttons with updated styling and emojis
    if st.button("📈 **ციკლების ანალიზი**", key="btn_cycles"):
        st.session_state.page = "📈 ციკლების ანალიზი"
    if st.button("🎯 **Futures სიგნალები**", key="btn_futures"):
        st.session_state.page = "🎯 Futures სიგნალები"
    if st.button("💼 **ჩემი პორტფოლიო**", key="btn_portfolio"):
        st.session_state.page = "💼 ჩემი პორტფოლიო"
    if st.button("📊 **ინფორმაცია**", key="btn_info"):
        st.session_state.page = "📊 ინფორმაცია"

# Main content
page = st.session_state.page # Use the page from session state

if page == "📈 ციკლების ანალიზი":
    st.title("📈 კრიპტოვალუტების ციკლების ანალიზი")
    
    if st.session_state.live_mode:
        st.success("� ლაივ რეჟიმი აქტიურია")
    
    # Crypto selector with search
    crypto_options = [
        'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'DOT', 'MATIC', 'AVAX',
        'SHIB', 'TRX', 'LINK', 'UNI', 'ATOM', 'LTC', 'ETC', 'BCH', 'XLM', 'ALGO',
        'VET', 'ICP', 'FIL', 'HBAR', 'AAVE', 'MANA', 'SAND', 'CRV', 'GRT', 'ENJ',
        'SUSHI', 'COMP', 'MKR', 'SNX', 'YFI', 'THETA', 'FTM', 'NEAR', 'LUNA',
        'CAKE', 'RUNE', 'KSM', 'EGLD', 'ZEC', 'DASH', 'XMR', 'FLOW', 'MINA',
        'ONE', 'CELO', 'BAT', 'ZRX', 'OMG', 'LRC', 'STORJ', 'REN', 'BAND',
        'PEPE', 'FLOKI', 'BONK', 'WIF', 'BOME', 'MYRO', 'POPCAT', 'NEIRO'
    ]
    
    # Changed column ratio to allow better stacking on small screens
    col1, col2 = st.columns(2) 
    with col1:
        selected_crypto = st.selectbox("აირჩიე კრიპტოვალუტა:", crypto_options)
    
    with col2:
        manual_crypto = st.text_input("ან შეიყვანეთ სიმბოლო:", placeholder="მაგ: PEPE")
    
    if manual_crypto:
        selected_crypto = manual_crypto.upper()
    
    if selected_crypto:
        # Create chart
        if selected_crypto == 'BTC':
            chart = create_cycle_overlay_chart(selected_crypto)
        else:
            chart = create_enhanced_chart(selected_crypto)
        
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        
        # Current price and metrics
        current_price = get_current_price(selected_crypto)
        if current_price > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label=f"{selected_crypto}/USD",
                    value=f"${current_price:.6f}" if current_price < 1 else f"${current_price:.2f}",
                    delta=None
                )
            
            # Additional metrics
            data = get_crypto_data(selected_crypto, "1mo")
            if data is not None:
                with col2:
                    volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                    st.metric("ვოლატილურობა", f"{volatility:.1f}%")
                
                with col3:
                    change_24h = (current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
                    st.metric("24h ცვლილება", f"{change_24h:.1f}%")

elif page == "🎯 Futures სიგნალები":
    st.title("🎯 Futures Trading სიგნალები")
    
    if st.session_state.live_mode:
        st.success("🔴 ლაივ რეჟიმი აქტიურია")
    
    # Crypto selector with manual input for Futures Signals
    crypto_options_futures = [
        'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'DOT', 'MATIC', 'AVAX',
        'SHIB', 'TRX', 'LINK', 'UNI', 'ATOM', 'LTC', 'ETC', 'BCH', 'XLM', 'ALGO',
        'PEPE', 'FLOKI', 'BONK', 'WIF', 'BOME', 'MYRO', 'POPCAT', 'NEIRO' # Added meme coins to futures options
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        selected_crypto = st.selectbox("აირჩიე კრიპტოვალუტა:", crypto_options_futures)
    
    with col2:
        manual_crypto = st.text_input("ან შეიყვანეთ სიმბოლო:", placeholder="მაგ: PEPE")
    
    if manual_crypto:
        selected_crypto = manual_crypto.upper()
    
    if selected_crypto:
        signals = get_futures_signals(selected_crypto)
        
        if signals:
            st.subheader("🎯 ტრეიდინგ სიგნალები")
            
            # Display signals with predicted min/max prices
            for signal in signals:
                st.markdown(f"**{signal['timeframe']}**")
                
                # Use columns for main signal, predicted price, min, and max
                col_sig, col_pred, col_min, col_max = st.columns(4)
                with col_sig:
                    st.metric(
                        label="სიგნალი",
                        value=signal['signal'],
                        delta=f"{signal['change_percent']:.1f}%"
                    )
                with col_pred:
                    st.metric(
                        label="პროგნოზი",
                        value=f"${signal['predicted_price']:.4f}" if signal['predicted_price'] < 1 else f"${signal['predicted_price']:.2f}"
                    )
                with col_min:
                    st.metric(
                        label="მინ. ფასი",
                        value=f"${signal['min_price']:.4f}" if signal['min_price'] < 1 else f"${signal['min_price']:.2f}"
                    )
                with col_max:
                    st.metric(
                        label="მაქს. ფასი",
                        value=f"${signal['max_price']:.4f}" if signal['max_price'] < 1 else f"${signal['max_price']:.2f}"
                    )
                
                # Confidence and visual indicator in a separate row/column for better alignment
                col_conf_status = st.columns([1, 3]) # Adjust ratio for confidence and status
                with col_conf_status[0]:
                    st.metric(
                        label="ნდობა",
                        value=f"{signal['confidence']:.0f}%"
                    )
                with col_conf_status[1]:
                    if signal['signal'] == 'LONG':
                        st.success("📈 LONG")
                    elif signal['signal'] == 'SHORT':
                        st.error("📉 SHORT")
                    else:
                        st.info("⏸️ HOLD")
                
                st.markdown("---") # Separator between signals
        
        # Risk management
        st.subheader("⚠️ რისკ მენეჯმენტი")
        
        col1, col2 = st.columns(2)
        with col1:
            leverage = st.slider("ლევერიჯი", 1, 20, 5)
        with col2:
            risk_percent = st.slider("რისკი (% პორტფოლიოდან)", 1, 10, 2)
        
        current_price = get_current_price(selected_crypto)
        if current_price > 0:
            st.write(f"**მიმდინარე ფასი:** ${current_price:.4f}")
            
            # Calculate position size
            account_balance = st.number_input("ანგარიშის ნაშთი ($)", min_value=100, value=1000)
            risk_amount = account_balance * (risk_percent / 100)
            
            st.write(f"**რისკის ოდენობა:** ${risk_amount:.2f}")
            st.write(f"**შემოთავაზებული პოზიციის ზომა:** {risk_amount * leverage:.2f} {selected_crypto}")

elif page == "💼 ჩემი პორტფოლიო":
    st.title("💼 ჩემი კრიპტო პორტფოლიო")
    
    # Add cryptocurrency to portfolio
    st.subheader("➕ კრიპტოვალუტის დამატება")
    
    col1, col2 = st.columns(2)
    
    with col1:
        crypto_options = [
            'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'DOT', 'MATIC', 'AVAX',
            'SHIB', 'TRX', 'LINK', 'UNI', 'ATOM', 'LTC', 'ETC', 'BCH', 'XLM', 'ALGO',
            'VET', 'ICP', 'FIL', 'HBAR', 'AAVE', 'MANA', 'SAND', 'CRV', 'GRT', 'ENJ',
            'SUSHI', 'COMP', 'MKR', 'SNX', 'YFI', 'THETA', 'FTM', 'NEAR', 'LUNA',
            'CAKE', 'RUNE', 'KSM', 'EGLD', 'ZEC', 'DASH', 'XMR', 'FLOW', 'MINA',
            'ONE', 'CELO', 'BAT', 'ZRX', 'OMG', 'LRC', 'STORJ', 'REN', 'BAND',
            'PEPE', 'FLOKI', 'BONK', 'WIF', 'BOME', 'MYRO', 'POPCAT', 'NEIRO'
        ]
        selected_crypto = st.selectbox("აირჩიე კრიპტოვალუტა:", crypto_options)
        
        # Manual input option
        manual_crypto = st.text_input("ან შეიყვანეთ სიმბოლო:", placeholder="მაგ: PEPE, FLOKI")
        
        # Use manual input if provided
        if manual_crypto:
            selected_crypto = manual_crypto.upper()
    
    with col2:
        amount = st.number_input("რაოდენობა:", min_value=0.0, step=0.1)
    
    if st.button("დამატება პორტფოლიოში"):
        if selected_crypto and amount > 0:
            if selected_crypto in st.session_state.portfolio:
                st.session_state.portfolio[selected_crypto] += amount
            else:
                st.session_state.portfolio[selected_crypto] = amount
            st.success(f"წარმატებით დაემატა {amount} {selected_crypto}")
            st.rerun()
    
    # Display portfolio
    if st.session_state.portfolio:
        st.subheader("📋 ჩემი პორტფოლიო")
        
        portfolio_data = []
        total_value = 0
        
        for crypto, amount in st.session_state.portfolio.items():
            current_price = get_current_price(crypto)
            total_crypto_value = amount * current_price
            total_value += total_crypto_value
            
            # Get advanced predictions
            data = get_crypto_data(crypto, "3mo")
            predictions = {}
            if data is not None:
                periods = [3, 7, 30, 90, 180, 365]
                for days in periods:
                    # Now getting min_price and max_price from the model
                    pred_price, min_price, max_price, confidence, trend = advanced_prediction_model(data, crypto, days)
                    predictions[days] = {
                        "price": pred_price,
                        "min_price": min_price,
                        "max_price": max_price,
                        "change": (pred_price - current_price) / current_price * 100
                    }
            
            portfolio_data.append({
                "კრიპტო": crypto,
                "რაოდენობა": f"{amount:.4f}",
                "მიმდინარე ფასი": f"${current_price:.6f}" if current_price < 1 else f"${current_price:.2f}",
                "ჯამური ღირებულება": f"${total_crypto_value:.2f}",
                "3 დღე (პროგნოზი)": f"${predictions.get(3, {}).get('price', 0):.6f}" if predictions.get(3, {}).get('price', 0) < 1 else f"${predictions.get(3, {}).get('price', 0):.2f} ({predictions.get(3, {}).get('change', 0):.1f}%)",
                "3 დღე (მინ)": f"${predictions.get(3, {}).get('min_price', 0):.6f}" if predictions.get(3, {}).get('min_price', 0) < 1 else f"${predictions.get(3, {}).get('min_price', 0):.2f}",
                "3 დღე (მაქს)": f"${predictions.get(3, {}).get('max_price', 0):.6f}" if predictions.get(3, {}).get('max_price', 0) < 1 else f"${predictions.get(3, {}).get('max_price', 0):.2f}",
                "1 კვირა (პროგნოზი)": f"${predictions.get(7, {}).get('price', 0):.6f}" if predictions.get(7, {}).get('price', 0) < 1 else f"${predictions.get(7, {}).get('price', 0):.2f} ({predictions.get(7, {}).get('change', 0):.1f}%)",
                "1 კვირა (მინ)": f"${predictions.get(7, {}).get('min_price', 0):.6f}" if predictions.get(7, {}).get('min_price', 0) < 1 else f"${predictions.get(7, {}).get('min_price', 0):.2f}",
                "1 კვირა (მაქს)": f"${predictions.get(7, {}).get('max_price', 0):.6f}" if predictions.get(7, {}).get('max_price', 0) < 1 else f"${predictions.get(7, {}).get('max_price', 0):.2f}",
                "1 თვე (პროგნოზი)": f"${predictions.get(30, {}).get('price', 0):.6f}" if predictions.get(30, {}).get('price', 0) < 1 else f"${predictions.get(30, {}).get('price', 0):.2f} ({predictions.get(30, {}).get('change', 0):.1f}%)",
                "1 თვე (მინ)": f"${predictions.get(30, {}).get('min_price', 0):.6f}" if predictions.get(30, {}).get('min_price', 0) < 1 else f"${predictions.get(30, {}).get('min_price', 0):.2f}",
                "1 თვე (მაქს)": f"${predictions.get(30, {}).get('max_price', 0):.6f}" if predictions.get(30, {}).get('max_price', 0) < 1 else f"${predictions.get(30, {}).get('max_price', 0):.2f}",
                "3 თვე (პროგნოზი)": f"${predictions.get(90, {}).get('price', 0):.6f}" if predictions.get(90, {}).get('price', 0) < 1 else f"${predictions.get(90, {}).get('price', 0):.2f} ({predictions.get(90, {}).get('change', 0):.1f}%)",
                "3 თვე (მინ)": f"${predictions.get(90, {}).get('min_price', 0):.6f}" if predictions.get(90, {}).get('min_price', 0) < 1 else f"${predictions.get(90, {}).get('min_price', 0):.2f}",
                "3 თვე (მაქს)": f"${predictions.get(90, {}).get('max_price', 0):.6f}" if predictions.get(90, {}).get('max_price', 0) < 1 else f"${predictions.get(90, {}).get('max_price', 0):.2f}",
                "6 თვე (პროგნოზი)": f"${predictions.get(180, {}).get('price', 0):.6f}" if predictions.get(180, {}).get('price', 0) < 1 else f"${predictions.get(180, {}).get('price', 0):.2f} ({predictions.get(180, {}).get('change', 0):.1f}%)",
                "6 თვე (მინ)": f"${predictions.get(180, {}).get('min_price', 0):.6f}" if predictions.get(180, {}).get('min_price', 0) < 1 else f"${predictions.get(180, {}).get('min_price', 0):.2f}",
                "6 თვე (მაქს)": f"${predictions.get(180, {}).get('max_price', 0):.6f}" if predictions.get(180, {}).get('max_price', 0) < 1 else f"${predictions.get(180, {}).get('max_price', 0):.2f}",
                "1 წელი (პროგნოზი)": f"${predictions.get(365, {}).get('price', 0):.6f}" if predictions.get(365, {}).get('price', 0) < 1 else f"${predictions.get(365, {}).get('price', 0):.2f} ({predictions.get(365, {}).get('change', 0):.1f}%)",
                "1 წელი (მინ)": f"${predictions.get(365, {}).get('min_price', 0):.6f}" if predictions.get(365, {}).get('min_price', 0) < 1 else f"${predictions.get(365, {}).get('min_price', 0):.2f}",
                "1 წელი (მაქს)": f"${predictions.get(365, {}).get('max_price', 0):.6f}" if predictions.get(365, {}).get('max_price', 0) < 1 else f"${predictions.get(365, {}).get('max_price', 0):.2f}"
            })
        
        df_portfolio = pd.DataFrame(portfolio_data)
        st.dataframe(df_portfolio, use_container_width=True)
        
        st.markdown(f"### 💰 პორტფოლიოს ჯამური ღირებულება: ${total_value:.2f}")
        
        # Remove from portfolio
        st.subheader("➖ კრიპტოვალუტის წაშლა")
        crypto_to_remove = st.selectbox("აირჩიე კრიპტოვალუტა წასაშლელად:", list(st.session_state.portfolio.keys()))
        if st.button("წაშლა პორტფოლიოდან"):
            if crypto_to_remove in st.session_state.portfolio:
                del st.session_state.portfolio[crypto_to_remove]
                st.success(f"{crypto_to_remove} წარმატებით წაიშალა პორტფოლიოდან.")
                st.rerun()
        
        st.subheader("🗑️ პორტფოლიოს გასუფთავება")
        if st.button("პორტფოლიოს გასუფთავება", help="წაშლის ყველა აქტივს პორტფოლიოდან"):
            st.session_state.portfolio = {}
            st.success("პორტფოლიო წარმატებით გასუფთავდა.")
            st.rerun()
    else:
        st.info("თქვენი პორტფოლიო ცარიელია. დაამატეთ კრიპტოვალუტები ზემოთ.")

elif page == "📊 ინფორმაცია":
    st.title("📊 ინფორმაცია აპლიკაციის შესახებ")
    st.markdown("""
    ეს აპლიკაცია შექმნილია კრიპტოვალუტების ბაზრის ანალიზისა და პროგნოზირებისთვის.
    
    **ძირითადი ფუნქციები:**
    
    * **ციკლების ანალიზი:** Bitcoin-ის ჰალვინგის ციკლების ისტორიული მონაცემების ვიზუალიზაცია და მიმდინარე ციკლის შედარება.
    * **გაუმჯობესებული ანალიზი:** სანთლების გრაფიკები, მოძრავი საშუალოები (MA7, MA21), მოცულობა და RSI ინდიკატორი.
    * **მოწინავე პროგნოზირება:** ფასების პროგნოზირება სხვადასხვა ვადებში, ნდობის დონით.
    * **Futures სიგნალები:** მოკლე და საშუალოვადიანი ტრეიდინგ სიგნალები (LONG/SHORT/HOLD) რისკ მენეჯმენტის რეკომენდაციებით.
    * **ჩემი პორტფოლიო:** თქვენი კრიპტო აქტივების მართვა, მიმდინარე ღირებულების და სამომავლო პროგნოზების ნახვა.
    
    **მონაცემთა წყარო:** Yahoo Finance (YFinance)
    
    ---
    
    **⚠️ მნიშვნელოვანი გაფრთხილება:**
    ეს აპლიკაცია განკუთვნილია მხოლოდ საგანმანათლებლო და საინფორმაციო მიზნებისთვის.
    კრიპტოვალუტებით ვაჭრობა მაღალ რისკებთან არის დაკავშირებული და შეიძლება გამოიწვიოს კაპიტალის სრული დაკარგვა.
    აპლიკაციაში მოცემული პროგნოზები ეფუძნება სტატისტიკურ მოდელებს და არ არის ფინანსური რჩევა.
    ნებისმიერი საინვესტიციო გადაწყვეტილების მიღებამდე ყოველთვის ჩაატარეთ საკუთარი კვლევა (DYOR - Do Your Own Research)
    და გაიარეთ კონსულტაცია ლიცენზირებულ ფინანსურ ექსპერტთან.
    
    ---
    
    **შექმნილია:** 2025
    """)

# Auto refresh for live mode
# Ensure this block is at the very end of the script to avoid re-running prematurely
if st.session_state.live_mode:
    time.sleep(auto_refresh)
    st.rerun()
