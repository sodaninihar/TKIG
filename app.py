"""
TKIG Portfolio Dashboard - Streamlit Version
============================================
A comprehensive portfolio management dashboard benchmarked to VTI.

To run:
    streamlit run app.py

To deploy (free):
    1. Push to GitHub
    2. Go to share.streamlit.io
    3. Connect your repo
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import functools
from pathlib import Path

# ============================================================================
# PAGE CONFIG (Must be first Streamlit command)
# ============================================================================

st.set_page_config(
    page_title="TKIG Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLING & THEME
# ============================================================================

COLORS = {
    'primary': '#0B5FFF',
    'accent1': '#2563EB',
    'accent2': '#1E40AF',
    'background': '#FAFAFA',
    'surface': '#FFFFFF',
    'text': '#0F172A',
    'text_muted': '#64748B',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'border': '#E2E8F0',
    'chart_green': '#10B981',
    'chart_red': '#EF4444',
    'chart_blue': '#3B82F6',
    'chart_purple': '#8B5CF6',
    'chart_orange': '#F97316'
}

# Custom CSS for professional look
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    .main {{
        background-color: {COLORS['background']};
    }}
    
    /* Professional metric cards */
    .stMetric {{
        background: linear-gradient(135deg, {COLORS['surface']} 0%, #F8FAFC 100%);
        padding: 24px 20px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
        border: 1px solid {COLORS['border']};
        transition: all 0.3s ease;
    }}
    
    .stMetric:hover {{
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }}
    
    .stMetric label {{
        color: {COLORS['text_muted']};
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }}
    
    .stMetric [data-testid="stMetricValue"] {{
        color: {COLORS['text']};
        font-size: 28px;
        font-weight: 700;
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {COLORS['surface']};
        padding: 8px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        background-color: transparent;
        border: none;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent1']} 100%);
        color: white !important;
    }}
    
    /* Headers */
    h1 {{
        color: {COLORS['text']};
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    
    h2, h3 {{
        color: {COLORS['text']};
        font-weight: 600;
        letter-spacing: -0.3px;
    }}
    
    /* Dataframes */
    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    
    /* Selectbox and inputs */
    .stSelectbox, .stNumberInput {{
        background: {COLORS['surface']};
        border-radius: 8px;
    }}
    
    /* Buttons */
    .stButton button {{
        border-radius: 8px;
        font-weight: 600;
        padding: 12px 24px;
        border: none;
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent1']} 100%);
        color: white;
        transition: all 0.3s ease;
    }}
    
    .stButton button:hover {{
        box-shadow: 0 4px 12px rgba(11, 95, 255, 0.3);
        transform: translateY(-2px);
    }}
    
    /* Cards */
    .metric-card {{
        background: {COLORS['surface']};
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid {COLORS['border']};
        margin-bottom: 16px;
    }}
    
    .risk-card {{
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid {COLORS['primary']};
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }}
    
    .alert-success {{
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border-left: 4px solid {COLORS['success']};
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
    }}
    
    .alert-warning {{
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border-left: 4px solid {COLORS['warning']};
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
    }}
    
    .alert-danger {{
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border-left: 4px solid {COLORS['danger']};
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
    }}
    
    /* Progress bars */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['accent1']} 100%);
    }}
</style>
""", unsafe_allow_html=True)

# Plotly template - professional styling
PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font={'family': 'Inter, sans-serif', 'size': 12, 'color': COLORS['text']},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        colorway=[COLORS['chart_blue'], COLORS['chart_green'], COLORS['chart_purple'], 
                 COLORS['chart_orange'], COLORS['primary'], COLORS['danger']],
        hovermode='x unified',
        xaxis={
            'gridcolor': COLORS['border'],
            'showgrid': True,
            'zeroline': False,
            'showline': True,
            'linewidth': 1,
            'linecolor': COLORS['border']
        },
        yaxis={
            'gridcolor': COLORS['border'],
            'showgrid': True,
            'zeroline': False,
            'showline': True,
            'linewidth': 1,
            'linecolor': COLORS['border']
        },
        title={'font': {'size': 16, 'color': COLORS['text'], 'family': 'Inter'}}
    )
)

# ============================================================================
# TICKER SYMBOL MAPPING
# ============================================================================

TICKER_MAP = {
    'BRKB': 'BRK-B',
    'BRK.B': 'BRK-B',
    'BRKA': 'BRK-A',
    'BRK.A': 'BRK-A',
    'BRK/B': 'BRK-B',
    'BRK/A': 'BRK-A'
}

def normalize_ticker(symbol: str) -> str:
    """Normalize ticker symbols for yfinance compatibility."""
    return TICKER_MAP.get(symbol.upper(), symbol.upper())

# ============================================================================
# DATA LOADERS WITH CACHING
# ============================================================================

@st.cache_data(ttl=300)  # 5 minute cache
def get_prices(symbols: List[str], period: str = '5y') -> pd.DataFrame:
    """Download historical adjusted close prices for symbols."""
    try:
        if len(symbols) == 0:
            return pd.DataFrame()
        
        # Download with proper error handling
        data = yf.download(symbols, period=period, progress=False, group_by='ticker', auto_adjust=True)
        
        if data.empty:
            return pd.DataFrame()
        
        if len(symbols) == 1:
            # Single ticker
            if 'Close' in data.columns:
                prices = data[['Close']].copy()
                prices.columns = symbols
            else:
                prices = data['Close'].to_frame()
                prices.columns = symbols
        else:
            # Multiple tickers
            prices = pd.DataFrame()
            for symbol in symbols:
                try:
                    if symbol in data.columns.get_level_values(0):
                        prices[symbol] = data[symbol]['Close']
                    elif 'Close' in data.columns:
                        # Fallback for different data structure
                        prices[symbol] = data['Close']
                except:
                    continue
        
        return prices.dropna(how='all')
    except Exception as e:
        st.error(f"Error downloading prices: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)  # 1 minute cache
def get_latest_quote(symbol: str) -> Dict:
    """Get latest quote data for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='2d')
        
        if len(hist) == 0:
            return {'price': None, 'change': None, 'change_pct': None, 'volume': None}
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        
        return {
            'price': current_price,
            'change': current_price - prev_close,
            'change_pct': ((current_price - prev_close) / prev_close * 100) if prev_close else 0,
            'volume': hist['Volume'].iloc[-1]
        }
    except Exception as e:
        return {'price': None, 'change': None, 'change_pct': None, 'volume': None}

@st.cache_data(ttl=86400)  # 24 hour cache
def get_basics(symbol: str) -> Dict:
    """Get basic information about a ticker."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'name': info.get('longName', info.get('shortName', symbol)),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0)
        }
    except Exception as e:
        return {'name': symbol, 'sector': 'Unknown', 'industry': 'Unknown', 'market_cap': 0}

@st.cache_data(ttl=86400)  # 24 hour cache
def get_spy_sector_allocation() -> Dict[str, float]:
    """Get SPY sector allocation (approximate, based on S&P 500)."""
    # Source: S&P 500 sector weights (approximate as of 2024)
    return {
        'Technology': 29.5,
        'Financial Services': 13.0,
        'Healthcare': 12.5,
        'Consumer Cyclical': 10.5,
        'Communication Services': 9.0,
        'Industrials': 8.5,
        'Consumer Defensive': 6.5,
        'Energy': 4.0,
        'Utilities': 2.5,
        'Real Estate': 2.5,
        'Basic Materials': 2.0,
        'Unknown': 0.0
    }

@st.cache_data(ttl=300)
def get_stock_info(symbol: str) -> Dict:
    """Get detailed stock information."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('forwardPE', info.get('trailingPE', 0)),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            '52w_high': info.get('fiftyTwoWeekHigh', 0),
            '52w_low': info.get('fiftyTwoWeekLow', 0),
            'avg_volume': info.get('averageVolume', 0),
            'description': info.get('longBusinessSummary', 'No description available')
        }
    except:
        return {
            'name': symbol,
            'sector': 'N/A',
            'industry': 'N/A',
            'market_cap': 0,
            'pe_ratio': 0,
            'dividend_yield': 0,
            'beta': 0,
            '52w_high': 0,
            '52w_low': 0,
            'avg_volume': 0,
            'description': 'Information not available'
        }

@st.cache_data(ttl=300)
def get_stock_price_history(symbol: str, period: str = '1y') -> pd.DataFrame:
    """Get price history for a stock."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        return hist
    except:
        return pd.DataFrame()

def _simple_sentiment(text: str) -> str:
    """Simple keyword-based sentiment analysis."""
    text = text.lower()
    positive_words = ['beat', 'surge', 'gain', 'rise', 'up', 'high', 'strong', 'growth', 'profit']
    negative_words = ['miss', 'fall', 'drop', 'down', 'low', 'weak', 'loss', 'cut', 'concern']
    
    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)
    
    if pos_count > neg_count:
        return 'Positive'
    elif neg_count > pos_count:
        return 'Negative'
    return 'Neutral'

# ============================================================================
# PORTFOLIO ENGINE
# ============================================================================

class PortfolioEngine:
    """Core portfolio analytics engine."""
    
    def __init__(self, holdings: pd.DataFrame, benchmark: str = 'VTI'):
        """Initialize portfolio engine."""
        self.holdings = holdings.copy()
        self.benchmark = benchmark
        self.symbols = holdings['Symbol'].tolist()
        self.all_symbols = self.symbols + [benchmark]
        
        # Load basics first
        self.basics = {symbol: get_basics(symbol) for symbol in self.symbols}
        
        # Load latest quotes
        self._refresh_quotes()
        
        # Load historical prices
        self.prices = get_prices(self.all_symbols, period='5y')
        
        # Verify benchmark loaded
        if self.benchmark not in self.prices.columns:
            st.warning(f"⚠️ Benchmark {self.benchmark} data not available. Downloading separately...")
            benchmark_prices = get_prices([self.benchmark], period='5y')
            if not benchmark_prices.empty:
                self.prices[self.benchmark] = benchmark_prices[self.benchmark]
            else:
                st.error(f"Failed to load benchmark {self.benchmark}")
                # Create dummy benchmark as fallback
                self.prices[self.benchmark] = pd.Series(100, index=self.prices.index)
        
        # Calculate portfolio metrics
        self._calculate_metrics()
        
        # Calculate portfolio beta
        self.portfolio_beta = self._calculate_portfolio_beta()
    
    def _refresh_quotes(self):
        """Refresh latest quotes for all holdings."""
        quotes = {symbol: get_latest_quote(symbol) for symbol in self.symbols}
        
        self.holdings['LastPrice'] = self.holdings['Symbol'].map(lambda s: quotes[s]['price'])
        self.holdings['DayChange'] = self.holdings['Symbol'].map(lambda s: quotes[s]['change'])
        self.holdings['DayChangePct'] = self.holdings['Symbol'].map(lambda s: quotes[s]['change_pct'])
        
        # Market value and P/L
        self.holdings['MarketValue'] = self.holdings['Qty'] * self.holdings['LastPrice']
        self.holdings['TotalPL'] = self.holdings['MarketValue'] - self.holdings['CostBasis']
        self.holdings['TotalPLPct'] = (self.holdings['TotalPL'] / self.holdings['CostBasis'] * 100)
        self.holdings['DayPL'] = self.holdings['Qty'] * self.holdings['DayChange']
        
        # Add basics
        self.holdings['Name'] = self.holdings['Symbol'].map(lambda s: self.basics.get(s, {}).get('name', s))
        self.holdings['Sector'] = self.holdings['Symbol'].map(lambda s: self.basics.get(s, {}).get('sector', 'Unknown'))
        self.holdings['Industry'] = self.holdings['Symbol'].map(lambda s: self.basics.get(s, {}).get('industry', 'Unknown'))
    
    def _calculate_metrics(self):
        """Calculate portfolio-level metrics."""
        self.total_market_value = self.holdings['MarketValue'].sum()
        self.total_cost_basis = self.holdings['CostBasis'].sum()
        self.total_pl = self.total_market_value - self.total_cost_basis
        self.total_pl_pct = (self.total_pl / self.total_cost_basis * 100) if self.total_cost_basis > 0 else 0
        self.total_day_pl = self.holdings['DayPL'].sum()
        self.total_day_pl_pct = (self.total_day_pl / (self.total_market_value - self.total_day_pl) * 100) if self.total_market_value > 0 else 0
        
        # Weights
        self.holdings['Weight'] = (self.holdings['MarketValue'] / self.total_market_value * 100)
        
        # Sector allocation
        self.sector_allocation = self.holdings.groupby('Sector')['Weight'].sum().sort_values(ascending=False)
    
    def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta vs benchmark."""
        try:
            port_ret, bench_ret = self.compute_returns()
            
            if len(port_ret) > 1 and len(bench_ret) > 1:
                # Calculate beta using covariance method
                covariance = np.cov(port_ret, bench_ret)[0, 1]
                bench_variance = np.var(bench_ret)
                beta = covariance / bench_variance if bench_variance > 0 else 1.0
                return beta
            else:
                return 1.0
        except Exception as e:
            st.warning(f"Could not calculate beta: {e}")
            return 1.0
    
    def compute_portfolio_timeseries(self, start_date: Optional[str] = None) -> pd.Series:
        """Compute historical portfolio value timeseries."""
        if start_date:
            prices = self.prices[self.prices.index >= start_date]
        else:
            prices = self.prices
        
        weights = self.holdings.set_index('Symbol')['Qty'].to_dict()
        portfolio_value = pd.Series(0, index=prices.index)
        
        for symbol in self.symbols:
            if symbol in prices.columns and symbol in weights:
                portfolio_value += prices[symbol] * weights[symbol]
        
        return portfolio_value.dropna()
    
    def compute_returns(self) -> Tuple[pd.Series, pd.Series]:
        """Compute portfolio and benchmark returns."""
        portfolio_ts = self.compute_portfolio_timeseries()
        portfolio_returns = portfolio_ts.pct_change().dropna()
        
        benchmark_returns = self.prices[self.benchmark].pct_change().dropna()
        
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        
        return portfolio_returns[common_dates], benchmark_returns[common_dates]
    
    def compute_performance_metrics(self, risk_free_rate: float = 0.0) -> Dict:
        """Compute comprehensive performance metrics."""
        port_ret, bench_ret = self.compute_returns()
        
        periods_per_year = 252
        
        # Returns
        port_ann_ret = (1 + port_ret.mean()) ** periods_per_year - 1
        bench_ann_ret = (1 + bench_ret.mean()) ** periods_per_year - 1
        
        # Volatility
        port_vol = port_ret.std() * np.sqrt(periods_per_year)
        bench_vol = bench_ret.std() * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        port_sharpe = (port_ann_ret - risk_free_rate) / port_vol if port_vol > 0 else 0
        bench_sharpe = (bench_ann_ret - risk_free_rate) / bench_vol if bench_vol > 0 else 0
        
        # Maximum drawdown
        port_ts = self.compute_portfolio_timeseries()
        port_cummax = port_ts.cummax()
        port_drawdown = (port_ts - port_cummax) / port_cummax
        port_max_dd = port_drawdown.min()
        
        bench_ts = self.prices[self.benchmark]
        bench_cummax = bench_ts.cummax()
        bench_drawdown = (bench_ts - bench_cummax) / bench_cummax
        bench_max_dd = bench_drawdown.min()
        
        # Tracking error and information ratio
        excess_returns = port_ret - bench_ret
        tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year) if excess_returns.std() > 0 else 0
        
        # Beta
        beta = self.portfolio_beta
        
        return {
            'portfolio_return': port_ann_ret,
            'benchmark_return': bench_ann_ret,
            'portfolio_volatility': port_vol,
            'benchmark_volatility': bench_vol,
            'portfolio_sharpe': port_sharpe,
            'benchmark_sharpe': bench_sharpe,
            'portfolio_max_drawdown': port_max_dd,
            'benchmark_max_drawdown': bench_max_dd,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta
        }
    
    def compute_attribution(self, period: str = 'YTD') -> pd.DataFrame:
        """Compute return attribution by ticker and sector."""
        end_date = datetime.now()
        if period == 'YTD':
            start_date = datetime(end_date.year, 1, 1)
        elif period == '1M':
            start_date = end_date - timedelta(days=30)
        elif period == '3M':
            start_date = end_date - timedelta(days=90)
        elif period == '6M':
            start_date = end_date - timedelta(days=180)
        elif period == '1Y':
            start_date = end_date - timedelta(days=365)
        else:
            start_date = self.prices.index.min()
        
        period_prices = self.prices[self.prices.index >= start_date]
        attribution = []
        
        for _, row in self.holdings.iterrows():
            symbol = row['Symbol']
            
            if symbol in period_prices.columns:
                price_start = period_prices[symbol].iloc[0]
                price_end = period_prices[symbol].iloc[-1]
                
                if pd.notna(price_start) and pd.notna(price_end) and price_start > 0:
                    ticker_return = (price_end - price_start) / price_start
                    weight = row['Weight'] / 100
                    contribution = ticker_return * weight
                    
                    attribution.append({
                        'Symbol': symbol,
                        'Sector': row['Sector'],
                        'Return': ticker_return * 100,
                        'Weight': row['Weight'],
                        'Contribution': contribution * 100
                    })
        
        return pd.DataFrame(attribution).sort_values('Contribution', ascending=False) if attribution else pd.DataFrame()
    
    def detect_anomalies(self) -> List[Dict]:
        """Detect unusual price movements and activity."""
        anomalies = []
        
        recent_prices = self.prices.tail(20)
        returns = recent_prices.pct_change().dropna()
        
        for symbol in self.symbols:
            if symbol not in returns.columns:
                continue
            
            symbol_returns = returns[symbol].dropna()
            
            if len(symbol_returns) < 5:
                continue
            
            latest_return = symbol_returns.iloc[-1]
            mean_return = symbol_returns.mean()
            std_return = symbol_returns.std()
            
            if std_return > 0:
                z_score = (latest_return - mean_return) / std_return
                
                if abs(z_score) > 2:
                    anomalies.append({
                        'symbol': symbol,
                        'type': 'Large Price Move',
                        'description': f"Latest return ({latest_return*100:.1f}%) is {abs(z_score):.1f} std devs from mean"
                    })
            
            if len(recent_prices[symbol].dropna()) >= 20:
                prices = recent_prices[symbol].dropna()
                sma = prices.rolling(20).mean()
                std = prices.rolling(20).std()
                upper_band = sma + (2 * std)
                lower_band = sma - (2 * std)
                
                current_price = prices.iloc[-1]
                
                if pd.notna(upper_band.iloc[-1]) and pd.notna(lower_band.iloc[-1]):
                    if current_price > upper_band.iloc[-1]:
                        anomalies.append({
                            'symbol': symbol,
                            'type': 'Bollinger Band Break',
                            'description': f"Price ${current_price:.2f} above upper band ${upper_band.iloc[-1]:.2f}"
                        })
                    elif current_price < lower_band.iloc[-1]:
                        anomalies.append({
                            'symbol': symbol,
                            'type': 'Bollinger Band Break',
                            'description': f"Price ${current_price:.2f} below lower band ${lower_band.iloc[-1]:.2f}"
                        })
        
        return anomalies

# ============================================================================
# DATA LOADING
# ============================================================================

def load_portfolio_from_csv(filepath: str) -> pd.DataFrame:
    """Load portfolio holdings from CSV file."""
    df = pd.read_csv(filepath)
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Map column variations
    column_map = {}
    for col in df.columns:
        if 'symbol' in col or 'ticker' in col:
            column_map[col] = 'Symbol'
        elif 'qty' in col or 'quantity' in col or 'shares' in col:
            column_map[col] = 'Qty'
        elif 'avgcost' in col or ('avg' in col and 'cost' in col):
            column_map[col] = 'AvgCost'
        elif 'costbasis' in col or ('cost' in col and 'basis' in col):
            column_map[col] = 'CostBasis'
    
    df = df.rename(columns=column_map)
    
    # Ensure required columns exist
    required = ['Symbol', 'Qty']
    if not all(col in df.columns for col in required):
        raise ValueError(f"CSV must contain: {required}")
    
    # Calculate missing cost fields
    if 'AvgCost' in df.columns and 'CostBasis' not in df.columns:
        df['CostBasis'] = df['Qty'] * df['AvgCost']
    elif 'CostBasis' in df.columns and 'AvgCost' not in df.columns:
        df['AvgCost'] = df['CostBasis'] / df['Qty']
    elif 'AvgCost' not in df.columns and 'CostBasis' not in df.columns:
        raise ValueError("Must provide either AvgCost or CostBasis")
    
    # Clean data
    df['Symbol'] = df['Symbol'].str.strip().str.upper().map(normalize_ticker)
    df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
    df['AvgCost'] = pd.to_numeric(df['AvgCost'], errors='coerce')
    df['CostBasis'] = pd.to_numeric(df['CostBasis'], errors='coerce')
    
    # Filter out zero quantity and VTI (benchmark)
    df = df[(df['Qty'] > 0) & (df['Symbol'] != 'VTI')].reset_index(drop=True)
    
    return df[['Symbol', 'Qty', 'AvgCost', 'CostBasis']]

# Default portfolio data - Updated with your actual holdings
DEFAULT_PORTFOLIO = """Symbol,Qty,AvgCost,CostBasis
AAPL,128,28.62,3663.40
ADBE,50,493.16,24658.00
ADI,126,191.90,24179.20
AMGN,108,300.38,32441.31
BLK,31,1001.37,31042.35
BRK-B,130,144.71,18811.89
EL,44,124.09,5459.94
EPD,844,26.12,22041.65
EVRG,276,58.75,16216.35
EXEL,822,40.37,33181.35
FRCB,65,128.85,8375.25
GOOGL,200,115.70,23139.10
HII,119,301.19,35841.36
HON,96,140.06,13445.97
JPM,100,296.84,29683.55
KO,724,63.17,45737.93
MA,86,159.21,13691.83
META,112,177.65,19896.96
MLM,65,193.54,12579.83
MSFT,164,69.69,11428.42
NKE,63,128.46,8093.29
ORLY,480,44.62,21416.00
PG,82,125.04,10252.92
PSX,75,118.45,8883.41
UNH,86,439.06,37759.52
"""

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    """Main Streamlit app."""
    
    # Header
    col_header1, col_header2 = st.columns([3, 1])
    
    with col_header1:
        st.markdown("""
        <h1 style='margin-bottom: 0;'>📊 TKIG Portfolio Dashboard</h1>
        <p style='color: #64748B; font-size: 14px; margin-top: 8px;'>
            <strong>Benchmarked to VTI</strong> • Last Updated: {}</p>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
    
    with col_header2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        st.markdown("---")
        
        # File upload with better UI
        st.markdown("### 📁 Portfolio Data")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'], help="Upload a CSV with columns: Symbol, Qty, AvgCost")
        
        if uploaded_file is not None:
            portfolio_df = load_portfolio_from_csv(uploaded_file)
            st.markdown(f"""
            <div class="alert-success">
                ✅ Loaded <strong>{len(portfolio_df)}</strong> positions
            </div>
            """, unsafe_allow_html=True)
        else:
            # Use default data
            import io
            portfolio_df = load_portfolio_from_csv(io.StringIO(DEFAULT_PORTFOLIO))
            st.markdown(f"""
            <div style="background: #F1F5F9; padding: 12px; border-radius: 8px; margin: 8px 0;">
                ℹ️ Using TKIG portfolio (Nov 2025)<br>
                <strong>{len(portfolio_df)}</strong> positions loaded
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick stats in sidebar
        with st.spinner("Loading portfolio data..."):
            engine = PortfolioEngine(portfolio_df)
        
        st.markdown("### 📈 Quick Stats")
        
        # Beta color coding
        if engine.portfolio_beta > 1.2:
            beta_color = '#EF4444'
            beta_label = 'High Vol'
        elif engine.portfolio_beta < 0.8:
            beta_color = '#10B981'
            beta_label = 'Defensive'
        else:
            beta_color = '#0B5FFF'
            beta_label = 'Market-Like'
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%); padding: 16px; border-radius: 8px; margin: 8px 0;">
            <div style="margin-bottom: 12px;">
                <span style="color: #64748B; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">Portfolio Value</span><br>
                <span style="color: #0F172A; font-size: 24px; font-weight: 700;">${engine.total_market_value:,.0f}</span>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="color: #64748B; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">Total Return</span><br>
                <span style="color: {'#10B981' if engine.total_pl >= 0 else '#EF4444'}; font-size: 20px; font-weight: 600;">
                    {engine.total_pl_pct:+.2f}%
                </span>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="color: #64748B; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">Day Change</span><br>
                <span style="color: {'#10B981' if engine.total_day_pl >= 0 else '#EF4444'}; font-size: 18px; font-weight: 600;">
                    ${engine.total_day_pl:+,.0f}
                </span>
            </div>
            <div style="background: rgba(11, 95, 255, 0.1); padding: 12px; border-radius: 6px; border-left: 4px solid {beta_color};">
                <span style="color: #64748B; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">⚡ Portfolio Beta</span><br>
                <span style="color: {beta_color}; font-size: 24px; font-weight: 700;">
                    {engine.portfolio_beta:.3f}
                </span><br>
                <span style="color: {beta_color}; font-size: 12px; font-weight: 600;">
                    {beta_label}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About")
        st.markdown("""
        <div style="font-size: 13px; color: #64748B; line-height: 1.6;">
            Professional portfolio analytics with institutional-grade metrics including VaR, CVaR, risk contribution analysis, and real-time market data.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div style="text-align: center; padding: 16px; background: linear-gradient(135deg, #0B5FFF 0%, #2563EB 100%); border-radius: 8px; color: white;">
            <strong>💎 Professional Edition</strong><br>
            <span style="font-size: 12px; opacity: 0.9;">CFA-Level Analytics</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview",
        "📋 Positions", 
        "🔍 Individual Holdings",
        "⚖️ Performance vs VTI",
        "🎯 Risk & Analytics",
        "🔮 What-If Analysis"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        # Executive Summary Banner
        perf_status = "Excellent" if engine.total_pl_pct > 10 else "Good" if engine.total_pl_pct > 0 else "Needs Attention"
        risk_level = "Low" if len(engine.holdings) >= 20 else "Moderate" if len(engine.holdings) >= 10 else "High"
        div_status = "Excellent" if len(engine.sector_allocation) >= 6 else "Good" if len(engine.sector_allocation) >= 4 else "Poor"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0B5FFF 0%, #2563EB 100%); padding: 28px; border-radius: 12px; margin-bottom: 24px; color: white;">
            <h2 style="color: white; margin: 0 0 16px 0; font-size: 24px;">📊 Portfolio Health Dashboard</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                <div style="background: rgba(255,255,255,0.15); padding: 16px; border-radius: 8px;">
                    <div style="font-size: 11px; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">Performance</div>
                    <div style="font-size: 22px; font-weight: 700;">{perf_status}</div>
                    <div style="font-size: 16px; margin-top: 4px;">{engine.total_pl_pct:+.2f}% Total Return</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 16px; border-radius: 8px;">
                    <div style="font-size: 11px; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">Concentration Risk</div>
                    <div style="font-size: 22px; font-weight: 700;">{risk_level}</div>
                    <div style="font-size: 16px; margin-top: 4px;">{len(engine.holdings)} Positions</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 16px; border-radius: 8px;">
                    <div style="font-size: 11px; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">Diversification</div>
                    <div style="font-size: 22px; font-weight: 700;">{div_status}</div>
                    <div style="font-size: 16px; margin-top: 4px;">{len(engine.sector_allocation)} Sectors</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # KPI Metrics Row with Beta
        st.markdown("### 📊 Key Portfolio Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Account Value", f"${engine.total_market_value:,.0f}", 
                     help="Total market value of all positions")
        
        with col2:
            st.metric("Total Return", f"{engine.total_pl_pct:+.2f}%",
                     f"${engine.total_pl:+,.0f}",
                     help="Total profit/loss")
        
        with col3:
            st.metric("Day Change", f"${engine.total_day_pl:+,.0f}",
                     f"{engine.total_day_pl_pct:+.2f}%",
                     help="Today's change")
        
        with col4:
            st.metric("# Positions", len(engine.holdings),
                     help="Number of individual holdings")
        
        with col5:
            beta_delta = "↑ Higher Vol" if engine.portfolio_beta > 1.0 else "↓ Lower Vol" if engine.portfolio_beta < 1.0 else "≈ Market"
            beta_delta_type = "normal" if 0.8 <= engine.portfolio_beta <= 1.2 else "inverse" if engine.portfolio_beta < 0.8 else "off"
            st.metric("**Portfolio Beta**", f"{engine.portfolio_beta:.3f}",
                     beta_delta,
                     delta_color=beta_delta_type,
                     help="Sensitivity to VTI. Beta=1.0 means moves with market. >1.0 = more volatile, <1.0 = less volatile")
        
        st.markdown("---")
        
        # === PORTFOLIO BETA CARD (PROMINENT) ===
        st.markdown("## 🎯 Portfolio Beta vs VTI")
        
        col_beta_main, col_beta_detail1, col_beta_detail2 = st.columns([2, 1, 1])
        
        with col_beta_main:
            # Determine beta status and color
            if engine.portfolio_beta > 1.2:
                beta_status = "⚠️ HIGH VOLATILITY"
                beta_color = COLORS['danger']
                beta_bg = "rgba(239, 68, 68, 0.1)"
                beta_message = f"Your portfolio is **{((engine.portfolio_beta - 1) * 100):.0f}% MORE volatile** than VTI. When VTI moves 1%, your portfolio typically moves {engine.portfolio_beta:.2f}%."
            elif engine.portfolio_beta < 0.8:
                beta_status = "🛡️ DEFENSIVE"
                beta_color = COLORS['success']
                beta_bg = "rgba(16, 185, 129, 0.1)"
                beta_message = f"Your portfolio is **{((1 - engine.portfolio_beta) * 100):.0f}% LESS volatile** than VTI. When VTI moves 1%, your portfolio typically moves {engine.portfolio_beta:.2f}%."
            else:
                beta_status = "⚖️ MARKET-ALIGNED"
                beta_color = COLORS['primary']
                beta_bg = "rgba(11, 95, 255, 0.1)"
                beta_message = f"Your portfolio moves **in line with the market** (VTI). When VTI moves 1%, your portfolio typically moves {engine.portfolio_beta:.2f}%."
            
            st.markdown(f"""
            <div style="background: {beta_bg}; padding: 24px; border-radius: 12px; border-left: 6px solid {beta_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                    <div>
                        <span style="font-size: 14px; color: {COLORS['text_muted']}; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Portfolio Beta</span>
                    </div>
                    <div>
                        <span style="font-size: 48px; font-weight: 700; color: {beta_color};">{engine.portfolio_beta:.3f}</span>
                    </div>
                </div>
                <div style="background: white; padding: 16px; border-radius: 8px; margin-bottom: 12px;">
                    <strong style="color: {beta_color}; font-size: 16px;">{beta_status}</strong>
                </div>
                <p style="color: {COLORS['text']}; font-size: 14px; line-height: 1.6; margin: 0;">
                    {beta_message}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_beta_detail1:
            st.markdown("**📊 Beta Scale**")
            st.markdown(f"""
            ```
            > 1.2   HIGH RISK
              1.0   MARKET
            < 0.8   DEFENSIVE
            
            You: {engine.portfolio_beta:.3f}
            ```
            """)
            
            # Visual indicator
            beta_position = min(max((engine.portfolio_beta - 0.5) / 1.5, 0), 1)
            st.progress(beta_position)
            st.caption("Position on risk spectrum")
        
        with col_beta_detail2:
            st.markdown("**🎲 Expected Moves**")
            st.markdown(f"""
            **If VTI moves:**
            - +1% → Portfolio: **{engine.portfolio_beta*1:+.2f}%**
            - -1% → Portfolio: **{engine.portfolio_beta*-1:+.2f}%**
            - +5% → Portfolio: **{engine.portfolio_beta*5:+.2f}%**
            - -5% → Portfolio: **{engine.portfolio_beta*-5:+.2f}%**
            - +10% → Portfolio: **{engine.portfolio_beta*10:+.2f}%**
            - -10% → Portfolio: **{engine.portfolio_beta*-10:+.2f}%**
            """)
        
        st.markdown("---")
        
        # === MAIN SECTION: SECTOR DEVIATION FROM SPY ===
        st.markdown("## Sector Analysis vs SPY")
        st.markdown("**Active sector bets relative to S&P 500 benchmark**")
        
        # Get SPY allocation
        spy_allocation = get_spy_sector_allocation()
        
        # Prepare comparison data
        all_sectors = set(list(engine.sector_allocation.index) + list(spy_allocation.keys()))
        
        comparison_data = []
        for sector in all_sectors:
            if sector == 'Unknown':
                continue
            portfolio_weight = engine.sector_allocation.get(sector, 0)
            spy_weight = spy_allocation.get(sector, 0)
            difference = portfolio_weight - spy_weight
            abs_difference = abs(difference)
            
            comparison_data.append({
                'Sector': sector,
                'Portfolio': portfolio_weight,
                'SPY': spy_weight,
                'Active Weight': difference,
                'Abs Deviation': abs_difference
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('Active Weight', ascending=False)
        
        # Create waterfall/deviation chart
        fig_deviation = go.Figure()
        
        colors = [COLORS['danger'] if x > 0 else COLORS['chart_blue'] for x in comparison_df['Active Weight']]
        
        fig_deviation.add_trace(go.Bar(
            x=comparison_df['Sector'],
            y=comparison_df['Active Weight'],
            marker_color=colors,
            text=comparison_df['Active Weight'].apply(lambda x: f"{x:+.1f}%"),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Active Weight: %{y:+.2f}%<br>Portfolio: %{customdata[0]:.1f}%<br>SPY: %{customdata[1]:.1f}%<extra></extra>',
            customdata=comparison_df[['Portfolio', 'SPY']].values
        ))
        
        # Add zero line
        fig_deviation.add_hline(y=0, line_dash="dash", line_color=COLORS['text_muted'], line_width=2)
        
        fig_deviation.update_layout(
            title="Active Sector Weights (Portfolio vs SPY)",
            xaxis_title="",
            yaxis_title="Active Weight (%)",
            template=PLOTLY_TEMPLATE,
            height=450,
            showlegend=False,
            annotations=[
                dict(
                    x=0.02, y=0.98,
                    xref='paper', yref='paper',
                    text='<b>Red = Overweight</b><br><b>Blue = Underweight</b>',
                    showarrow=False,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor=COLORS['border'],
                    borderwidth=1,
                    borderpad=8,
                    font=dict(size=11)
                )
            ]
        )
        
        st.plotly_chart(fig_deviation, use_container_width=True)
        
        # === SECTOR ANALYSIS METRICS ===
        col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
        
        # Calculate active share (sum of absolute deviations / 2)
        active_share = comparison_df['Abs Deviation'].sum() / 2
        
        # Tracking error contribution from sector bets
        overweight_risk = comparison_df[comparison_df['Active Weight'] > 5]['Active Weight'].sum()
        underweight_risk = abs(comparison_df[comparison_df['Active Weight'] < -5]['Active Weight'].sum())
        
        # Number of significant bets
        sig_overweight = len(comparison_df[comparison_df['Active Weight'] > 5])
        sig_underweight = len(comparison_df[comparison_df['Active Weight'] < -5])
        
        with col_metrics1:
            st.metric("Active Share", f"{active_share:.1f}%",
                     help="Degree of differentiation from SPY (sum of absolute deviations / 2)")
        
        with col_metrics2:
            st.metric("Overweight Risk", f"{overweight_risk:.1f}%",
                     help="Total overweight positions >5%")
        
        with col_metrics3:
            st.metric("Underweight Risk", f"{underweight_risk:.1f}%",
                     help="Total underweight positions <-5%")
        
        with col_metrics4:
            st.metric("Significant Bets", f"{sig_overweight + sig_underweight}",
                     help="Sectors with >5% deviation")
        
        st.markdown("---")
        
        # === POSITION-LEVEL ANALYSIS ===
        st.markdown("## Position Analysis")
        
        col_pos1, col_pos2 = st.columns(2)
        
        with col_pos1:
            st.markdown("### Top Day Movers")
            top_movers = engine.holdings.nlargest(5, 'DayChangePct')[['Symbol', 'Name', 'DayChangePct', 'DayPL', 'Weight']]
            
            for _, row in top_movers.iterrows():
                color = COLORS['success'] if row['DayChangePct'] > 0 else COLORS['danger']
                st.markdown(f"""
                <div style="background: white; padding: 14px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid {color}; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="font-size: 15px; color: {COLORS['text']};">{row['Symbol']}</strong> 
                            <span style="font-size: 11px; color: {COLORS['text_muted']};"> • {row['Weight']:.1f}% of portfolio</span><br>
                            <span style="font-size: 11px; color: {COLORS['text_muted']};">{row['Name'][:35]}</span>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 18px; font-weight: 700; color: {color};">{row['DayChangePct']:+.2f}%</div>
                            <div style="font-size: 12px; color: {COLORS['text_muted']};">${row['DayPL']:+,.0f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_pos2:
            st.markdown("### Concentration Analysis")
            
            # HHI (Herfindahl-Hirschman Index)
            weights_squared = (engine.holdings['Weight'] ** 2).sum()
            hhi = weights_squared
            
            # Effective number of positions
            effective_n = 10000 / hhi if hhi > 0 else 0
            
            st.metric("HHI (Concentration Index)", f"{hhi:.0f}",
                     help="Herfindahl-Hirschman Index: <1500 = diversified, >2500 = concentrated")
            
            st.metric("Effective # of Stocks", f"{effective_n:.1f}",
                     help="Diversification-weighted number of positions")
            
            # Top N concentration
            top5_weight = engine.holdings.nlargest(5, 'Weight')['Weight'].sum()
            top10_weight = engine.holdings.nlargest(10, 'Weight')['Weight'].sum()
            
            st.metric("Top 5 Concentration", f"{top5_weight:.1f}%")
            st.metric("Top 10 Concentration", f"{top10_weight:.1f}%")
            
            # Concentration warning
            if top5_weight > 50:
                st.warning("⚠️ High concentration risk in top 5 holdings")
            elif hhi > 2500:
                st.warning("⚠️ Portfolio shows high concentration (HHI > 2500)")
            else:
                st.success("✅ Well-diversified portfolio structure")
        
        st.markdown("---")
        
        # === PERFORMANCE DISTRIBUTION ===
        st.markdown("## Return Distribution Analysis")
        
        col_dist1, col_dist2, col_dist3 = st.columns(3)
        
        with col_dist1:
            # Winners vs Losers
            winners = len(engine.holdings[engine.holdings['TotalPLPct'] > 0])
            losers = len(engine.holdings[engine.holdings['TotalPLPct'] <= 0])
            win_rate = (winners / len(engine.holdings) * 100) if len(engine.holdings) > 0 else 0
            
            win_loss_fig = go.Figure(data=[go.Pie(
                labels=['Winners', 'Losers'],
                values=[winners, losers],
                hole=0.5,
                marker={'colors': [COLORS['success'], COLORS['danger']]},
                textinfo='label+value',
                textfont={'size': 13, 'color': 'white'}
            )])
            
            win_loss_fig.update_layout(
                title=f"Win Rate: {win_rate:.0f}%",
                template=PLOTLY_TEMPLATE,
                height=300,
                showlegend=False,
                annotations=[dict(text=f'{winners}/{len(engine.holdings)}', x=0.5, y=0.5, 
                                font_size=20, showarrow=False, font_color=COLORS['text'])]
            )
            st.plotly_chart(win_loss_fig, use_container_width=True)
        
        with col_dist2:
            # Return distribution histogram
            perf_fig = go.Figure()
            
            perf_fig.add_trace(go.Histogram(
                x=engine.holdings['TotalPLPct'],
                nbinsx=15,
                marker_color=COLORS['primary'],
                opacity=0.8,
                name='Positions'
            ))
            
            # Add mean line
            mean_return = engine.holdings['TotalPLPct'].mean()
            perf_fig.add_vline(x=mean_return, line_dash="dash", line_color=COLORS['danger'],
                              annotation_text=f"Mean: {mean_return:.1f}%", annotation_position="top")
            
            perf_fig.update_layout(
                title="Return Distribution",
                xaxis_title="Total Return (%)",
                yaxis_title="# of Positions",
                template=PLOTLY_TEMPLATE,
                height=300,
                showlegend=False
            )
            st.plotly_chart(perf_fig, use_container_width=True)
        
        with col_dist3:
            # Performance metrics
            avg_return = engine.holdings['TotalPLPct'].mean()
            median_return = engine.holdings['TotalPLPct'].median()
            best_return = engine.holdings['TotalPLPct'].max()
            worst_return = engine.holdings['TotalPLPct'].min()
            
            st.metric("Average Return", f"{avg_return:+.2f}%")
            st.metric("Median Return", f"{median_return:+.2f}%")
            st.metric("Best Performer", f"{best_return:+.2f}%")
            st.metric("Worst Performer", f"{worst_return:+.2f}%")
            
            # Return dispersion
            std_return = engine.holdings['TotalPLPct'].std()
            st.metric("Return Dispersion (StdDev)", f"{std_return:.2f}%",
                     help="Measures consistency of returns across positions")
    
    # TAB 2: POSITIONS
    with tab2:
        st.subheader("Holdings Detail")
        
        positions_data = engine.holdings[[
            'Symbol', 'Name', 'Sector', 'Qty', 'AvgCost', 'LastPrice',
            'DayChange', 'DayChangePct', 'TotalPL', 'TotalPLPct', 'Weight', 'MarketValue'
        ]].copy()
        
        # Format the dataframe
        positions_data['Qty'] = positions_data['Qty'].apply(lambda x: f"{x:,.0f}")
        positions_data['AvgCost'] = positions_data['AvgCost'].apply(lambda x: f"${x:,.2f}")
        positions_data['LastPrice'] = positions_data['LastPrice'].apply(lambda x: f"${x:,.2f}")
        positions_data['DayChange'] = positions_data['DayChange'].apply(lambda x: f"${x:+,.2f}")
        positions_data['DayChangePct'] = positions_data['DayChangePct'].apply(lambda x: f"{x:+.2f}%")
        positions_data['TotalPL'] = positions_data['TotalPL'].apply(lambda x: f"${x:+,.2f}")
        positions_data['TotalPLPct'] = positions_data['TotalPLPct'].apply(lambda x: f"{x:+.2f}%")
        positions_data['Weight'] = positions_data['Weight'].apply(lambda x: f"{x:.2f}%")
        positions_data['MarketValue'] = positions_data['MarketValue'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(positions_data, hide_index=True, use_container_width=True, height=600)
    
    # TAB 3: INDIVIDUAL HOLDINGS
    with tab3:
        st.subheader("🔍 Individual Stock Analysis")
        
        # Stock selector
        selected_symbol = st.selectbox(
            "Select a stock from your portfolio",
            options=engine.holdings['Symbol'].tolist(),
            index=0
        )
        
        # Get stock data
        holding_info = engine.holdings[engine.holdings['Symbol'] == selected_symbol].iloc[0]
        stock_info = get_stock_info(selected_symbol)
        
        # Header with stock name
        st.markdown(f"## {stock_info['name']} ({selected_symbol})")
        st.markdown(f"**{stock_info['sector']} • {stock_info['industry']}**")
        
        # Key metrics row
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Your Shares", f"{holding_info['Qty']:.0f}")
        
        with col2:
            st.metric("Avg Cost", f"${holding_info['AvgCost']:.2f}")
        
        with col3:
            st.metric("Current Price", f"${holding_info['LastPrice']:.2f}",
                     f"{holding_info['DayChangePct']:+.2f}%")
        
        with col4:
            st.metric("Market Value", f"${holding_info['MarketValue']:,.0f}")
        
        with col5:
            st.metric("Total P/L", f"${holding_info['TotalPL']:+,.0f}",
                     f"{holding_info['TotalPLPct']:+.2f}%")
        
        with col6:
            st.metric("Portfolio Weight", f"{holding_info['Weight']:.2f}%")
        
        st.markdown("---")
        
        # Price chart with timeframe selector
        col_chart, col_info = st.columns([2, 1])
        
        with col_chart:
            st.subheader("📊 Price Chart")
            
            # Timeframe selector
            timeframe = st.selectbox(
                "Timeframe",
                options=['1D', '5D', '1M', '3M', '6M', '1Y', '2Y', '5Y', 'MAX'],
                index=5,  # Default to 1Y
                key=f"timeframe_{selected_symbol}"
            )
            
            # Map timeframes to yfinance periods
            period_map = {
                '1D': '1d',
                '5D': '5d',
                '1M': '1mo',
                '3M': '3mo',
                '6M': '6mo',
                '1Y': '1y',
                '2Y': '2y',
                '5Y': '5y',
                'MAX': 'max'
            }
            
            # Get price history
            with st.spinner(f"Loading {timeframe} data..."):
                price_history = get_stock_price_history(selected_symbol, period_map[timeframe])
            
            if not price_history.empty:
                # Create candlestick chart
                fig = go.Figure()
                
                # Add candlestick
                fig.add_trace(go.Candlestick(
                    x=price_history.index,
                    open=price_history['Open'],
                    high=price_history['High'],
                    low=price_history['Low'],
                    close=price_history['Close'],
                    name='Price'
                ))
                
                # Add volume bars
                fig.add_trace(go.Bar(
                    x=price_history.index,
                    y=price_history['Volume'],
                    name='Volume',
                    yaxis='y2',
                    marker_color=COLORS['text_muted'],
                    opacity=0.3
                ))
                
                # Add your average cost line
                fig.add_hline(
                    y=holding_info['AvgCost'],
                    line_dash="dash",
                    line_color=COLORS['warning'],
                    annotation_text=f"Your Avg Cost: ${holding_info['AvgCost']:.2f}",
                    annotation_position="right"
                )
                
                # Update layout
                fig.update_layout(
                    template=PLOTLY_TEMPLATE,
                    height=500,
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    yaxis2=dict(
                        title="Volume",
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    hovermode='x unified',
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Price statistics
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                
                current_price = price_history['Close'].iloc[-1]
                period_high = price_history['High'].max()
                period_low = price_history['Low'].min()
                period_return = ((current_price - price_history['Close'].iloc[0]) / price_history['Close'].iloc[0]) * 100
                
                with col_stats1:
                    st.metric(f"{timeframe} Return", f"{period_return:+.2f}%")
                
                with col_stats2:
                    st.metric(f"{timeframe} High", f"${period_high:.2f}")
                
                with col_stats3:
                    st.metric(f"{timeframe} Low", f"${period_low:.2f}")
                
                with col_stats4:
                    st.metric("Volatility", f"{price_history['Close'].pct_change().std() * 100:.2f}%")
            
            else:
                st.warning(f"No price data available for {selected_symbol}")
        
        with col_info:
            st.subheader("📋 Company Info")
            
            # Format market cap
            if stock_info['market_cap'] > 1e12:
                market_cap_str = f"${stock_info['market_cap']/1e12:.2f}T"
            elif stock_info['market_cap'] > 1e9:
                market_cap_str = f"${stock_info['market_cap']/1e9:.2f}B"
            elif stock_info['market_cap'] > 1e6:
                market_cap_str = f"${stock_info['market_cap']/1e6:.2f}M"
            else:
                market_cap_str = f"${stock_info['market_cap']:,.0f}"
            
            # Format metrics properly
            pe_ratio_str = f"{stock_info['pe_ratio']:.2f}" if stock_info['pe_ratio'] else 'N/A'
            div_yield_str = f"{stock_info['dividend_yield']*100:.2f}%" if stock_info['dividend_yield'] else 'N/A'
            beta_str = f"{stock_info['beta']:.2f}" if stock_info['beta'] else 'N/A'
            high_52w_str = f"${stock_info['52w_high']:.2f}" if stock_info['52w_high'] else 'N/A'
            low_52w_str = f"${stock_info['52w_low']:.2f}" if stock_info['52w_low'] else 'N/A'
            avg_vol_str = f"{stock_info['avg_volume']:,}" if stock_info['avg_volume'] else 'N/A'
            
            st.markdown(f"""
            **Market Cap:** {market_cap_str}
            
            **P/E Ratio:** {pe_ratio_str}
            
            **Dividend Yield:** {div_yield_str}
            
            **Beta:** {beta_str}
            
            **52W High:** {high_52w_str}
            
            **52W Low:** {low_52w_str}
            
            **Avg Volume:** {avg_vol_str}
            """)
            
            # Price vs 52-week range indicator
            if stock_info['52w_high'] and stock_info['52w_low']:
                price_range = stock_info['52w_high'] - stock_info['52w_low']
                price_position = (holding_info['LastPrice'] - stock_info['52w_low']) / price_range if price_range > 0 else 0
                
                st.markdown("**52-Week Range Position:**")
                st.progress(price_position)
                st.caption(f"{price_position*100:.1f}% from low to high")
        
        # Company description
        st.markdown("---")
        st.subheader("📖 About the Company")
        st.write(stock_info['description'])
    
    # TAB 4: PERFORMANCE
    with tab4:
        st.subheader("Performance vs VTI")
        
        # Check if benchmark data is available
        if engine.benchmark not in engine.prices.columns:
            st.error(f"Benchmark {engine.benchmark} data not available. Cannot display performance comparison.")
        else:
            # Cumulative return chart
            port_ts = engine.compute_portfolio_timeseries()
            bench_ts = engine.prices[engine.benchmark]
            
            port_norm = port_ts / port_ts.iloc[0] * 100
            bench_norm = bench_ts / bench_ts.iloc[0] * 100
            
            cum_return_fig = go.Figure()
            cum_return_fig.add_trace(go.Scatter(x=port_norm.index, y=port_norm.values, name='Portfolio',
                                               line={'color': COLORS['primary'], 'width': 2}))
            cum_return_fig.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm.values, name='VTI',
                                               line={'color': COLORS['text_muted'], 'width': 2, 'dash': 'dash'}))
            cum_return_fig.update_layout(
                title="Cumulative Return (Normalized to 100)",
                xaxis_title="Date",
                yaxis_title="Value",
                template=PLOTLY_TEMPLATE,
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(cum_return_fig, use_container_width=True)
            
            # Performance metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Performance Metrics")
                metrics = engine.compute_performance_metrics()
                
                metrics_df = pd.DataFrame([
                    {'Metric': 'Annualized Return', 'Portfolio': f"{metrics['portfolio_return']*100:.2f}%", 
                     'VTI': f"{metrics['benchmark_return']*100:.2f}%"},
                    {'Metric': 'Volatility', 'Portfolio': f"{metrics['portfolio_volatility']*100:.2f}%", 
                     'VTI': f"{metrics['benchmark_volatility']*100:.2f}%"},
                    {'Metric': 'Sharpe Ratio', 'Portfolio': f"{metrics['portfolio_sharpe']:.2f}", 
                     'VTI': f"{metrics['benchmark_sharpe']:.2f}"},
                    {'Metric': 'Max Drawdown', 'Portfolio': f"{metrics['portfolio_max_drawdown']*100:.2f}%", 
                     'VTI': f"{metrics['benchmark_max_drawdown']*100:.2f}%"},
                    {'Metric': 'Tracking Error', 'Portfolio': f"{metrics['tracking_error']*100:.2f}%", 'VTI': '-'},
                    {'Metric': 'Information Ratio', 'Portfolio': f"{metrics['information_ratio']:.2f}", 'VTI': '-'},
                    {'Metric': 'Beta vs VTI', 'Portfolio': f"{metrics['beta']:.2f}", 'VTI': '1.00'}
                ])
                
                st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
            with col2:
                st.subheader("Rolling 60-Day Excess Return")
                port_ret, bench_ret = engine.compute_returns()
                excess_ret = (port_ret - bench_ret).rolling(60).mean() * 100
                
                excess_fig = go.Figure()
                excess_fig.add_trace(go.Scatter(
                    x=excess_ret.index,
                    y=excess_ret.values,
                    name='60-Day Excess Return',
                    fill='tozeroy',
                    line={'color': COLORS['primary']}
                ))
                excess_fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Excess Return (%)",
                    template=PLOTLY_TEMPLATE,
                    height=350
                )
                st.plotly_chart(excess_fig, use_container_width=True)
            
            # Attribution
            st.subheader("YTD Return Attribution by Ticker")
            attribution = engine.compute_attribution('YTD')
            
            if not attribution.empty:
                attr_fig = go.Figure()
                attr_fig.add_trace(go.Bar(
                    x=attribution['Symbol'],
                    y=attribution['Contribution'],
                    marker_color=[COLORS['success'] if x > 0 else COLORS['danger'] for x in attribution['Contribution']]
                ))
                attr_fig.update_layout(
                    xaxis_title="Symbol",
                    yaxis_title="Contribution to Return (%)",
                    template=PLOTLY_TEMPLATE,
                    height=350
                )
                st.plotly_chart(attr_fig, use_container_width=True)
    
    # TAB 5: RISK & ANALYTICS
    with tab5:
        st.markdown("## 🎯 Risk & Analytics")
        st.markdown("**Professional risk assessment and portfolio analytics**")
        st.markdown("---")
        
        # Get returns for calculations
        port_ret, bench_ret = engine.compute_returns()
        
        # === SECTION 1: RISK METRICS DASHBOARD ===
        st.subheader("📊 Risk Metrics Dashboard")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Calculate advanced metrics
        portfolio_vol = port_ret.std() * np.sqrt(252) * 100
        var_95 = np.percentile(port_ret, 5) * 100
        cvar_95 = port_ret[port_ret <= np.percentile(port_ret, 5)].mean() * 100
        skewness = stats.skew(port_ret)
        kurtosis = stats.kurtosis(port_ret)
        
        with col1:
            st.metric("Annualized Volatility", f"{portfolio_vol:.2f}%",
                     help="Standard deviation of returns (annualized)")
        
        with col2:
            st.metric("Value at Risk (95%)", f"{var_95:.2f}%",
                     help="Maximum expected loss in a day with 95% confidence")
        
        with col3:
            st.metric("CVaR (95%)", f"{cvar_95:.2f}%",
                     help="Expected loss when VaR threshold is breached")
        
        with col4:
            st.metric("Skewness", f"{skewness:.3f}",
                     delta="Positive" if skewness > 0 else "Negative",
                     help="Asymmetry of return distribution")
        
        with col5:
            st.metric("Excess Kurtosis", f"{kurtosis:.3f}",
                     delta="Fat tails" if kurtosis > 0 else "Thin tails",
                     help="Tail risk indicator (excess over normal distribution)")
        
        st.markdown("---")
        
        # === SECTION 2: ANOMALY DETECTION ===
        col_anomaly, col_stats = st.columns([1, 1])
        
        with col_anomaly:
            st.subheader("🚨 Anomaly Detection")
            anomalies = engine.detect_anomalies()
            
            if anomalies:
                st.markdown(f"""
                <div class="alert-warning">
                    <strong>⚠️ {len(anomalies)} Anomalies Detected</strong><br>
                    Positions showing unusual price movements or technical signals
                </div>
                """, unsafe_allow_html=True)
                
                for anom in anomalies:
                    st.markdown(f"""
                    <div class="risk-card">
                        <strong style="color: {COLORS['primary']}; font-size: 15px;">{anom['symbol']}</strong><br>
                        <span style="color: {COLORS['warning']}; font-weight: 600;">{anom['type']}</span><br>
                        <span style="color: {COLORS['text_muted']}; font-size: 13px;">{anom['description']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-success">
                    <strong>✅ No Anomalies Detected</strong><br>
                    All positions are trading within normal parameters
                </div>
                """, unsafe_allow_html=True)
        
        with col_stats:
            st.subheader("📈 Return Distribution Analysis")
            
            # Return distribution histogram
            hist_fig = go.Figure()
            hist_fig.add_trace(go.Histogram(
                x=port_ret * 100,
                nbinsx=50,
                name='Returns',
                marker_color=COLORS['chart_blue'],
                opacity=0.7
            ))
            
            # Add normal distribution overlay
            mu = port_ret.mean() * 100
            sigma = port_ret.std() * 100
            x_range = np.linspace(port_ret.min() * 100, port_ret.max() * 100, 100)
            normal_dist = stats.norm.pdf(x_range, mu, sigma) * len(port_ret) * (port_ret.max() - port_ret.min()) * 100 / 50
            
            hist_fig.add_trace(go.Scatter(
                x=x_range,
                y=normal_dist,
                mode='lines',
                name='Normal Distribution',
                line={'color': COLORS['danger'], 'width': 2, 'dash': 'dash'}
            ))
            
            hist_fig.update_layout(
                title="Daily Returns Distribution vs Normal",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                template=PLOTLY_TEMPLATE,
                height=350,
                showlegend=True,
                barmode='overlay'
            )
            st.plotly_chart(hist_fig, use_container_width=True)
        
        st.markdown("---")
        
        # === SECTION 3: DRAWDOWN ANALYSIS ===
        col_dd1, col_dd2 = st.columns([2, 1])
        
        with col_dd1:
            st.subheader("📉 Drawdown Analysis")
            
            port_ts = engine.compute_portfolio_timeseries()
            cummax = port_ts.cummax()
            drawdown = (port_ts - cummax) / cummax * 100
            
            # Create drawdown figure with underwater plot
            dd_fig = go.Figure()
            
            dd_fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill='tozeroy',
                name='Drawdown',
                line={'color': COLORS['danger'], 'width': 0},
                fillcolor=f'rgba(239, 68, 68, 0.3)'
            ))
            
            dd_fig.update_layout(
                title="Portfolio Drawdown (Underwater Chart)",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                template=PLOTLY_TEMPLATE,
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(dd_fig, use_container_width=True)
        
        with col_dd2:
            st.subheader("📊 Drawdown Statistics")
            
            # Calculate drawdown metrics
            max_dd = drawdown.min()
            current_dd = drawdown.iloc[-1]
            
            # Find drawdown periods
            is_drawdown = drawdown < -1  # More than 1% drawdown
            dd_starts = is_drawdown & ~is_drawdown.shift(1, fill_value=False)
            dd_ends = ~is_drawdown & is_drawdown.shift(1, fill_value=False)
            
            # Count drawdown episodes
            num_drawdowns = dd_starts.sum()
            
            # Average drawdown duration
            if num_drawdowns > 0:
                avg_duration = len(drawdown[is_drawdown]) / num_drawdowns
            else:
                avg_duration = 0
            
            st.metric("Maximum Drawdown", f"{max_dd:.2f}%")
            st.metric("Current Drawdown", f"{current_dd:.2f}%")
            st.metric("Drawdown Episodes", f"{num_drawdowns}")
            st.metric("Avg Duration", f"{avg_duration:.0f} days")
            
            # Recovery metrics
            if current_dd < -1:
                st.markdown("""
                <div class="alert-warning">
                    <strong>⚠️ In Drawdown</strong><br>
                    Portfolio is {:.2f}% below peak
                </div>
                """.format(abs(current_dd)), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-success">
                    <strong>✅ At/Near Peak</strong><br>
                    Portfolio is at or near all-time high
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # === SECTION 4: CORRELATION & RISK DECOMPOSITION ===
        col_corr, col_contrib = st.columns([1, 1])
        
        with col_corr:
            st.subheader("🔗 Correlation Matrix")
            
            # Calculate correlations
            returns = engine.prices[engine.symbols].pct_change().dropna()
            corr_matrix = returns.corr()
            
            # Create enhanced heatmap
            heatmap_fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale=[
                    [0, COLORS['danger']],
                    [0.5, '#FFFFFF'],
                    [1, COLORS['chart_green']]
                ],
                zmid=0,
                zmin=-1,
                zmax=1,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 8},
                colorbar=dict(title="Correlation")
            ))
            
            heatmap_fig.update_layout(
                title="Asset Correlation Matrix",
                template=PLOTLY_TEMPLATE,
                height=500,
                xaxis={'side': 'bottom'},
                yaxis={'autorange': 'reversed'}
            )
            
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Correlation insights
            avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            st.info(f"**Average Correlation:** {avg_corr:.3f}")
            
            if avg_corr > 0.7:
                st.warning("⚠️ High average correlation - portfolio may lack diversification")
            elif avg_corr > 0.4:
                st.success("✅ Moderate correlation - good diversification balance")
            else:
                st.success("✅ Low correlation - excellent diversification")
        
        with col_contrib:
            st.subheader("⚖️ Risk Contribution Analysis")
            
            # Calculate marginal risk contribution
            weights = engine.holdings.set_index('Symbol')['Weight'].to_dict()
            weight_array = np.array([weights.get(s, 0)/100 for s in engine.symbols if s in returns.columns])
            
            cov_matrix = returns[engine.symbols].cov() * 252  # Annualized
            
            # Portfolio variance
            port_variance = np.dot(weight_array, np.dot(cov_matrix.values, weight_array))
            port_vol_calc = np.sqrt(port_variance)
            
            # Marginal risk contribution
            marginal_contrib = np.dot(cov_matrix.values, weight_array) / port_vol_calc
            risk_contrib = weight_array * marginal_contrib
            risk_contrib_pct = (risk_contrib / risk_contrib.sum()) * 100
            
            # Create dataframe
            risk_df = pd.DataFrame({
                'Symbol': [s for s in engine.symbols if s in returns.columns],
                'Weight (%)': weight_array * 100,
                'Risk Contrib (%)': risk_contrib_pct
            }).sort_values('Risk Contrib (%)', ascending=False)
            
            # Bar chart
            risk_contrib_fig = go.Figure()
            
            risk_contrib_fig.add_trace(go.Bar(
                x=risk_df['Symbol'].head(10),
                y=risk_df['Risk Contrib (%)'].head(10),
                name='Risk Contribution',
                marker_color=COLORS['chart_purple'],
                text=risk_df['Risk Contrib (%)'].head(10).apply(lambda x: f'{x:.1f}%'),
                textposition='outside'
            ))
            
            risk_contrib_fig.add_trace(go.Scatter(
                x=risk_df['Symbol'].head(10),
                y=risk_df['Weight (%)'].head(10),
                name='Portfolio Weight',
                mode='lines+markers',
                line={'color': COLORS['warning'], 'width': 2, 'dash': 'dash'},
                yaxis='y2'
            ))
            
            risk_contrib_fig.update_layout(
                title="Top 10 Risk Contributors",
                xaxis_title="Symbol",
                yaxis_title="Risk Contribution (%)",
                yaxis2=dict(
                    title="Portfolio Weight (%)",
                    overlaying='y',
                    side='right'
                ),
                template=PLOTLY_TEMPLATE,
                height=450,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(risk_contrib_fig, use_container_width=True)
            
            # Risk concentration alert
            top_risk_contrib = risk_df['Risk Contrib (%)'].head(5).sum()
            if top_risk_contrib > 50:
                st.warning(f"⚠️ Top 5 positions contribute {top_risk_contrib:.1f}% of portfolio risk")
            else:
                st.success(f"✅ Risk well-distributed: Top 5 contribute {top_risk_contrib:.1f}% of risk")
        
        st.markdown("---")
        
        # === SECTION 5: ROLLING METRICS ===
        st.subheader("📈 Rolling Risk Metrics (90-Day Window)")
        
        # Calculate rolling metrics
        rolling_vol = port_ret.rolling(90).std() * np.sqrt(252) * 100
        rolling_sharpe = (port_ret.rolling(90).mean() * 252) / (port_ret.rolling(90).std() * np.sqrt(252))
        
        col_roll1, col_roll2 = st.columns(2)
        
        with col_roll1:
            st.markdown("**Rolling 90-Day Volatility**")
            vol_fig = go.Figure()
            vol_fig.add_trace(
                go.Scatter(x=rolling_vol.index, y=rolling_vol.values, name='Volatility',
                          fill='tozeroy', line={'color': COLORS['chart_purple']})
            )
            vol_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                template=PLOTLY_TEMPLATE,
                height=300,
                showlegend=False
            )
            st.plotly_chart(vol_fig, use_container_width=True)
        
        with col_roll2:
            st.markdown("**Rolling 90-Day Sharpe Ratio**")
            sharpe_fig = go.Figure()
            sharpe_fig.add_trace(
                go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name='Sharpe Ratio',
                          line={'color': COLORS['chart_blue'], 'width': 2})
            )
            sharpe_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Sharpe Ratio",
                template=PLOTLY_TEMPLATE,
                height=300,
                showlegend=False
            )
            st.plotly_chart(sharpe_fig, use_container_width=True)
    
    # TAB 6: WHAT-IF ANALYSIS
    with tab6:
        st.subheader("🔮 What-If Analysis & Portfolio Optimizer")
        
        st.markdown("""
        Simulate portfolio changes and see the impact on your returns, risk, and allocation.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("💰 Add Cash / New Position")
            
            cash_to_add = st.number_input("Cash to Invest ($)", min_value=0, value=10000, step=1000)
            
            new_symbol = st.text_input("Ticker Symbol (optional)", value="", placeholder="e.g., NVDA")
            
            if new_symbol:
                new_symbol = new_symbol.upper().strip()
                try:
                    quote = get_latest_quote(new_symbol)
                    if quote['price']:
                        shares_to_buy = int(cash_to_add / quote['price'])
                        st.info(f"💡 You can buy **{shares_to_buy} shares** of {new_symbol} at ${quote['price']:.2f}")
                        
                        if st.button("Simulate Purchase"):
                            # Create new portfolio with added position
                            new_row = pd.DataFrame([{
                                'Symbol': new_symbol,
                                'Qty': shares_to_buy,
                                'AvgCost': quote['price'],
                                'CostBasis': cash_to_add
                            }])
                            
                            simulated_portfolio = pd.concat([portfolio_df, new_row], ignore_index=True)
                            sim_engine = PortfolioEngine(simulated_portfolio)
                            
                            st.success(f"✅ Simulation Complete!")
                            
                            # Show comparison
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Current Value", f"${engine.total_market_value:,.0f}")
                                st.metric("Current Positions", len(engine.holdings))
                            with col_b:
                                st.metric("New Value", f"${sim_engine.total_market_value:,.0f}",
                                         f"+${sim_engine.total_market_value - engine.total_market_value:,.0f}")
                                st.metric("New Positions", len(sim_engine.holdings))
                    else:
                        st.error(f"Could not fetch price for {new_symbol}")
                except:
                    st.error(f"Invalid ticker symbol: {new_symbol}")
        
        with col2:
            st.subheader("📉 Sell Position")
            
            symbol_to_sell = st.selectbox("Select Position to Sell", engine.holdings['Symbol'].tolist())
            
            current_holding = engine.holdings[engine.holdings['Symbol'] == symbol_to_sell].iloc[0]
            
            st.write(f"**Current Holdings:** {current_holding['Qty']:.0f} shares")
            st.write(f"**Market Value:** ${current_holding['MarketValue']:,.2f}")
            st.write(f"**Unrealized P/L:** ${current_holding['TotalPL']:+,.2f} ({current_holding['TotalPLPct']:+.2f}%)")
            
            shares_to_sell = st.slider("Shares to Sell", 0, int(current_holding['Qty']), int(current_holding['Qty']))
            
            if shares_to_sell > 0:
                proceeds = shares_to_sell * current_holding['LastPrice']
                realized_pl = (current_holding['LastPrice'] - current_holding['AvgCost']) * shares_to_sell
                
                st.info(f"💵 **Proceeds:** ${proceeds:,.2f}")
                st.info(f"📊 **Realized P/L:** ${realized_pl:+,.2f}")
                
                if st.button("Simulate Sale"):
                    # Create new portfolio with reduced position
                    simulated_portfolio = portfolio_df.copy()
                    mask = simulated_portfolio['Symbol'] == symbol_to_sell
                    
                    new_qty = simulated_portfolio.loc[mask, 'Qty'].values[0] - shares_to_sell
                    
                    if new_qty > 0:
                        simulated_portfolio.loc[mask, 'Qty'] = new_qty
                        simulated_portfolio.loc[mask, 'CostBasis'] = new_qty * current_holding['AvgCost']
                    else:
                        simulated_portfolio = simulated_portfolio[~mask]
                    
                    sim_engine = PortfolioEngine(simulated_portfolio)
                    
                    st.success(f"✅ Simulation Complete!")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Current Weight", f"{current_holding['Weight']:.2f}%")
                    with col_b:
                        if symbol_to_sell in sim_engine.holdings['Symbol'].values:
                            new_weight = sim_engine.holdings[sim_engine.holdings['Symbol'] == symbol_to_sell]['Weight'].values[0]
                            st.metric("New Weight", f"{new_weight:.2f}%", f"{new_weight - current_holding['Weight']:.2f}%")
                        else:
                            st.metric("New Weight", "0%", "-100%")
        
        st.markdown("---")
        
        # Portfolio rebalancing suggestions
        st.subheader("⚖️ Rebalancing Suggestions")
        
        # Find overweight positions (>10% of portfolio)
        overweight = engine.holdings[engine.holdings['Weight'] > 10]
        
        if not overweight.empty:
            st.warning(f"⚠️ **{len(overweight)} positions** are overweight (>10% of portfolio)")
            
            for _, row in overweight.iterrows():
                excess_weight = row['Weight'] - 10
                excess_value = (excess_weight / 100) * engine.total_market_value
                
                st.write(f"**{row['Symbol']}:** {row['Weight']:.2f}% (Target: 10%)")
                st.write(f"   → Consider reducing by ${excess_value:,.0f} ({excess_weight:.2f}%)")
        else:
            st.success("✅ No positions are overweight. Portfolio is well-balanced!")
        
        # Sector concentration
        st.subheader("🏢 Sector Concentration Check")
        overweight_sectors = engine.sector_allocation[engine.sector_allocation > 25]
        
        if not overweight_sectors.empty:
            st.warning(f"⚠️ **{len(overweight_sectors)} sectors** are overweight (>25%)")
            for sector, weight in overweight_sectors.items():
                st.write(f"**{sector}:** {weight:.2f}% (Target: <25%)")
        else:
            st.success("✅ Sector allocation is well-diversified!")

if __name__ == "__main__":
    main()
