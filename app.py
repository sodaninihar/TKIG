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
    'background': '#FFFFFF',
    'surface': '#F8FAFC',
    'text': '#1E293B',
    'text_muted': '#334155',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'border': '#E2E8F0'
}

# Custom CSS
st.markdown(f"""
<style>
    .main {{
        background-color: {COLORS['background']};
    }}
    .stMetric {{
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .stMetric label {{
        color: {COLORS['text_muted']};
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .stMetric .metric-value {{
        color: {COLORS['primary']};
        font-size: 28px;
        font-weight: 700;
    }}
    h1 {{
        color: {COLORS['primary']};
    }}
    .positive {{
        color: {COLORS['success']};
    }}
    .negative {{
        color: {COLORS['danger']};
    }}
</style>
""", unsafe_allow_html=True)

# Plotly template
PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font={'family': 'Inter, sans-serif', 'size': 12, 'color': COLORS['text']},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        colorway=[COLORS['primary'], COLORS['accent1'], COLORS['accent2'], '#8B5CF6', '#EC4899', '#F59E0B'],
        hovermode='closest',
        xaxis={'gridcolor': COLORS['border'], 'showgrid': True},
        yaxis={'gridcolor': COLORS['border'], 'showgrid': True}
    )
)

# ============================================================================
# TICKER SYMBOL MAPPING
# ============================================================================

TICKER_MAP = {
    'BRKB': 'BRK-B',
    'BRK.B': 'BRK-B',
    'BRKA': 'BRK-A',
    'BRK.A': 'BRK-A'
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

@st.cache_data(ttl=300)  # 5 minute cache
def get_news(symbols: List[str], limit: int = 50) -> pd.DataFrame:
    """Get news for portfolio symbols."""
    news_items = []
    
    for symbol in symbols[:10]:  # Limit to first 10 symbols to avoid rate limits
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            for item in news[:limit]:
                news_items.append({
                    'symbol': symbol,
                    'title': item.get('title', ''),
                    'source': item.get('publisher', 'Unknown'),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    'link': item.get('link', ''),
                    'sentiment': _simple_sentiment(item.get('title', ''))
                })
        except:
            continue
    
    return pd.DataFrame(news_items).sort_values('published', ascending=False).reset_index(drop=True) if news_items else pd.DataFrame()

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
        if len(port_ret) > 1 and len(bench_ret) > 1:
            covariance = np.cov(port_ret, bench_ret)[0, 1]
            bench_variance = np.var(bench_ret)
            beta = covariance / bench_variance if bench_variance > 0 else 1.0
        else:
            beta = 1.0
        
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

# Default portfolio data
DEFAULT_PORTFOLIO = """Symbol,Qty,AvgCost
AAPL,128,235.90
ADBE,50,346.98
ADI,100,244.68
AMGN,40,273.96
BLK,15,1112.63
BRK-B,130,491.12
CARR,300,61.01
EL,45,87.00
EPD,600,31.67
EVRG,220,72.31
GOOGL,200,249.98
HON,96,211.67
KO,109,66.22
LMT,66,473.18
MA,86,582.49
META,112,764.23
MLM,65,611.92
MSFT,164,513.57
MTB,65,196.90
NKE,63,73.12
ORLY,480,106.19
PEP,45,141.32
PG,82,156.77
PLD,100,114.34
PSX,75,131.10
UNH,48,349.16
SPY,203,660.02
XLV,190,132.07
"""

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    """Main Streamlit app."""
    
    # Header
    st.title("📊 TKIG Portfolio Dashboard")
    st.markdown(f"**Benchmarked to VTI** • Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Portfolio CSV", type=['csv'])
        
        if uploaded_file is not None:
            portfolio_df = load_portfolio_from_csv(uploaded_file)
        else:
            # Use default data
            import io
            portfolio_df = load_portfolio_from_csv(io.StringIO(DEFAULT_PORTFOLIO))
        
        st.success(f"✓ Loaded {len(portfolio_df)} positions")
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("Professional portfolio analytics with real-time data from Yahoo Finance.")
    
    # Initialize portfolio engine
    with st.spinner("Loading portfolio data..."):
        engine = PortfolioEngine(portfolio_df)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview",
        "📋 Positions", 
        "⚖️ Performance vs VTI",
        "🎯 Risk & Analytics",
        "📰 News & Events"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        # KPI Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Account Value",
                f"${engine.total_market_value:,.0f}",
                f"{engine.total_pl_pct:+.2f}%"
            )
        
        with col2:
            st.metric(
                "Cost Basis",
                f"${engine.total_cost_basis:,.0f}"
            )
        
        with col3:
            st.metric(
                "Unrealized P/L",
                f"${engine.total_pl:+,.0f}",
                f"{engine.total_pl_pct:+.2f}%"
            )
        
        with col4:
            st.metric(
                "Day P/L",
                f"${engine.total_day_pl:+,.0f}",
                f"{engine.total_day_pl_pct:+.2f}%"
            )
        
        with col5:
            st.metric(
                "# Positions",
                len(engine.holdings)
            )
        
        st.markdown("---")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Day Movers")
            top_movers = engine.holdings.nlargest(5, 'DayChangePct')[['Symbol', 'Name', 'DayChangePct', 'DayPL']]
            
            # Color code the dataframe
            def color_negative_red(val):
                color = COLORS['success'] if val > 0 else COLORS['danger']
                return f'color: {color}'
            
            st.dataframe(
                top_movers.style.applymap(color_negative_red, subset=['DayChangePct', 'DayPL']),
                hide_index=True,
                use_container_width=True
            )
        
        with col2:
            st.subheader("Sector Allocation")
            sector_fig = go.Figure(data=[go.Pie(
                labels=engine.sector_allocation.index,
                values=engine.sector_allocation.values,
                hole=0.4,
                marker={'colors': [COLORS['primary'], COLORS['accent1'], COLORS['accent2'], '#8B5CF6', '#EC4899', '#F59E0B']}
            )])
            sector_fig.update_layout(template=PLOTLY_TEMPLATE, height=300, showlegend=True)
            st.plotly_chart(sector_fig, use_container_width=True)
        
        # Top 10 concentration
        st.subheader("Top 10 Concentration")
        top10 = engine.holdings.nlargest(10, 'Weight')[['Symbol', 'Weight']]
        conc_fig = go.Figure(data=[go.Bar(x=top10['Symbol'], y=top10['Weight'], marker_color=COLORS['primary'])])
        conc_fig.update_layout(
            xaxis_title="Symbol",
            yaxis_title="Weight (%)",
            template=PLOTLY_TEMPLATE,
            height=350
        )
        st.plotly_chart(conc_fig, use_container_width=True)
    
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
    
    # TAB 3: PERFORMANCE
    with tab3:
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
    
    # TAB 4: RISK & ANALYTICS
    with tab4:
        st.subheader("Risk & Analytics")
        
        # Anomaly detection
        st.subheader("🚨 Anomaly Radar")
        anomalies = engine.detect_anomalies()
        
        if anomalies:
            anomaly_df = pd.DataFrame(anomalies)
            st.warning(f"Found {len(anomalies)} anomalies")
            st.dataframe(anomaly_df, hide_index=True, use_container_width=True)
        else:
            st.success("No anomalies detected")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Drawdown")
            port_ts = engine.compute_portfolio_timeseries()
            cummax = port_ts.cummax()
            drawdown = (port_ts - cummax) / cummax * 100
            
            dd_fig = go.Figure()
            dd_fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill='tozeroy',
                name='Drawdown',
                line={'color': COLORS['danger']}
            ))
            dd_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                template=PLOTLY_TEMPLATE,
                height=400
            )
            st.plotly_chart(dd_fig, use_container_width=True)
        
        with col2:
            st.subheader("Correlation Matrix")
            returns = engine.prices[engine.symbols].pct_change().dropna()
            corr_matrix = returns.corr()
            
            heatmap_fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 8}
            ))
            heatmap_fig.update_layout(
                template=PLOTLY_TEMPLATE,
                height=400
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # TAB 5: NEWS & EVENTS
    with tab5:
        st.subheader("News & Events")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📰 News Feed")
            news_df = get_news(engine.symbols, limit=10)
            
            if not news_df.empty:
                for _, row in news_df.head(10).iterrows():
                    sentiment_color = {
                        'Positive': COLORS['success'],
                        'Negative': COLORS['danger'],
                        'Neutral': COLORS['text_muted']
                    }.get(row['sentiment'], COLORS['text_muted'])
                    
                    with st.container():
                        st.markdown(f"""
                        <div style="padding: 15px; background: white; border-radius: 8px; margin-bottom: 10px; border: 1px solid {COLORS['border']};">
                            <span style="background: {COLORS['primary']}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-right: 8px;">{row['symbol']}</span>
                            <span style="color: {sentiment_color}; font-weight: bold;">{row['sentiment']}</span>
                            <h4 style="margin-top: 10px; margin-bottom: 5px;">{row['title']}</h4>
                            <p style="color: {COLORS['text_muted']}; font-size: 12px;">{row['source']} • {row['published'].strftime('%Y-%m-%d %H:%M')} • <a href="{row['link']}" target="_blank">Read more →</a></p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No news available")
        
        with col2:
            st.subheader("📅 Upcoming Earnings")
            # Note: earnings calendar requires more API calls, simplified for now
            st.info("Earnings calendar feature coming soon")

if __name__ == "__main__":
    main()
