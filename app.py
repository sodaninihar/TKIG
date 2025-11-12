"""
TKIG Portfolio Dashboard - Enhanced Version with Transaction Management
=========================================================================
Professional portfolio management with buy/sell functionality and persistent storage.

Features:
- SQLite database for persistent storage
- Buy/Sell positions directly from dashboard
- Transaction history tracking
- Automatic portfolio updates
- Fixed VTI benchmark plotting
- Fixed excess return calculations

To run:
    streamlit run app_enhanced.py
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
import sqlite3
from pathlib import Path
import json

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="TKIG Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATABASE SETUP
# ============================================================================

DB_PATH = Path("portfolio.db")

def init_database():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Transactions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            transaction_type TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            transaction_date TEXT NOT NULL,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Portfolio settings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()

def load_initial_portfolio():
    """Load initial portfolio from uploaded Excel file into database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if already loaded
    cursor.execute("SELECT COUNT(*) FROM transactions")
    if cursor.fetchone()[0] > 0:
        conn.close()
        return
    
    # Load from Excel
    try:
        df = pd.read_excel('/mnt/user-data/uploads/Corporate-Positions-2025-11-11_xl.xlsx', skiprows=2)
        df = df[df['Symbol'].notna()].copy()
        df = df[df['Security Type'] != '--'].copy()
        
        # Clean data
        df['Symbol'] = df['Symbol'].str.strip().str.upper().str.replace('BRK/B', 'BRK-B')
        
        def clean_currency(val):
            if pd.isna(val) or val in ['--', '']:
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            return float(str(val).replace('$', '').replace(',', ''))
        
        df['Qty'] = pd.to_numeric(df['Qty (Quantity)'], errors='coerce').fillna(0)
        df['Cost_Basis'] = df['Cost Basis'].apply(clean_currency)
        df['AvgCost'] = df.apply(lambda row: row['Cost_Basis'] / row['Qty'] if row['Qty'] > 0 else 0, axis=1)
        
        # Insert as initial BUY transactions
        for _, row in df.iterrows():
            if row['Qty'] > 0:
                cursor.execute("""
                    INSERT INTO transactions (symbol, transaction_type, quantity, price, transaction_date, notes)
                    VALUES (?, 'BUY', ?, ?, ?, 'Initial portfolio load')
                """, (row['Symbol'], row['Qty'], row['AvgCost'], '2025-11-11'))
        
        conn.commit()
        st.success(f"✅ Loaded {len(df)} initial positions from Excel")
    except Exception as e:
        st.warning(f"No initial portfolio file found: {e}")
    
    conn.close()

def get_current_positions() -> pd.DataFrame:
    """Calculate current positions from transaction history."""
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT 
            symbol,
            SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE -quantity END) as qty,
            SUM(CASE WHEN transaction_type = 'BUY' THEN quantity * price ELSE -quantity * price END) as total_cost
        FROM transactions
        GROUP BY symbol
        HAVING qty > 0
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) == 0:
        return pd.DataFrame(columns=['Symbol', 'Qty', 'AvgCost', 'CostBasis'])
    
    df['AvgCost'] = df['total_cost'] / df['qty']
    df['CostBasis'] = df['total_cost']
    df = df.rename(columns={'symbol': 'Symbol', 'qty': 'Qty'})
    
    return df[['Symbol', 'Qty', 'AvgCost', 'CostBasis']].reset_index(drop=True)

def add_transaction(symbol: str, trans_type: str, quantity: float, price: float, date: str, notes: str = ""):
    """Add a new transaction to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO transactions (symbol, transaction_type, quantity, price, transaction_date, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (symbol.upper(), trans_type, quantity, price, date, notes))
    
    conn.commit()
    conn.close()

def get_transaction_history() -> pd.DataFrame:
    """Get all transactions."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM transactions ORDER BY transaction_date DESC, created_at DESC
    """, conn)
    conn.close()
    return df

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

# Custom CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
    .main {{ background-color: {COLORS['background']}; }}
    
    .stMetric {{
        background: linear-gradient(135deg, {COLORS['surface']} 0%, #F8FAFC 100%);
        padding: 24px 20px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid {COLORS['border']};
    }}
    
    .stMetric label {{
        color: {COLORS['text_muted']};
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }}
    
    h1, h2, h3 {{ color: {COLORS['text']}; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font={'family': 'Inter, sans-serif', 'size': 12, 'color': COLORS['text']},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        colorway=[COLORS['chart_blue'], COLORS['chart_green'], COLORS['chart_purple']],
        hovermode='x unified',
        xaxis={'gridcolor': COLORS['border'], 'showgrid': True},
        yaxis={'gridcolor': COLORS['border'], 'showgrid': True}
    )
)

# ============================================================================
# DATA FUNCTIONS
# ============================================================================

TICKER_MAP = {
    'BRKB': 'BRK-B', 'BRK.B': 'BRK-B', 'BRK/B': 'BRK-B',
    'BRKA': 'BRK-A', 'BRK.A': 'BRK-A', 'BRK/A': 'BRK-A'
}

def normalize_ticker(symbol: str) -> str:
    return TICKER_MAP.get(symbol.upper(), symbol.upper())

@st.cache_data(ttl=300)
def get_prices(symbols: List[str], period: str = '5y') -> pd.DataFrame:
    """Download historical prices."""
    if len(symbols) == 0:
        return pd.DataFrame()
    
    try:
        data = yf.download(symbols, period=period, progress=False, auto_adjust=True)
        
        if data.empty:
            return pd.DataFrame()
        
        # Extract Close prices
        if len(symbols) == 1:
            if 'Close' in data.columns:
                prices = data[['Close']].copy()
                prices.columns = symbols
            else:
                prices = data.to_frame()
                prices.columns = symbols
        else:
            prices = pd.DataFrame()
            for symbol in symbols:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if symbol in data.columns.get_level_values(0):
                            prices[symbol] = data[symbol]['Close']
                    else:
                        prices[symbol] = data['Close']
                except:
                    continue
        
        return prices.dropna(how='all')
    except Exception as e:
        st.error(f"Error downloading prices: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_latest_quote(symbol: str) -> Dict:
    """Get latest quote."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='5d')
        
        if len(hist) == 0:
            return {'price': None, 'change': None, 'change_pct': None}
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        
        return {
            'price': current_price,
            'change': current_price - prev_close,
            'change_pct': ((current_price - prev_close) / prev_close * 100) if prev_close else 0
        }
    except:
        return {'price': None, 'change': None, 'change_pct': None}

@st.cache_data(ttl=86400)
def get_basics(symbol: str) -> Dict:
    """Get basic stock info."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'name': info.get('longName', info.get('shortName', symbol)),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown')
        }
    except:
        return {'name': symbol, 'sector': 'Unknown', 'industry': 'Unknown'}

# ============================================================================
# PORTFOLIO ENGINE
# ============================================================================

class PortfolioEngine:
    """Enhanced portfolio engine."""
    
    def __init__(self, holdings: pd.DataFrame, benchmark: str = 'VTI'):
        self.holdings = holdings.copy()
        self.benchmark = benchmark
        self.symbols = holdings['Symbol'].tolist()
        self.all_symbols = self.symbols + [benchmark]
        
        # Load data
        self.basics = {symbol: get_basics(symbol) for symbol in self.symbols}
        self._refresh_quotes()
        
        # Load prices with better error handling
        st.info("📊 Loading market data...")
        self.prices = get_prices(self.all_symbols, period='5y')
        
        # Ensure benchmark loaded
        if self.benchmark not in self.prices.columns or self.prices[self.benchmark].isna().all():
            st.warning(f"Loading {self.benchmark} separately...")
            bench_prices = get_prices([self.benchmark], period='5y')
            if not bench_prices.empty:
                self.prices[self.benchmark] = bench_prices[self.benchmark]
        
        self._calculate_metrics()
        self.portfolio_beta = self._calculate_portfolio_beta()
    
    def _refresh_quotes(self):
        """Refresh quotes."""
        quotes = {symbol: get_latest_quote(symbol) for symbol in self.symbols}
        
        self.holdings['LastPrice'] = self.holdings['Symbol'].map(lambda s: quotes[s]['price'])
        self.holdings['DayChange'] = self.holdings['Symbol'].map(lambda s: quotes[s]['change'])
        self.holdings['DayChangePct'] = self.holdings['Symbol'].map(lambda s: quotes[s]['change_pct'])
        
        self.holdings['MarketValue'] = self.holdings['Qty'] * self.holdings['LastPrice']
        self.holdings['TotalPL'] = self.holdings['MarketValue'] - self.holdings['CostBasis']
        
        # Safe return calculation
        self.holdings['TotalPLPct'] = self.holdings.apply(
            lambda row: (row['TotalPL'] / row['CostBasis'] * 100) if row['CostBasis'] > 0 else 0,
            axis=1
        )
        self.holdings['TotalPLPct'].replace([np.inf, -np.inf], 0, inplace=True)
        self.holdings['TotalPLPct'].fillna(0, inplace=True)
        
        self.holdings['DayPL'] = self.holdings['Qty'] * self.holdings['DayChange']
        
        # Add basics
        self.holdings['Name'] = self.holdings['Symbol'].map(lambda s: self.basics.get(s, {}).get('name', s))
        self.holdings['Sector'] = self.holdings['Symbol'].map(lambda s: self.basics.get(s, {}).get('sector', 'Unknown'))
        self.holdings['Industry'] = self.holdings['Symbol'].map(lambda s: self.basics.get(s, {}).get('industry', 'Unknown'))
    
    def _calculate_metrics(self):
        """Calculate metrics."""
        self.total_market_value = self.holdings['MarketValue'].sum()
        self.total_cost_basis = self.holdings['CostBasis'].sum()
        self.total_pl = self.total_market_value - self.total_cost_basis
        self.total_pl_pct = (self.total_pl / self.total_cost_basis * 100) if self.total_cost_basis > 0 else 0
        self.total_day_pl = self.holdings['DayPL'].sum()
        self.total_day_pl_pct = (self.total_day_pl / (self.total_market_value - self.total_day_pl) * 100) if self.total_market_value > 0 else 0
        
        self.holdings['Weight'] = (self.holdings['MarketValue'] / self.total_market_value * 100)
        self.sector_allocation = self.holdings.groupby('Sector')['Weight'].sum().sort_values(ascending=False)
    
    def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta."""
        try:
            port_ret, bench_ret = self.compute_returns()
            
            if len(port_ret) > 10 and len(bench_ret) > 10:
                covariance = np.cov(port_ret, bench_ret)[0, 1]
                bench_variance = np.var(bench_ret)
                beta = covariance / bench_variance if bench_variance > 0 else 1.0
                return beta
            return 1.0
        except:
            return 1.0
    
    def compute_portfolio_timeseries(self) -> pd.Series:
        """Compute portfolio value over time."""
        weights = self.holdings.set_index('Symbol')['Qty'].to_dict()
        portfolio_value = pd.Series(0.0, index=self.prices.index)
        
        for symbol in self.symbols:
            if symbol in self.prices.columns and symbol in weights:
                portfolio_value += self.prices[symbol].fillna(0) * weights[symbol]
        
        return portfolio_value.replace(0, np.nan).dropna()
    
    def compute_returns(self) -> Tuple[pd.Series, pd.Series]:
        """Compute returns."""
        portfolio_ts = self.compute_portfolio_timeseries()
        portfolio_returns = portfolio_ts.pct_change().dropna()
        
        if self.benchmark in self.prices.columns:
            benchmark_returns = self.prices[self.benchmark].pct_change().dropna()
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            return portfolio_returns[common_dates], benchmark_returns[common_dates]
        
        return portfolio_returns, portfolio_returns
    
    def compute_performance_metrics(self) -> Dict:
        """Compute performance metrics."""
        port_ret, bench_ret = self.compute_returns()
        
        if len(port_ret) < 2:
            return self._default_metrics()
        
        periods_per_year = 252
        
        port_ann_ret = (1 + port_ret.mean()) ** periods_per_year - 1
        bench_ann_ret = (1 + bench_ret.mean()) ** periods_per_year - 1
        
        port_vol = port_ret.std() * np.sqrt(periods_per_year)
        bench_vol = bench_ret.std() * np.sqrt(periods_per_year)
        
        port_sharpe = port_ann_ret / port_vol if port_vol > 0 else 0
        bench_sharpe = bench_ann_ret / bench_vol if bench_vol > 0 else 0
        
        # Drawdowns
        port_ts = self.compute_portfolio_timeseries()
        port_cummax = port_ts.cummax()
        port_drawdown = (port_ts - port_cummax) / port_cummax
        port_max_dd = port_drawdown.min()
        
        bench_ts = self.prices[self.benchmark]
        bench_cummax = bench_ts.cummax()
        bench_drawdown = (bench_ts - bench_cummax) / bench_cummax
        bench_max_dd = bench_drawdown.min()
        
        # Tracking error
        excess_returns = port_ret - bench_ret
        tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
        information_ratio = (excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)) if excess_returns.std() > 0 else 0
        
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
            'beta': self.portfolio_beta
        }
    
    def _default_metrics(self):
        """Default metrics when calculation fails."""
        return {
            'portfolio_return': 0, 'benchmark_return': 0,
            'portfolio_volatility': 0, 'benchmark_volatility': 0,
            'portfolio_sharpe': 0, 'benchmark_sharpe': 0,
            'portfolio_max_drawdown': 0, 'benchmark_max_drawdown': 0,
            'tracking_error': 0, 'information_ratio': 0,
            'beta': 1.0
        }

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main app."""
    
    # Initialize database
    init_database()
    load_initial_portfolio()
    
    # Header
    st.title("📊 TKIG Portfolio Dashboard - Enhanced")
    st.markdown(f"**Live Portfolio Management** • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load portfolio
    portfolio_df = get_current_positions()
    
    if len(portfolio_df) == 0:
        st.warning("No positions found. Add your first position below!")
        portfolio_df = pd.DataFrame(columns=['Symbol', 'Qty', 'AvgCost', 'CostBasis'])
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Portfolio Management")
        
        # Transaction form
        with st.expander("➕ Add Transaction", expanded=False):
            with st.form("add_transaction"):
                trans_type = st.selectbox("Type", ["BUY", "SELL"])
                symbol = st.text_input("Symbol", placeholder="AAPL").upper()
                quantity = st.number_input("Quantity", min_value=0.01, value=1.0, step=0.01)
                price = st.number_input("Price per Share", min_value=0.01, value=100.0, step=0.01)
                trans_date = st.date_input("Date", value=datetime.now())
                notes = st.text_area("Notes (optional)")
                
                submitted = st.form_submit_button("Add Transaction")
                
                if submitted and symbol:
                    add_transaction(symbol, trans_type, quantity, price, str(trans_date), notes)
                    st.success(f"✅ Added {trans_type} {quantity} {symbol} @ ${price}")
                    st.rerun()
        
        st.markdown("---")
        
        # Quick stats (if portfolio exists)
        if len(portfolio_df) > 0:
            with st.spinner("Loading portfolio..."):
                engine = PortfolioEngine(portfolio_df)
            
            st.markdown("### 📈 Quick Stats")
            
            # Beta card
            if engine.portfolio_beta > 1.2:
                beta_color = COLORS['danger']
                beta_label = 'High Vol'
            elif engine.portfolio_beta < 0.8:
                beta_color = COLORS['success']
                beta_label = 'Defensive'
            else:
                beta_color = COLORS['primary']
                beta_label = 'Market-Like'
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%); padding: 16px; border-radius: 8px; margin: 8px 0;">
                <div style="margin-bottom: 12px;">
                    <span style="color: #64748B; font-size: 11px;">PORTFOLIO VALUE</span><br>
                    <span style="color: #0F172A; font-size: 24px; font-weight: 700;">${engine.total_market_value:,.0f}</span>
                </div>
                <div style="margin-bottom: 12px;">
                    <span style="color: #64748B; font-size: 11px;">TOTAL RETURN</span><br>
                    <span style="color: {'#10B981' if engine.total_pl >= 0 else '#EF4444'}; font-size: 20px; font-weight: 600;">
                        {engine.total_pl_pct:+.2f}%
                    </span>
                </div>
                <div style="background: rgba(11, 95, 255, 0.1); padding: 12px; border-radius: 6px; border-left: 4px solid {beta_color};">
                    <span style="color: #64748B; font-size: 11px;">⚡ PORTFOLIO BETA</span><br>
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
        
        # View transactions
        if st.button("📋 View Transaction History"):
            st.session_state['show_transactions'] = True
    
    # Main content
    if len(portfolio_df) == 0:
        st.info("👆 Add your first position using the form in the sidebar")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "📋 Positions", "⚖️ Performance vs VTI"])
    
    # TAB 1: OVERVIEW
    with tab1:
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Account Value", f"${engine.total_market_value:,.0f}")
        
        with col2:
            st.metric("Cost Basis", f"${engine.total_cost_basis:,.0f}")
        
        with col3:
            st.metric("Total Return", f"{engine.total_pl_pct:+.2f}%", f"${engine.total_pl:+,.0f}")
        
        with col4:
            st.metric("Day Change", f"${engine.total_day_pl:+,.0f}", f"{engine.total_day_pl_pct:+.2f}%")
        
        with col5:
            st.metric("Positions", len(engine.holdings))
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Holdings")
            top_holdings = engine.holdings.nlargest(10, 'MarketValue')[['Symbol', 'MarketValue', 'Weight']]
            
            fig = go.Figure(data=[go.Bar(
                x=top_holdings['Symbol'],
                y=top_holdings['Weight'],
                text=top_holdings['Weight'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto',
                marker_color=COLORS['primary']
            )])
            fig.update_layout(
                title="Top 10 Holdings by Weight",
                xaxis_title="",
                yaxis_title="Weight (%)",
                template=PLOTLY_TEMPLATE,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sector Allocation")
            
            fig = go.Figure(data=[go.Pie(
                labels=engine.sector_allocation.index,
                values=engine.sector_allocation.values,
                hole=0.4
            )])
            fig.update_layout(template=PLOTLY_TEMPLATE, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: POSITIONS
    with tab2:
        st.subheader("Current Positions")
        
        display_df = engine.holdings[[
            'Symbol', 'Name', 'Qty', 'AvgCost', 'LastPrice',
            'MarketValue', 'TotalPL', 'TotalPLPct', 'Weight'
        ]].copy()
        
        # Format
        display_df['Qty'] = display_df['Qty'].apply(lambda x: f"{x:,.2f}")
        display_df['AvgCost'] = display_df['AvgCost'].apply(lambda x: f"${x:,.2f}")
        display_df['LastPrice'] = display_df['LastPrice'].apply(lambda x: f"${x:,.2f}")
        display_df['MarketValue'] = display_df['MarketValue'].apply(lambda x: f"${x:,.0f}")
        display_df['TotalPL'] = display_df['TotalPL'].apply(lambda x: f"${x:+,.0f}")
        display_df['TotalPLPct'] = display_df['TotalPLPct'].apply(lambda x: f"{x:+.2f}%")
        display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(display_df, hide_index=True, use_container_width=True, height=600)
    
    # TAB 3: PERFORMANCE
    with tab3:
        st.subheader("Performance vs VTI")
        
        # Check benchmark data
        if engine.benchmark not in engine.prices.columns or len(engine.prices[engine.benchmark].dropna()) < 10:
            st.error(f"❌ Insufficient {engine.benchmark} data. Cannot display performance comparison.")
            return
        
        # Cumulative return chart - FIXED
        port_ts = engine.compute_portfolio_timeseries()
        bench_ts = engine.prices[engine.benchmark].copy()
        
        # Align dates
        common_dates = port_ts.index.intersection(bench_ts.index)
        port_ts = port_ts[common_dates]
        bench_ts = bench_ts[common_dates]
        
        if len(port_ts) < 2:
            st.warning("Insufficient data for comparison")
            return
        
        # Normalize to 100
        port_norm = port_ts / port_ts.iloc[0] * 100
        bench_norm = bench_ts / bench_ts.iloc[0] * 100
        
        fig = go.Figure()
        
        # Portfolio line
        fig.add_trace(go.Scatter(
            x=port_norm.index,
            y=port_norm.values,
            name='Portfolio',
            line={'color': COLORS['primary'], 'width': 2},
            mode='lines'
        ))
        
        # VTI line - FIXED
        fig.add_trace(go.Scatter(
            x=bench_norm.index,
            y=bench_norm.values,
            name='VTI',
            line={'color': COLORS['text_muted'], 'width': 2, 'dash': 'dash'},
            mode='lines'
        ))
        
        fig.update_layout(
            title="Cumulative Return (Normalized to 100)",
            xaxis_title="Date",
            yaxis_title="Value",
            template=PLOTLY_TEMPLATE,
            height=400,
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
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
                {'Metric': 'Beta vs VTI', 'Portfolio': f"{metrics['beta']:.2f}", 'VTI': '1.00'}
            ])
            
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("Rolling 60-Day Excess Return")
            
            # FIXED excess return calculation
            port_ret, bench_ret = engine.compute_returns()
            
            if len(port_ret) > 60:
                excess_ret = (port_ret - bench_ret).rolling(60).mean() * 252 * 100  # Annualized
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=excess_ret.index,
                    y=excess_ret.values,
                    name='60-Day Excess Return',
                    fill='tozeroy',
                    line={'color': COLORS['primary']},
                    mode='lines'
                ))
                fig2.add_hline(y=0, line_dash="dash", line_color=COLORS['text_muted'])
                fig2.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Excess Return (%)",
                    template=PLOTLY_TEMPLATE,
                    height=350,
                    showlegend=False
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Need at least 60 days of data for rolling calculations")
    
    # Transaction history modal
    if st.session_state.get('show_transactions', False):
        with st.expander("📋 Transaction History", expanded=True):
            trans_df = get_transaction_history()
            if len(trans_df) > 0:
                trans_df['amount'] = trans_df['quantity'] * trans_df['price']
                display_trans = trans_df[['transaction_date', 'symbol', 'transaction_type', 'quantity', 'price', 'amount', 'notes']]
                st.dataframe(display_trans, use_container_width=True, height=400)
            else:
                st.info("No transactions yet")
            
            if st.button("Close"):
                st.session_state['show_transactions'] = False
                st.rerun()

if __name__ == "__main__":
    main()
