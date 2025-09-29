import pandas as pd
import numpy as np

class BacktestingEngine:
    """
    A simple event-driven backtesting engine.
    """
    def __init__(self, market_data):
        """
        Initializes the backtesting engine with market data.

        Args:
            market_data (pd.DataFrame): DataFrame containing market data,
                                        indexed by date/time.
        """
        self.market_data = market_data
        self.events = []
        self.portfolio = {} # Placeholder for portfolio state
        self.cash = 1_000_000 # Starting cash
        self.history = [] # To store portfolio value over time

    def _process_event(self, event):
        """
        Processes a single event (e.g., market data update, signal).
        """
        # This method will be extended by specific strategies
        pass

    def run(self, strategy):
        """
        Runs the backtest with a given strategy.

        Args:
            strategy (Strategy): An instance of a trading strategy.
        """
        print("Starting backtest...")
        for index, row in self.market_data.iterrows():
            current_date = index
            # Create a market event
            market_event = {'type': 'MARKET', 'date': current_date, 'data': row}
            self.events.append(market_event)

            # Strategy generates signals based on market data
            signals = strategy.generate_signals(market_event, self.portfolio)
            for signal in signals:
                self.events.append(signal)

            # Process all events for the current time step
            while self.events:
                event = self.events.pop(0)
                self._process_event(event) # Placeholder for event processing

            # Update portfolio history (simplified)
            self.history.append({'date': current_date, 'cash': self.cash, 'portfolio_value': self.cash}) # Simplified

        print("Backtest complete.")
        return pd.DataFrame(self.history).set_index('date')

class Strategy:
    """
    Base class for a trading strategy.
    """
    def __init__(self):
        pass

    def generate_signals(self, market_event, portfolio):
        """
        Generates trading signals based on market data.

        Args:
            market_event (dict): Current market data.
            portfolio (dict): Current portfolio state.

        Returns:
            list: A list of signal events.
        """
        return []

if __name__ == '__main__':
    # Example Usage (dummy market data)
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=10, freq='D'))
    dummy_market_data = pd.DataFrame(
        {'asset_price': np.random.rand(10) * 100 + 90},
        index=dates
    )

    class DummyStrategy(Strategy):
        def generate_signals(self, market_event, portfolio):
            # Example: Buy if price is below 95
            if market_event['data']['asset_price'] < 95:
                print(f"[{market_event['date']}] Signal: Buy asset at {market_event['data']['asset_price']:.2f}")
                return [{'type': 'BUY', 'date': market_event['date'], 'asset': 'asset', 'quantity': 10}]
            return []

    engine = BacktestingEngine(dummy_market_data)
    strategy = DummyStrategy()
    results = engine.run(strategy)
    print("\nBacktest Results (simplified):")
    print(results)
