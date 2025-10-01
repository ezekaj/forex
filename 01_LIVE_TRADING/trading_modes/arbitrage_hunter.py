"""
ARBITRAGE OPPORTUNITY HUNTER
Detects price differences between brokers and data sources
Can make risk-free profits from price discrepancies
"""

import os
import sys
import time
import logging
import requests
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArbitrageHunter:
    """
    Hunts for arbitrage opportunities across multiple sources
    Types: Triangular, Cross-broker, Latency arbitrage
    """
    
    def __init__(self):
        # Price sources (you'd add more brokers here)
        self.sources = {
            'alpha_vantage': self.get_alpha_vantage_price,
            'demo_broker1': self.get_demo_price_1,
            'demo_broker2': self.get_demo_price_2,
        }
        
        # Arbitrage thresholds
        self.min_profit_pips = 2  # Minimum 2 pips profit
        self.max_spread = 3  # Maximum acceptable spread
        self.execution_time = 0.5  # Estimated execution time in seconds
        
        # Tracking
        self.opportunities_found = 0
        self.total_potential_profit = 0
        
    def get_alpha_vantage_price(self, pair="EURUSD") -> Dict:
        """Get price from Alpha Vantage"""
        try:
            # This would use your real API key
            api_key = os.getenv("ALPHAVANTAGE_API_KEY", "demo")
            url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=EUR&to_currency=USD&apikey={api_key}"
            
            # For demo, return simulated price
            import random
            base = 1.0850
            return {
                'bid': base + random.uniform(-0.0005, 0.0005),
                'ask': base + random.uniform(0.0001, 0.0003),
                'timestamp': datetime.now()
            }
        except:
            return None
            
    def get_demo_price_1(self, pair="EURUSD") -> Dict:
        """Simulated broker 1 price (would be real broker API)"""
        import random
        base = 1.0850
        spread = 0.0002
        
        bid = base + random.uniform(-0.0008, 0.0008)
        return {
            'bid': bid,
            'ask': bid + spread,
            'timestamp': datetime.now()
        }
        
    def get_demo_price_2(self, pair="EURUSD") -> Dict:
        """Simulated broker 2 price (would be real broker API)"""
        import random
        base = 1.0850
        spread = 0.0003
        
        # Intentionally create occasional arbitrage opportunity
        if random.random() < 0.1:  # 10% chance
            base += random.choice([-0.0015, 0.0015])  # Price discrepancy
            
        bid = base + random.uniform(-0.0005, 0.0005)
        return {
            'bid': bid,
            'ask': bid + spread,
            'timestamp': datetime.now()
        }
        
    def detect_cross_broker_arbitrage(self, pair="EURUSD") -> List[Dict]:
        """
        Detect arbitrage between different brokers
        Buy low at one broker, sell high at another
        """
        
        opportunities = []
        prices = {}
        
        # Get prices from all sources
        for source_name, price_func in self.sources.items():
            price = price_func(pair)
            if price:
                prices[source_name] = price
                
        # Compare all pairs of sources
        source_names = list(prices.keys())
        for i in range(len(source_names)):
            for j in range(i + 1, len(source_names)):
                source1 = source_names[i]
                source2 = source_names[j]
                
                price1 = prices[source1]
                price2 = prices[source2]
                
                # Check if we can buy at source1 and sell at source2
                profit_pips = (price2['bid'] - price1['ask']) * 10000
                if profit_pips >= self.min_profit_pips:
                    opportunities.append({
                        'type': 'cross_broker',
                        'buy_source': source1,
                        'sell_source': source2,
                        'buy_price': price1['ask'],
                        'sell_price': price2['bid'],
                        'profit_pips': profit_pips,
                        'timestamp': datetime.now()
                    })
                    
                # Check reverse direction
                profit_pips = (price1['bid'] - price2['ask']) * 10000
                if profit_pips >= self.min_profit_pips:
                    opportunities.append({
                        'type': 'cross_broker',
                        'buy_source': source2,
                        'sell_source': source1,
                        'buy_price': price2['ask'],
                        'sell_price': price1['bid'],
                        'profit_pips': profit_pips,
                        'timestamp': datetime.now()
                    })
                    
        return opportunities
        
    def detect_triangular_arbitrage(self) -> List[Dict]:
        """
        Detect triangular arbitrage opportunities
        Example: EUR/USD -> USD/JPY -> EUR/JPY
        """
        
        opportunities = []
        
        # Get prices for triangle (would use real data)
        # For demo, simulate prices
        import random
        
        eurusd = 1.0850 + random.uniform(-0.001, 0.001)
        usdjpy = 110.50 + random.uniform(-0.1, 0.1)
        eurjpy = 119.90 + random.uniform(-0.1, 0.1)
        
        # Calculate implied EUR/JPY rate
        implied_eurjpy = eurusd * usdjpy
        
        # Check for arbitrage
        discrepancy = (eurjpy - implied_eurjpy) / eurjpy * 100
        
        if abs(discrepancy) > 0.05:  # 0.05% threshold
            opportunities.append({
                'type': 'triangular',
                'pairs': ['EUR/USD', 'USD/JPY', 'EUR/JPY'],
                'prices': {
                    'EURUSD': eurusd,
                    'USDJPY': usdjpy,
                    'EURJPY': eurjpy,
                    'implied_EURJPY': implied_eurjpy
                },
                'discrepancy_pct': discrepancy,
                'profit_potential': abs(discrepancy) * 1000,  # Per $1000 traded
                'timestamp': datetime.now()
            })
            
        return opportunities
        
    def detect_latency_arbitrage(self, pair="EURUSD") -> List[Dict]:
        """
        Detect latency arbitrage (one source is faster than another)
        """
        
        opportunities = []
        prices_history = []
        
        # Collect prices over time
        for _ in range(5):
            prices = {}
            for source_name, price_func in self.sources.items():
                price = price_func(pair)
                if price:
                    prices[source_name] = price
                    
            prices_history.append(prices)
            time.sleep(0.1)  # 100ms between checks
            
        # Analyze for latency patterns
        if len(prices_history) >= 2:
            for i in range(1, len(prices_history)):
                prev_prices = prices_history[i-1]
                curr_prices = prices_history[i]
                
                for source in curr_prices.keys():
                    if source in prev_prices:
                        # Check if one source is leading
                        price_change = curr_prices[source]['bid'] - prev_prices[source]['bid']
                        
                        # Check if other sources follow
                        for other_source in curr_prices.keys():
                            if other_source != source and other_source in prev_prices:
                                other_change = curr_prices[other_source]['bid'] - prev_prices[other_source]['bid']
                                
                                # If source moved but other didn't yet (latency)
                                if abs(price_change) > 0.0001 and abs(other_change) < 0.00005:
                                    opportunities.append({
                                        'type': 'latency',
                                        'fast_source': source,
                                        'slow_source': other_source,
                                        'price_move': price_change * 10000,  # in pips
                                        'latency_ms': 100,  # Estimated
                                        'timestamp': datetime.now()
                                    })
                                    
        return opportunities
        
    def execute_arbitrage(self, opportunity: Dict):
        """
        Execute arbitrage trade (would place real orders)
        """
        
        opp_type = opportunity['type']
        
        if opp_type == 'cross_broker':
            logger.info(f"🎯 ARBITRAGE EXECUTION:")
            logger.info(f"   Buy at {opportunity['buy_source']}: {opportunity['buy_price']:.5f}")
            logger.info(f"   Sell at {opportunity['sell_source']}: {opportunity['sell_price']:.5f}")
            logger.info(f"   Profit: {opportunity['profit_pips']:.1f} pips")
            
            # In real implementation:
            # 1. Place buy order at source 1
            # 2. Simultaneously place sell order at source 2
            # 3. Close both when filled
            
            self.total_potential_profit += opportunity['profit_pips']
            
        elif opp_type == 'triangular':
            logger.info(f"🔺 TRIANGULAR ARBITRAGE:")
            logger.info(f"   Pairs: {', '.join(opportunity['pairs'])}")
            logger.info(f"   Discrepancy: {opportunity['discrepancy_pct']:.3f}%")
            logger.info(f"   Profit potential: ${opportunity['profit_potential']:.2f} per $1000")
            
        elif opp_type == 'latency':
            logger.info(f"⚡ LATENCY ARBITRAGE:")
            logger.info(f"   Fast source: {opportunity['fast_source']}")
            logger.info(f"   Slow source: {opportunity['slow_source']}")
            logger.info(f"   Price lead: {opportunity['price_move']:.1f} pips")
            
    def run_hunter(self):
        """Main arbitrage hunting loop"""
        
        logger.info("="*60)
        logger.info("🎯 ARBITRAGE HUNTER ACTIVATED")
        logger.info("Searching for risk-free profit opportunities...")
        logger.info("="*60)
        
        check_count = 0
        
        try:
            while True:
                check_count += 1
                
                # Check for different types of arbitrage
                opportunities = []
                
                # 1. Cross-broker arbitrage
                cross_broker = self.detect_cross_broker_arbitrage()
                opportunities.extend(cross_broker)
                
                # 2. Triangular arbitrage
                triangular = self.detect_triangular_arbitrage()
                opportunities.extend(triangular)
                
                # 3. Latency arbitrage
                if check_count % 10 == 0:  # Check less frequently
                    latency = self.detect_latency_arbitrage()
                    opportunities.extend(latency)
                    
                # Process opportunities
                for opp in opportunities:
                    self.opportunities_found += 1
                    logger.info(f"\n💰 OPPORTUNITY #{self.opportunities_found} FOUND!")
                    self.execute_arbitrage(opp)
                    
                # Status update
                if check_count % 100 == 0:
                    logger.info(f"\n📊 Status: Checks: {check_count} | "
                              f"Opportunities: {self.opportunities_found} | "
                              f"Potential Profit: {self.total_potential_profit:.1f} pips")
                    
                # Rate limiting
                time.sleep(1)  # Check every second
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Shutting down arbitrage hunter...")
            
        finally:
            logger.info(f"\n📈 Final Report:")
            logger.info(f"Total opportunities found: {self.opportunities_found}")
            logger.info(f"Total potential profit: {self.total_potential_profit:.1f} pips")
            
            if self.opportunities_found > 0:
                logger.info(f"Average profit per opportunity: "
                          f"{self.total_potential_profit/self.opportunities_found:.1f} pips")

if __name__ == "__main__":
    hunter = ArbitrageHunter()
    hunter.run_hunter()