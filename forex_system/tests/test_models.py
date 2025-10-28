"""
Test database models (Trade, Position, Opportunity, Account, Configuration, LogEvent).

Covers:
- CRUD operations
- Model relationships
- Validation and constraints
- Property methods
- Serialization (to_dict)
"""
import pytest
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError

from forex_system.models import Trade, Position, Opportunity


class TestTradeModel:
    """Test Trade model CRUD and business logic."""

    def test_create_trade(self, db_session):
        """Test creating a new trade."""
        trade = Trade(
            pair="EURUSD",
            direction="BUY",
            strategy="scalping",
            size=1.0,
            leverage=50,
            entry_price=1.0850,
            entry_time=datetime.utcnow(),
            stop_loss=1.0800,
            take_profit=1.0950,
            confidence_score=0.85
        )
        db_session.add(trade)
        db_session.commit()

        assert trade.id is not None
        assert trade.pair == "EURUSD"
        assert trade.direction == "BUY"
        assert trade.size == 1.0

    def test_trade_auto_timestamps(self, db_session):
        """Test that created_at and updated_at are set automatically."""
        trade = Trade(
            pair="GBPUSD",
            direction="SELL",
            strategy="day_trading",
            size=0.5,
            leverage=30,
            entry_price=1.2650,
            entry_time=datetime.utcnow(),
            stop_loss=1.2700
        )
        db_session.add(trade)
        db_session.commit()

        assert trade.created_at is not None
        assert trade.updated_at is not None
        assert trade.created_at == trade.updated_at

    def test_trade_update_timestamp(self, db_session):
        """Test that updated_at changes on modification."""
        trade = Trade(
            pair="USDJPY",
            direction="BUY",
            strategy="swing_trading",
            size=1.0,
            leverage=20,
            entry_price=149.50,
            entry_time=datetime.utcnow(),
            stop_loss=149.00
        )
        db_session.add(trade)
        db_session.commit()
        original_updated = trade.updated_at

        # Update trade
        trade.exit_price = 150.50
        trade.exit_time = datetime.utcnow()
        db_session.commit()

        assert trade.updated_at > original_updated

    def test_trade_with_all_fields(self, db_session):
        """Test trade with all optional fields populated."""
        trade = Trade(
            pair="AUDUSD",
            direction="SELL",
            strategy="scalping",
            size=2.0,
            leverage=50,
            entry_price=0.6550,
            entry_time=datetime.utcnow() - timedelta(hours=1),
            exit_price=0.6520,
            exit_time=datetime.utcnow(),
            exit_reason="TP",
            stop_loss=0.6580,
            take_profit=0.6520,
            risk_reward_ratio=3.0,
            pips=30.0,
            profit_loss=600.0,
            profit_loss_percentage=6.0,
            confidence_score=0.92,
            indicators_aligned=5,
            indicator_details={"RSI": 70, "MACD": "bearish", "BB": "upper"},
            was_auto_executed=True,
            user_approved=False,
            execution_delay_seconds=0.5,
            spread_at_entry=1.5,
            volatility_at_entry=0.002,
            trend_at_entry="bearish",
            notes="High confidence scalp"
        )
        db_session.add(trade)
        db_session.commit()

        assert trade.exit_reason == "TP"
        assert trade.pips == 30.0
        assert trade.profit_loss == 600.0
        assert trade.indicator_details["RSI"] == 70

    def test_trade_to_dict(self, sample_trade):
        """Test trade serialization to dictionary."""
        trade_dict = sample_trade.to_dict()

        assert trade_dict["pair"] == "EURUSD"
        assert trade_dict["direction"] == "BUY"
        assert trade_dict["strategy"] == "scalping"
        assert trade_dict["entry_price"] == 1.0850
        assert "entry_time" in trade_dict

    def test_trade_duration_minutes(self, db_session):
        """Test duration calculation."""
        entry = datetime.utcnow()
        exit_time = entry + timedelta(minutes=30)

        trade = Trade(
            pair="EURUSD",
            direction="BUY",
            strategy="scalping",
            size=1.0,
            leverage=50,
            entry_price=1.0850,
            entry_time=entry,
            exit_time=exit_time,
            stop_loss=1.0800
        )
        db_session.add(trade)
        db_session.commit()

        assert trade.duration_minutes == 30.0

    def test_trade_is_winner(self, db_session):
        """Test is_winner property."""
        trade = Trade(
            pair="GBPUSD",
            direction="BUY",
            strategy="day_trading",
            size=1.0,
            leverage=30,
            entry_price=1.2650,
            entry_time=datetime.utcnow(),
            stop_loss=1.2600,
            profit_loss=150.0
        )
        db_session.add(trade)
        db_session.commit()

        assert trade.is_winner is True

    def test_trade_is_loser(self, db_session):
        """Test is_loser property."""
        trade = Trade(
            pair="USDJPY",
            direction="SELL",
            strategy="scalping",
            size=1.0,
            leverage=50,
            entry_price=149.50,
            entry_time=datetime.utcnow(),
            stop_loss=149.80,
            profit_loss=-80.0
        )
        db_session.add(trade)
        db_session.commit()

        assert trade.is_loser is True

    def test_trade_repr(self, sample_trade):
        """Test string representation."""
        repr_str = repr(sample_trade)
        assert "Trade" in repr_str
        assert sample_trade.pair in repr_str


class TestPositionModel:
    """Test Position model CRUD and business logic."""

    def test_create_position(self, db_session):
        """Test creating a new position."""
        position = Position(
            pair="EURUSD",
            direction="BUY",
            size=1.0,
            leverage=50,
            entry_price=1.0850,
            stop_loss=1.0800,
            take_profit=1.0950
        )
        db_session.add(position)
        db_session.commit()

        assert position.id is not None
        assert position.pair == "EURUSD"
        assert position.unrealized_pnl == 0.0

    def test_position_with_current_price(self, db_session):
        """Test position with current price tracking."""
        position = Position(
            pair="GBPUSD",
            direction="SELL",
            size=0.5,
            leverage=30,
            entry_price=1.2650,
            current_price=1.2640,
            stop_loss=1.2700,
            take_profit=1.2550,
            unrealized_pnl=50.0,
            unrealized_pips=10.0
        )
        db_session.add(position)
        db_session.commit()

        assert position.current_price == 1.2640
        assert position.unrealized_pnl == 50.0
        assert position.unrealized_pips == 10.0

    def test_position_trailing_stop(self, db_session):
        """Test position with trailing stop enabled."""
        position = Position(
            pair="USDJPY",
            direction="BUY",
            size=1.0,
            leverage=30,
            entry_price=149.50,
            stop_loss=149.00,
            trailing_stop_enabled=True,
            trailing_stop_distance_pips=20.0
        )
        db_session.add(position)
        db_session.commit()

        assert position.trailing_stop_enabled is True
        assert position.trailing_stop_distance_pips == 20.0

    def test_position_auto_timestamps(self, db_session):
        """Test automatic timestamp fields."""
        position = Position(
            pair="AUDUSD",
            direction="BUY",
            size=1.0,
            leverage=30,
            entry_price=0.6550,
            stop_loss=0.6500
        )
        db_session.add(position)
        db_session.commit()

        assert position.opened_at is not None
        assert position.last_updated_at is not None

    def test_position_linked_to_trade(self, db_session):
        """Test position can link to a trade."""
        trade = Trade(
            pair="EURUSD",
            direction="BUY",
            strategy="scalping",
            size=1.0,
            leverage=50,
            entry_price=1.0850,
            entry_time=datetime.utcnow(),
            stop_loss=1.0800
        )
        db_session.add(trade)
        db_session.commit()

        position = Position(
            pair="EURUSD",
            direction="BUY",
            size=1.0,
            leverage=50,
            entry_price=1.0850,
            stop_loss=1.0800,
            trade_id=trade.id
        )
        db_session.add(position)
        db_session.commit()

        assert position.trade_id == trade.id

    def test_position_repr(self, sample_position):
        """Test string representation."""
        repr_str = repr(sample_position)
        assert "Position" in repr_str
        assert sample_position.pair in repr_str

    def test_multiple_positions_same_pair(self, db_session):
        """Test multiple positions can exist for same pair."""
        position1 = Position(
            pair="EURUSD",
            direction="BUY",
            size=1.0,
            leverage=50,
            entry_price=1.0850,
            stop_loss=1.0800
        )
        position2 = Position(
            pair="EURUSD",
            direction="SELL",
            size=0.5,
            leverage=50,
            entry_price=1.0860,
            stop_loss=1.0910
        )
        db_session.add_all([position1, position2])
        db_session.commit()

        positions = db_session.query(Position).filter_by(pair="EURUSD").all()
        assert len(positions) == 2


class TestOpportunityModel:
    """Test Opportunity model CRUD and business logic."""

    def test_create_opportunity(self, db_session):
        """Test creating a trading opportunity."""
        opportunity = Opportunity(
            pair="EURUSD",
            direction="BUY",
            strategy="scalping",
            timeframe="5m",
            proposed_entry_price=1.0850,
            proposed_size=1.0,
            proposed_stop_loss=1.0800,
            proposed_take_profit=1.0950,
            leverage=50,
            confidence_score=0.85,
            risk_reward_ratio=2.0,
            indicators_aligned=4,
            current_price=1.0850,
            expires_at=datetime.utcnow() + timedelta(minutes=5)
        )
        db_session.add(opportunity)
        db_session.commit()

        assert opportunity.id is not None
        assert opportunity.pair == "EURUSD"
        assert opportunity.confidence_score == 0.85

    def test_opportunity_qualifies_for_auto_execute(self, db_session):
        """Test high-confidence opportunity qualifies for auto-execute."""
        opportunity = Opportunity(
            pair="GBPUSD",
            direction="SELL",
            strategy="day_trading",
            timeframe="1h",
            proposed_entry_price=1.2650,
            proposed_size=1.0,
            proposed_stop_loss=1.2700,
            proposed_take_profit=1.2550,
            leverage=30,
            confidence_score=0.88,
            risk_reward_ratio=3.0,
            indicators_aligned=5,
            current_price=1.2650,
            qualifies_for_auto_execute=True
        )
        db_session.add(opportunity)
        db_session.commit()

        assert opportunity.qualifies_for_auto_execute is True

    def test_opportunity_with_market_context(self, db_session):
        """Test opportunity with full market context."""
        opportunity = Opportunity(
            pair="USDJPY",
            direction="BUY",
            strategy="swing_trading",
            timeframe="4h",
            proposed_entry_price=149.50,
            proposed_size=1.0,
            proposed_stop_loss=149.00,
            proposed_take_profit=151.00,
            leverage=20,
            confidence_score=0.82,
            risk_reward_ratio=3.0,
            indicators_aligned=6,
            current_price=149.50,
            spread_pips=2.0,
            volatility=0.003,
            trend="bullish",
            indicator_details={
                "RSI": 45,
                "MACD": "bullish_crossover",
                "BB": "lower_band_bounce",
                "EMA20": "above",
                "EMA50": "above",
                "Volume": "increasing"
            }
        )
        db_session.add(opportunity)
        db_session.commit()

        assert opportunity.spread_pips == 2.0
        assert opportunity.volatility == 0.003
        assert opportunity.trend == "bullish"
        assert opportunity.indicator_details["MACD"] == "bullish_crossover"

    def test_opportunity_user_decision_tracking(self, db_session):
        """Test user decision tracking."""
        opportunity = Opportunity(
            pair="AUDUSD",
            direction="BUY",
            strategy="scalping",
            timeframe="1m",
            proposed_entry_price=0.6550,
            proposed_size=1.0,
            proposed_stop_loss=0.6530,
            proposed_take_profit=0.6580,
            leverage=50,
            confidence_score=0.75,
            risk_reward_ratio=1.5,
            indicators_aligned=3,
            current_price=0.6550,
            was_sent_to_user=True,
            user_decision="approved",
            user_decision_time=datetime.utcnow()
        )
        db_session.add(opportunity)
        db_session.commit()

        assert opportunity.was_sent_to_user is True
        assert opportunity.user_decision == "approved"
        assert opportunity.user_decision_time is not None

    def test_opportunity_execution_tracking(self, db_session):
        """Test execution tracking."""
        opportunity = Opportunity(
            pair="USDCAD",
            direction="SELL",
            strategy="day_trading",
            timeframe="15m",
            proposed_entry_price=1.3550,
            proposed_size=1.0,
            proposed_stop_loss=1.3600,
            proposed_take_profit=1.3450,
            leverage=30,
            confidence_score=0.80,
            risk_reward_ratio=2.0,
            indicators_aligned=4,
            current_price=1.3550,
            was_executed=True,
            execution_time=datetime.utcnow()
        )
        db_session.add(opportunity)
        db_session.commit()

        # Link to trade
        trade = Trade(
            pair="USDCAD",
            direction="SELL",
            strategy="day_trading",
            size=1.0,
            leverage=30,
            entry_price=1.3550,
            entry_time=datetime.utcnow(),
            stop_loss=1.3600,
            opportunity_id=opportunity.id
        )
        db_session.add(trade)
        db_session.commit()

        opportunity.trade_id = trade.id
        db_session.commit()

        assert opportunity.was_executed is True
        assert opportunity.trade_id == trade.id

    def test_opportunity_to_dict(self, sample_opportunity):
        """Test opportunity serialization."""
        opp_dict = sample_opportunity.to_dict()

        assert opp_dict["pair"] == "USDJPY"
        assert opp_dict["direction"] == "BUY"
        assert opp_dict["confidence_score"] == 0.82
        assert opp_dict["qualifies_for_auto_execute"] is True

    def test_opportunity_expires(self, db_session):
        """Test opportunity expiration."""
        expired = datetime.utcnow() - timedelta(minutes=10)
        opportunity = Opportunity(
            pair="EURJPY",
            direction="BUY",
            strategy="scalping",
            timeframe="1m",
            proposed_entry_price=162.50,
            proposed_size=1.0,
            proposed_stop_loss=162.00,
            leverage=50,
            confidence_score=0.80,
            risk_reward_ratio=2.0,
            indicators_aligned=4,
            current_price=162.50,
            expires_at=expired
        )
        db_session.add(opportunity)
        db_session.commit()

        assert opportunity.expires_at < datetime.utcnow()

    def test_opportunity_repr(self, sample_opportunity):
        """Test string representation."""
        repr_str = repr(sample_opportunity)
        assert "Opportunity" in repr_str
        assert sample_opportunity.pair in repr_str

    def test_filter_opportunities_by_strategy(self, db_session):
        """Test querying opportunities by strategy."""
        opp1 = Opportunity(
            pair="EURUSD", direction="BUY", strategy="scalping", timeframe="1m",
            proposed_entry_price=1.0850, proposed_size=1.0,
            proposed_stop_loss=1.0800, leverage=50,
            confidence_score=0.80, risk_reward_ratio=2.0,
            indicators_aligned=4, current_price=1.0850
        )
        opp2 = Opportunity(
            pair="GBPUSD", direction="SELL", strategy="day_trading", timeframe="1h",
            proposed_entry_price=1.2650, proposed_size=1.0,
            proposed_stop_loss=1.2700, leverage=30,
            confidence_score=0.85, risk_reward_ratio=3.0,
            indicators_aligned=5, current_price=1.2650
        )
        db_session.add_all([opp1, opp2])
        db_session.commit()

        scalping_opps = db_session.query(Opportunity).filter_by(strategy="scalping").all()
        assert len(scalping_opps) == 1
        assert scalping_opps[0].pair == "EURUSD"


class TestModelRelationships:
    """Test relationships between models."""

    def test_opportunity_to_trade_relationship(self, db_session):
        """Test opportunity can link to executed trade."""
        # Create opportunity
        opportunity = Opportunity(
            pair="EURUSD", direction="BUY", strategy="scalping", timeframe="5m",
            proposed_entry_price=1.0850, proposed_size=1.0,
            proposed_stop_loss=1.0800, proposed_take_profit=1.0950,
            leverage=50, confidence_score=0.85, risk_reward_ratio=2.0,
            indicators_aligned=4, current_price=1.0850
        )
        db_session.add(opportunity)
        db_session.commit()

        # Execute trade
        trade = Trade(
            pair="EURUSD", direction="BUY", strategy="scalping",
            size=1.0, leverage=50, entry_price=1.0850,
            entry_time=datetime.utcnow(), stop_loss=1.0800,
            take_profit=1.0950, opportunity_id=opportunity.id
        )
        db_session.add(trade)
        db_session.commit()

        # Link back
        opportunity.was_executed = True
        opportunity.trade_id = trade.id
        opportunity.execution_time = datetime.utcnow()
        db_session.commit()

        assert trade.opportunity_id == opportunity.id
        assert opportunity.trade_id == trade.id

    def test_position_to_trade_relationship(self, db_session):
        """Test position can link to originating trade."""
        trade = Trade(
            pair="GBPUSD", direction="SELL", strategy="day_trading",
            size=1.0, leverage=30, entry_price=1.2650,
            entry_time=datetime.utcnow(), stop_loss=1.2700
        )
        db_session.add(trade)
        db_session.commit()

        position = Position(
            pair="GBPUSD", direction="SELL", size=1.0, leverage=30,
            entry_price=1.2650, stop_loss=1.2700,
            trade_id=trade.id
        )
        db_session.add(position)
        db_session.commit()

        assert position.trade_id == trade.id


class TestModelConstraints:
    """Test model constraints and validation."""

    def test_trade_requires_pair(self, db_session):
        """Test trade requires pair field."""
        trade = Trade(
            direction="BUY",
            strategy="scalping",
            size=1.0,
            leverage=50,
            entry_price=1.0850,
            entry_time=datetime.utcnow(),
            stop_loss=1.0800
        )
        db_session.add(trade)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_position_requires_pair(self, db_session):
        """Test position requires pair field."""
        position = Position(
            direction="BUY",
            size=1.0,
            leverage=50,
            entry_price=1.0850,
            stop_loss=1.0800
        )
        db_session.add(position)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_opportunity_requires_confidence_score(self, db_session):
        """Test opportunity requires confidence score."""
        opportunity = Opportunity(
            pair="EURUSD",
            direction="BUY",
            strategy="scalping",
            timeframe="5m",
            proposed_entry_price=1.0850,
            proposed_size=1.0,
            proposed_stop_loss=1.0800,
            leverage=50,
            risk_reward_ratio=2.0,
            indicators_aligned=4,
            current_price=1.0850
        )
        db_session.add(opportunity)

        with pytest.raises(IntegrityError):
            db_session.commit()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
