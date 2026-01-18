# План разработки системы из 11 торговых ботов

## Общая архитектура

```
Layer 1 (Data Collection)     Layer 2 (Analysis)      Layer 3 (Risk)
┌─────────────────────┐      ┌─────────────────┐      ┌─────────────┐
│ 1. Price Scanner    │─────▶│ 5. Signal Gen   │─────▶│ 8. Risk Mgr │
│ 2. Volume Analyzer  │─────▶│ 6. Pattern Det  │      └─────────────┘
│ 3. Sentiment Track  │─────▶│ 7. Strategy Eng │              │
│ 4. Whale Watcher    │      └─────────────────┘              ▼
└─────────────────────┘                               Layer 4 (Execution)
                                                      ┌─────────────────┐
                                                      │ 9. Order Exec   │
                                                      │ 10. Position Mgr│
                                                      └─────────────────┘
                                                              │
                                                              ▼
                                                      Layer 5 (Monitor)
                                                      ┌─────────────────┐
                                                      │ 11. Monitor     │
                                                      └─────────────────┘
```

## Активы для торговли (начальные)
- BTCUSDT
- ETHUSDT
- BNBUSDT
- SOLUSDT
- ADAUSDT

## Статус разработки

| # | Бот | Статус | Файл | Бэктест |
|---|-----|--------|------|---------|
| 1 | Price Scanner | ✅ Готов | `bots/price_scanner.py` | ✅ 335k свечей |
| 2 | Volume Analyzer | ✅ Готов | `bots/volume_analyzer.py` | ⏳ |
| 3 | Sentiment Tracker | ✅ Готов | `bots/sentiment_tracker.py` | ⏳ |
| 4 | Whale Watcher | ✅ Готов | `bots/whale_watcher.py` | ⏳ |
| 5 | Signal Generator | ✅ Готов | `bots/signal_generator.py` | ⏳ |
| 6 | Pattern Detector | ✅ Готов | `bots/pattern_detector.py` | ⏳ |
| 7 | Strategy Engine | ✅ Готов | `bots/strategy_engine.py` | ⏳ |
| 8 | Risk Manager | ✅ Готов | `bots/risk_manager.py` | ⏳ |
| 9 | Order Executor | ✅ Готов | `bots/order_executor.py` | ⏳ |
| 10 | Position Manager | ✅ Готов | `bots/position_manager.py` | ⏳ |
| 11 | Monitor & Report | ✅ Готов | `bots/monitor.py` | ⏳ |

### Скачанные данные (2020-2026)
| Актив | 1h свечей | 4h свечей | 1d свечей |
|-------|-----------|-----------|-----------|
| BTCUSDT | 52,985 | 13,254 | 2,210 |
| ETHUSDT | 52,985 | 13,254 | 2,210 |
| BNBUSDT | 52,985 | 13,254 | 2,210 |
| SOLUSDT | 47,639 | 11,916 | 1,987 |
| ADAUSDT | 52,985 | 13,254 | 2,210 |
| **ИТОГО** | **259,579** | **64,932** | **10,827** |

**База данных:** `data/prices.db` (54.38 MB)

---

## Фаза 1: Data Collection Layer

### Bot 1: Price Scanner
**Цель:** Сбор и хранение ценовых данных

**Функционал:**
- Скачивание исторических данных за 4 года (2020-2024)
- Real-time WebSocket для живых цен
- Хранение в SQLite база данных
- Поддержка интервалов: 1m, 5m, 15m, 1h, 4h, 1d

**Данные OHLCV:**
- Open, High, Low, Close, Volume
- Timestamp в UTC

**API:** Binance Public API (klines endpoint)

**Файлы:**
- `bots/price_scanner.py` - основной код
- `data/historical/` - исторические данные
- `data/prices.db` - SQLite база

---

### Bot 2: Volume Analyzer
**Цель:** Анализ объемов торгов

**Индикаторы:**
- VWAP (Volume Weighted Average Price)
- OBV (On Balance Volume)
- Volume Profile
- Аномальные всплески объема (>2σ от среднего)

**Сигналы:**
- `volume_spike` - резкий рост объема
- `accumulation` - накопление
- `distribution` - распределение

---

### Bot 3: Sentiment Tracker
**Цель:** Отслеживание рыночных настроений

**Источники:**
- Fear & Greed Index (alternative.me API)
- Binance Funding Rate
- Long/Short Ratio
- Open Interest

**Сигналы:**
- `extreme_fear` - возможность покупки
- `extreme_greed` - возможность продажи
- `funding_rate_extreme` - перегрев рынка

---

### Bot 4: Whale Watcher
**Цель:** Отслеживание крупных игроков

**Мониторинг:**
- Крупные ордера в orderbook (>$100k)
- Большие транзакции на блокчейне
- Движение средств с/на биржи

**API:**
- Binance Orderbook WebSocket
- Whale Alert API (опционально)

---

## Фаза 2: Analysis Layer

### Bot 5: Signal Generator
**Цель:** Генерация торговых сигналов

**Индикаторы:**
- RSI (14) - перекупленность/перепроданность
- MACD (12, 26, 9) - тренд и моментум
- Bollinger Bands (20, 2) - волатильность
- EMA (9, 21, 50, 200) - тренд
- Stochastic RSI - точки входа

**Логика сигналов:**
```python
LONG_SIGNAL:
  - RSI < 30 AND
  - MACD histogram > 0 AND
  - Price > EMA21

SHORT_SIGNAL:
  - RSI > 70 AND
  - MACD histogram < 0 AND
  - Price < EMA21
```

---

### Bot 6: Pattern Detector
**Цель:** Распознавание графических паттернов

**Свечные паттерны:**
- Doji, Hammer, Shooting Star
- Engulfing (бычий/медвежий)
- Morning/Evening Star
- Three White Soldiers / Three Black Crows

**Уровни:**
- Support/Resistance автодетекция
- Pivot Points
- Fibonacci уровни

**Формации:**
- Double Top/Bottom
- Head & Shoulders
- Triangle, Wedge, Flag

---

### Bot 7: Strategy Engine
**Цель:** Объединение всех сигналов

**Скоринг:**
```python
score = (
    signal_generator_score * 0.3 +
    pattern_detector_score * 0.2 +
    volume_analyzer_score * 0.2 +
    sentiment_score * 0.15 +
    whale_activity_score * 0.15
)

if score > 0.7: STRONG_BUY
if score > 0.5: BUY
if score < -0.5: SELL
if score < -0.7: STRONG_SELL
```

---

## Фаза 3: Risk Management Layer

### Bot 8: Risk Manager
**Цель:** Защита капитала

**Правила:**
- Максимум 1-2% риска на сделку
- Максимум 5% в одной позиции
- Максимум 20% общего капитала в позициях
- Stop Loss обязателен для каждой сделки

**Расчет размера позиции:**
```python
position_size = (account_balance * risk_percent) / (entry_price - stop_loss)
```

**Стоп-лоссы:**
- ATR-based: SL = Entry - (2 * ATR)
- Percent-based: SL = Entry * 0.98 (2%)
- Structure-based: под/над уровнем поддержки

---

## Фаза 4: Execution Layer

### Bot 9: Order Executor
**Цель:** Исполнение ордеров

**Типы ордеров:**
- Market Order - немедленное исполнение
- Limit Order - по указанной цене
- Stop-Limit - стоп с лимитом

**Контроль:**
- Slippage < 0.1%
- Retry logic при ошибках
- Rate limiting

---

### Bot 10: Position Manager
**Цель:** Управление открытыми позициями

**Функционал:**
- Trailing Stop (подтягивание стопа)
- Частичное закрытие (TP1, TP2, TP3)
- Break-even перенос стопа
- Time-based exit (макс. время в позиции)

---

## Фаза 5: Monitoring Layer

### Bot 11: Monitor & Report
**Цель:** Мониторинг и отчетность

**Метрики:**
- Total P&L
- Win Rate
- Profit Factor
- Sharpe Ratio
- Max Drawdown
- Average Trade Duration

**Алерты:**
- Telegram бот для уведомлений
- Email для дневных отчетов
- Dashboard для визуализации

---

## Бэктестинг

### Данные
- Период: 01.01.2020 - 31.12.2024 (4+ года)
- Активы: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, ADAUSDT
- Интервалы: 1h (основной), 15m, 4h (дополнительные)

### Процесс
1. **Train период:** 2020-2022 (оптимизация параметров)
2. **Validation:** 2023 (проверка без изменений)
3. **Test:** 2024 (финальная проверка)

### Метрики успеха
- Win Rate > 55%
- Profit Factor > 1.5
- Sharpe Ratio > 1.0
- Max Drawdown < 20%

### Walk-Forward Analysis
- Окно оптимизации: 6 месяцев
- Окно тестирования: 2 месяца
- Скользящее по всему периоду

---

## Порядок разработки

1. ✅ **Price Scanner** - скачивание данных
2. ⏳ **Signal Generator** - базовые сигналы
3. ⏳ **Risk Manager** - защита капитала
4. ⏳ **Order Executor + Position Manager** - исполнение
5. ⏳ **Volume Analyzer + Pattern Detector** - улучшение
6. ⏳ **Strategy Engine** - объединение
7. ⏳ **Whale Watcher + Sentiment Tracker** - доп. данные
8. ⏳ **Monitor & Report** - мониторинг

---

## Binance API Endpoints

### REST API
- `GET /api/v3/klines` - исторические свечи
- `GET /api/v3/ticker/price` - текущая цена
- `GET /api/v3/depth` - orderbook
- `POST /api/v3/order` - создание ордера

### WebSocket Streams
- `<symbol>@kline_<interval>` - свечи в реальном времени
- `<symbol>@depth` - orderbook updates
- `<symbol>@trade` - сделки

### Rate Limits
- 1200 requests/minute (REST)
- 5 messages/second (WebSocket)

---

## Файловая структура

```
trading-system/
├── bots/
│   ├── __init__.py
│   ├── price_scanner.py      # Bot 1
│   ├── volume_analyzer.py    # Bot 2
│   ├── sentiment_tracker.py  # Bot 3
│   ├── whale_watcher.py      # Bot 4
│   ├── signal_generator.py   # Bot 5
│   ├── pattern_detector.py   # Bot 6
│   ├── strategy_engine.py    # Bot 7
│   ├── risk_manager.py       # Bot 8
│   ├── order_executor.py     # Bot 9
│   ├── position_manager.py   # Bot 10
│   └── monitor.py            # Bot 11
├── data/
│   ├── historical/           # CSV файлы с историей
│   └── prices.db             # SQLite база
├── backtest/
│   ├── engine.py             # Движок бэктеста
│   ├── results/              # Результаты тестов
│   └── optimize.py           # Оптимизация параметров
├── dashboard/
│   └── templates/
│       └── index.html        # Веб-интерфейс
└── BOTS_DEVELOPMENT_PLAN.md  # Этот файл
```

---

## Заметки

### Текущий прогресс
- [x] Настроен Binance Testnet API
- [x] Базовый dashboard создан
- [x] Скачивание исторических данных (335,338 свечей за 6 лет)
- [x] Bot 1: Price Scanner - готов
- [ ] Bot 5: Signal Generator - в работе
- [ ] Bot 8: Risk Manager
- [ ] Bot 9-10: Order Executor + Position Manager

### Важные решения
- Используем SQLite для простоты (можно мигрировать на PostgreSQL)
- Paper trading перед реальной торговлей
- 1% риск на сделку как начальная настройка

---

*Последнее обновление: 2026-01-18*
