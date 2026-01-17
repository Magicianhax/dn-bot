/**
 * Ethereal Trading Dashboard - Frontend Application
 */

class Dashboard {
    constructor() {
        this.ws = null;
        this.wsConnected = false;
        this.updateInterval = null;
        this.currentSettings = {};
        
        this.elements = {
            // Status
            botStatus: document.getElementById('bot-status'),
            wsStatus: document.getElementById('ws-status'),
            
            // Buttons
            btnStart: document.getElementById('btn-start'),
            btnStop: document.getElementById('btn-stop'),
            btnCloseAll: document.getElementById('btn-close-all'),
            btnEditSettings: document.getElementById('btn-edit-settings'),
            btnSaveSettings: document.getElementById('btn-save-settings'),
            btnCancelSettings: document.getElementById('btn-cancel-settings'),
            
            // Accounts
            acc1Balance: document.getElementById('acc1-balance'),
            acc2Balance: document.getElementById('acc2-balance'),
            acc1Positions: document.getElementById('acc1-positions'),
            acc2Positions: document.getElementById('acc2-positions'),
            acc1Address: document.getElementById('acc1-address'),
            acc2Address: document.getElementById('acc2-address'),
            
            // Settings Display
            settingPairs: document.getElementById('setting-pairs'),
            settingPosition: document.getElementById('setting-position'),
            settingLeverage: document.getElementById('setting-leverage'),
            settingSltp: document.getElementById('setting-sltp'),
            settingHold: document.getElementById('setting-hold'),
            settingMaxTrades: document.getElementById('setting-max-trades'),
            
            // Settings Modal
            settingsModal: document.getElementById('settings-modal'),
            pairsCheckboxes: document.getElementById('pairs-checkboxes'),
            inputUseFullBalance: document.getElementById('input-use-full-balance'),
            inputPosition: document.getElementById('input-position'),
            inputMinBalance: document.getElementById('input-min-balance'),
            inputLeverage: document.getElementById('input-leverage'),
            inputSl: document.getElementById('input-sl'),
            inputTp: document.getElementById('input-tp'),
            inputMinHold: document.getElementById('input-min-hold'),
            inputMaxHold: document.getElementById('input-max-hold'),
            inputMinDelay: document.getElementById('input-min-delay'),
            inputMaxDelay: document.getElementById('input-max-delay'),
            inputMaxTrades: document.getElementById('input-max-trades'),
            
            // Logs
            logsContainer: document.getElementById('logs-container'),
            btnClearLogs: document.getElementById('btn-clear-logs'),
            
            // Stats
            statPnl: document.getElementById('stat-pnl'),
            statVolume: document.getElementById('stat-volume'),
            statBuyVol: document.getElementById('stat-buy-vol'),
            statSellVol: document.getElementById('stat-sell-vol'),
            statTrades: document.getElementById('stat-trades'),
            
            // Active Trade
            activeTrade: document.getElementById('active-trade-content'),
            
            // Prices
            pricesGrid: document.getElementById('prices-grid'),
            
            // History
            historyList: document.getElementById('history-list'),
            
            // Footer
            lastUpdate: document.getElementById('last-update'),
        };
        
        this.init();
    }
    
    async init() {
        // Connect WebSocket
        this.connectWebSocket();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Initial data fetch
        await this.fetchAll();
        
        // Start periodic updates
        this.updateInterval = setInterval(() => this.fetchAll(), 5000);
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            this.wsConnected = true;
            this.elements.wsStatus.classList.add('connected');
            console.log('WebSocket connected');
        };
        
        this.ws.onclose = () => {
            this.wsConnected = false;
            this.elements.wsStatus.classList.remove('connected');
            console.log('WebSocket disconnected');
            
            // Reconnect after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };
        
        // Keep-alive ping
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'bot_status':
                this.updateBotStatus(data.running);
                break;
            case 'trade_open':
                this.fetchActiveTrades();
                this.fetchTrades();
                break;
            case 'trade_close':
                this.fetchActiveTrades();
                this.fetchTrades();
                this.fetchVolume();
                break;
            case 'log':
                this.addLog(data.log);
                break;
            case 'settings_updated':
                this.fetchStatus();
                break;
            case 'pong':
                // Keep-alive response
                break;
        }
    }
    
    setupEventListeners() {
        this.elements.btnStart.addEventListener('click', () => this.startBot());
        this.elements.btnStop.addEventListener('click', () => this.stopBot());
        this.elements.btnCloseAll.addEventListener('click', () => this.closeAll());
        
        // Settings modal
        this.elements.btnEditSettings.addEventListener('click', () => this.openSettingsModal());
        this.elements.btnSaveSettings.addEventListener('click', () => this.saveSettings());
        this.elements.btnCancelSettings.addEventListener('click', () => this.closeSettingsModal());
        
        // Close modal on overlay click
        this.elements.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.elements.settingsModal) {
                this.closeSettingsModal();
            }
        });
        
        // Use full balance toggle
        this.elements.inputUseFullBalance.addEventListener('change', (e) => {
            this.elements.inputPosition.disabled = e.target.checked;
        });
        
        // Clear logs
        this.elements.btnClearLogs.addEventListener('click', () => this.clearLogs());
    }
    
    async fetchAll() {
        try {
            await Promise.all([
                this.fetchStatus(),
                this.fetchActiveTrades(),
                this.fetchPositions(),
                this.fetchPrices(),
                this.fetchTrades(),
                this.fetchVolume(),
            ]);
            
            this.elements.lastUpdate.textContent = `Last update: ${new Date().toLocaleTimeString()}`;
        } catch (error) {
            console.error('Failed to fetch data:', error);
        }
    }
    
    async fetchStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            // Bot status
            this.updateBotStatus(data.bot_running);
            
            // Account balances
            if (data.accounts) {
                this.elements.acc1Balance.textContent = this.formatMoney(data.accounts.account1_balance || 0);
                this.elements.acc2Balance.textContent = this.formatMoney(data.accounts.account2_balance || 0);
            }
            
            // Settings
            if (data.settings) {
                this.currentSettings = data.settings;
                this.updateSettingsDisplay(data.settings);
            }
            
            // Stats
            if (data.stats) {
                const pnl = data.stats.total_pnl || 0;
                this.elements.statPnl.textContent = this.formatMoney(pnl);
                this.elements.statPnl.className = `stat-value ${pnl >= 0 ? 'positive' : 'negative'}`;
                this.elements.statTrades.textContent = data.stats.total_trades || 0;
            }
            
            // Volume
            if (data.volume) {
                this.elements.statVolume.textContent = this.formatMoney(data.volume.total_volume || 0);
                this.elements.statBuyVol.textContent = this.formatMoney(data.volume.total_buy_volume || 0);
                this.elements.statSellVol.textContent = this.formatMoney(data.volume.total_sell_volume || 0);
            }
        } catch (error) {
            console.error('Failed to fetch status:', error);
        }
    }
    
    async fetchActiveTrades() {
        try {
            const response = await fetch('/api/trades/active');
            const data = await response.json();
            
            if (data.trades && data.trades.length > 0) {
                const trade = data.trades[0];
                this.renderActiveTrade(trade);
            } else {
                this.elements.activeTrade.innerHTML = '<div class="no-trade">No active trade</div>';
            }
        } catch (error) {
            console.error('Failed to fetch active trades:', error);
        }
    }
    
    renderActiveTrade(trade) {
        const pnl = trade.total_pnl || 0;
        const pnlClass = pnl >= 0 ? 'positive' : 'negative';
        
        const acc1Side = trade.account1_is_long ? 'LONG' : 'SHORT';
        const acc2Side = trade.account1_is_long ? 'SHORT' : 'LONG';
        
        this.elements.activeTrade.innerHTML = `
            <div class="active-trade-compact">
                <div class="trade-info-row">
                    <span class="trade-ticker">${trade.product_id}</span>
                    <span class="trade-side-badge ${acc1Side.toLowerCase()}">${acc1Side}</span>
                    <span class="trade-side-badge ${acc2Side.toLowerCase()}">${acc2Side}</span>
                </div>
                <div class="trade-stats-row">
                    <div class="trade-stat-inline">
                        <span class="stat-label">Entry</span>
                        <span class="stat-value">${this.formatMoney(trade.entry_price)}</span>
                    </div>
                    <div class="trade-stat-inline">
                        <span class="stat-label">Size</span>
                        <span class="stat-value">${trade.size}</span>
                    </div>
                    <div class="trade-stat-inline">
                        <span class="stat-label">PnL</span>
                        <span class="stat-value ${pnlClass}">${this.formatMoney(pnl)}</span>
                    </div>
                    <div class="trade-stat-inline">
                        <span class="stat-label">Hold</span>
                        <span class="stat-value">${trade.hold_time_minutes?.toFixed(1) || 0}m / ${trade.target_hold_minutes?.toFixed(1) || 0}m</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    async fetchPositions() {
        try {
            const response = await fetch('/api/positions');
            const data = await response.json();

            // Update wallet addresses
            if (this.elements.acc1Address) {
                this.elements.acc1Address.textContent = data.account1_address || '';
            }
            if (this.elements.acc2Address) {
                this.elements.acc2Address.textContent = data.account2_address || '';
            }

            this.renderPositions(data.account1, this.elements.acc1Positions);
            this.renderPositions(data.account2, this.elements.acc2Positions);
        } catch (error) {
            console.error('Failed to fetch positions:', error);
        }
    }

    renderPositions(positions, element) {
        if (!positions || positions.length === 0) {
            element.innerHTML = '<div class="no-positions">No positions</div>';
            return;
        }

        element.innerHTML = positions.map(p => {
            const pnl = p.unrealized_pnl || 0;
            const pnlClass = pnl >= 0 ? 'positive' : 'negative';
            const sideClass = p.side === 'LONG' ? 'long' : 'short';
            const size = Math.abs(p.size || 0);
            const leverage = p.leverage || 1;
            const notional = p.notional || 0;
            const entryPrice = p.entry_price || 0;
            const markPrice = p.mark_price || entryPrice;

            return `
                <div class="position-item-detailed">
                    <div class="position-header">
                        <span class="position-ticker">${p.ticker}</span>
                        <span class="position-side ${sideClass}">${p.side}</span>
                    </div>
                    <div class="position-details">
                        <div class="position-detail">
                            <span class="detail-label">Size</span>
                            <span class="detail-value">${size.toFixed(6)}</span>
                        </div>
                        <div class="position-detail">
                            <span class="detail-label">Leverage</span>
                            <span class="detail-value">${leverage}x</span>
                        </div>
                        <div class="position-detail">
                            <span class="detail-label">Notional</span>
                            <span class="detail-value">${this.formatMoney(notional)}</span>
                        </div>
                        <div class="position-detail">
                            <span class="detail-label">Entry</span>
                            <span class="detail-value">${this.formatMoney(entryPrice)}</span>
                        </div>
                        <div class="position-detail">
                            <span class="detail-label">Mark</span>
                            <span class="detail-value">${this.formatMoney(markPrice)}</span>
                        </div>
                        <div class="position-detail">
                            <span class="detail-label">PnL</span>
                            <span class="detail-value ${pnlClass}">${this.formatMoney(pnl)}</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }
    
    async fetchPrices() {
        try {
            const response = await fetch('/api/prices');
            const data = await response.json();
            
            if (data.prices) {
                for (const [ticker, price] of Object.entries(data.prices)) {
                    const el = document.getElementById(`price-${ticker}`);
                    if (el) {
                        el.textContent = this.formatMoney(price);
                    }
                }
            }
        } catch (error) {
            console.error('Failed to fetch prices:', error);
        }
    }
    
    async fetchTrades() {
        try {
            const response = await fetch('/api/trades?limit=20');
            const data = await response.json();
            
            if (data.trades && data.trades.length > 0) {
                this.elements.historyList.innerHTML = data.trades.map(t => this.renderHistoryItem(t)).join('');
            } else {
                this.elements.historyList.innerHTML = '<div class="no-history">No trades yet</div>';
            }
        } catch (error) {
            console.error('Failed to fetch trades:', error);
        }
    }
    
    renderHistoryItem(trade) {
        const pnl = trade.total_pnl || 0;
        const pnlClass = pnl >= 0 ? 'positive' : 'negative';
        const statusClass = (trade.status || '').toLowerCase().replace('_', '-');
        
        const date = trade.opened_at ? new Date(trade.opened_at).toLocaleString() : '-';
        const reason = trade.close_reason || trade.status || '-';
        
        return `
            <div class="history-item ${statusClass}">
                <div class="history-row">
                    <span class="history-ticker">${trade.product_id}</span>
                    <span class="history-pnl ${pnlClass}">${this.formatMoney(pnl)}</span>
                </div>
                <div class="history-meta">
                    <span>${date}</span>
                    <span class="history-reason">${reason}</span>
                </div>
            </div>
        `;
    }
    
    async fetchVolume() {
        try {
            const response = await fetch('/api/volume');
            const data = await response.json();
            
            if (data.summary) {
                this.elements.statVolume.textContent = this.formatMoney(data.summary.total_volume || 0);
                this.elements.statBuyVol.textContent = this.formatMoney(data.summary.total_buy_volume || 0);
                this.elements.statSellVol.textContent = this.formatMoney(data.summary.total_sell_volume || 0);
            }
        } catch (error) {
            console.error('Failed to fetch volume:', error);
        }
    }
    
    updateBotStatus(running) {
        if (running) {
            this.elements.botStatus.classList.add('running');
            this.elements.botStatus.querySelector('.status-text').textContent = 'RUNNING';
            this.elements.btnStart.disabled = true;
            this.elements.btnStop.disabled = false;
        } else {
            this.elements.botStatus.classList.remove('running');
            this.elements.botStatus.querySelector('.status-text').textContent = 'OFFLINE';
            this.elements.btnStart.disabled = false;
            this.elements.btnStop.disabled = true;
        }
    }
    
    async startBot() {
        try {
            this.elements.btnStart.disabled = true;
            this.elements.btnStart.textContent = 'Starting...';
            
            const response = await fetch('/api/bot/start', { method: 'POST' });
            const data = await response.json();
            
            if (data.success) {
                this.updateBotStatus(true);
            } else {
                alert('Failed to start bot: ' + data.message);
            }
        } catch (error) {
            console.error('Failed to start bot:', error);
            alert('Failed to start bot');
        } finally {
            this.elements.btnStart.innerHTML = '<span>▶</span> START';
        }
    }
    
    async stopBot() {
        try {
            this.elements.btnStop.disabled = true;
            this.elements.btnStop.textContent = 'Stopping...';
            
            const response = await fetch('/api/bot/stop', { method: 'POST' });
            const data = await response.json();
            
            if (data.success) {
                this.updateBotStatus(false);
            } else {
                alert('Failed to stop bot: ' + data.message);
            }
        } catch (error) {
            console.error('Failed to stop bot:', error);
            alert('Failed to stop bot');
        } finally {
            this.elements.btnStop.innerHTML = '<span>■</span> STOP';
        }
    }
    
    async closeAll() {
        if (!confirm('Are you sure you want to close all positions?')) {
            return;
        }
        
        try {
            const response = await fetch('/api/bot/close-all', { method: 'POST' });
            const data = await response.json();
            
            if (data.success) {
                await this.fetchAll();
            } else {
                alert('Failed to close positions: ' + data.message);
            }
        } catch (error) {
            console.error('Failed to close positions:', error);
            alert('Failed to close positions');
        }
    }
    
    formatMoney(value) {
        if (value === null || value === undefined) return '$0.00';
        const num = parseFloat(value);
        if (Math.abs(num) >= 1000) {
            return '$' + num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        }
        return '$' + num.toFixed(2);
    }
    
    async openSettingsModal() {
        // Fetch available products for checkboxes
        try {
            const resp = await fetch('/api/products/available');
            const data = await resp.json();
            this.renderPairCheckboxes(data.products || []);
        } catch (e) {
            console.error('Failed to fetch products:', e);
        }
        
        // Populate form with current settings
        const s = this.currentSettings;
        
        // Set selected pairs
        const selectedPairs = (s.trading_pairs || '').split(',').map(p => p.trim());
        document.querySelectorAll('.pair-checkbox input').forEach(cb => {
            cb.checked = selectedPairs.includes(cb.value);
            cb.parentElement.classList.toggle('selected', cb.checked);
        });
        
        this.elements.inputUseFullBalance.checked = s.use_full_balance !== false;
        this.elements.inputPosition.value = s.position_size || 50;
        this.elements.inputPosition.disabled = this.elements.inputUseFullBalance.checked;
        this.elements.inputMinBalance.value = s.min_balance_threshold || 10;
        this.elements.inputLeverage.value = s.leverage || 10;
        this.elements.inputSl.value = s.stop_loss_percent || 5;
        this.elements.inputTp.value = s.take_profit_percent || 5;
        this.elements.inputMinHold.value = s.min_hold_time_minutes || 30;
        this.elements.inputMaxHold.value = s.max_hold_time_minutes || 120;
        this.elements.inputMinDelay.value = s.min_trade_delay_seconds || 60;
        this.elements.inputMaxDelay.value = s.max_trade_delay_seconds || 300;
        this.elements.inputMaxTrades.value = s.max_daily_trades || 100;
        
        this.elements.settingsModal.style.display = 'flex';
    }
    
    renderPairCheckboxes(products) {
        const container = this.elements.pairsCheckboxes;
        container.innerHTML = products.map(p => `
            <label class="pair-checkbox">
                <input type="checkbox" value="${p}">
                <span>${p.replace('USD', '')}</span>
            </label>
        `).join('');
        
        // Add click handlers
        container.querySelectorAll('.pair-checkbox').forEach(label => {
            label.addEventListener('click', (e) => {
                if (e.target.tagName !== 'INPUT') {
                    const cb = label.querySelector('input');
                    cb.checked = !cb.checked;
                }
                label.classList.toggle('selected', label.querySelector('input').checked);
            });
        });
    }
    
    closeSettingsModal() {
        this.elements.settingsModal.style.display = 'none';
    }
    
    async saveSettings() {
        // Get selected pairs from checkboxes
        const selectedPairs = [];
        document.querySelectorAll('.pair-checkbox input:checked').forEach(cb => {
            selectedPairs.push(cb.value);
        });
        
        if (selectedPairs.length === 0) {
            alert('Please select at least one trading pair');
            return;
        }
        
        const newSettings = {
            trading_pairs: selectedPairs.join(','),
            use_full_balance: this.elements.inputUseFullBalance.checked,
            position_size: parseFloat(this.elements.inputPosition.value) || 50,
            min_balance_threshold: parseFloat(this.elements.inputMinBalance.value) || 10,
            leverage: parseInt(this.elements.inputLeverage.value) || 10,
            stop_loss_percent: parseFloat(this.elements.inputSl.value) || 5,
            take_profit_percent: parseFloat(this.elements.inputTp.value) || 5,
            min_hold_time_minutes: parseInt(this.elements.inputMinHold.value) || 30,
            max_hold_time_minutes: parseInt(this.elements.inputMaxHold.value) || 120,
            min_trade_delay_seconds: parseInt(this.elements.inputMinDelay.value) || 60,
            max_trade_delay_seconds: parseInt(this.elements.inputMaxDelay.value) || 300,
            max_daily_trades: parseInt(this.elements.inputMaxTrades.value) || 100,
        };
        
        try {
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newSettings),
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.closeSettingsModal();
                this.currentSettings = newSettings;
                this.updateSettingsDisplay(newSettings);
                this.addLog({ level: 'INFO', message: 'Settings saved successfully' });
            } else {
                alert('Failed to save settings: ' + data.message);
            }
        } catch (error) {
            console.error('Failed to save settings:', error);
            alert('Failed to save settings');
        }
    }
    
    updateSettingsDisplay(settings) {
        this.elements.settingPairs.textContent = settings.trading_pairs || '-';
        
        if (settings.use_full_balance) {
            this.elements.settingPosition.textContent = 'Full Balance';
        } else {
            this.elements.settingPosition.textContent = this.formatMoney(settings.position_size || 0);
        }
        
        this.elements.settingLeverage.textContent = `${settings.leverage || 0}x`;
        this.elements.settingSltp.textContent = `${settings.stop_loss_percent || 0}% / ${settings.take_profit_percent || 0}%`;
        this.elements.settingHold.textContent = `${settings.min_hold_time_minutes || 0}-${settings.max_hold_time_minutes || 0}m`;
        this.elements.settingMaxTrades.textContent = settings.max_daily_trades || 0;
    }
    
    addLog(log) {
        const container = this.elements.logsContainer;
        const time = log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
        const levelClass = `log-${(log.level || 'info').toLowerCase()}`;
        
        const entry = document.createElement('div');
        entry.className = `log-entry ${levelClass}`;
        entry.innerHTML = `<span class="log-time">${time}</span><span class="log-message">${log.message}</span>`;
        
        // Remove placeholder
        const placeholder = container.querySelector('.log-entry:only-child');
        if (placeholder && placeholder.textContent.includes('Waiting')) {
            placeholder.remove();
        }
        
        container.appendChild(entry);
        
        // Auto-scroll to right
        container.scrollLeft = container.scrollWidth;
        
        // Limit logs
        while (container.children.length > 50) {
            container.removeChild(container.firstChild);
        }
    }
    
    clearLogs() {
        this.elements.logsContainer.innerHTML = '<div class="log-entry log-info">Logs cleared</div>';
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});
