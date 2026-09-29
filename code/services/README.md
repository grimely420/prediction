# Linux Systemd Services

These systemd service files allow the prediction system to run automatically as background services on Linux.

## Service Overview

| Service | Description | Runs As | Port |
|---------|-------------|---------|------|
| `bitcoin-collector` | Collects BTC prices every 5s | User | - |
| `bnb-collector` | Collects BNB prices every 5s | User | - |
| `bitcoin-api` | BTC prediction API | User | 5001 |
| `bnb-api` | BNB prediction API | User | 5002 |
| `bitcoin-generator` | Makes BTC predictions every 15min | User | - |
| `bnb-generator` | Makes BNB predictions every 5min | User | - |

## Installation


# Navigate to services directory
cd /home/$USER/prediction/services

# Run install script as root
sudo ./install_services.sh


## Usage

### Start All Services

sudo systemctl start bitcoin-collector bnb-collector
sudo systemctl start bitcoin-api bnb-api
sudo systemctl start bitcoin-generator bnb-generator


### Enable Auto-Start on Boot

sudo systemctl enable bitcoin-collector bnb-collector
sudo systemctl enable bitcoin-api bnb-api
sudo systemctl enable bitcoin-generator bnb-generator


### Check Status

# Check all services
sudo systemctl status bitcoin-collector
sudo systemctl status bitcoin-api
sudo systemctl status bitcoin-generator
sudo systemctl status bnb-collector
sudo systemctl status bnb-api
sudo systemctl status bnb-generator

# Or check all at once
sudo systemctl list-units --type=service | grep -E "(bitcoin|bnb)"


### View Logs

# Real-time log monitoring
sudo tail -f /var/log/prediction-system/btc-collector.log
sudo tail -f /var/log/prediction-system/btc-api.log
sudo tail -f /var/log/prediction-system/btc-generator.log
sudo tail -f /var/log/prediction-system/bnb-collector.log
sudo tail -f /var/log/prediction-system/bnb-api.log
sudo tail -f /var/log/prediction-system/bnb-generator.log


### Stop Services

sudo systemctl stop bitcoin-generator bnb-generator
sudo systemctl stop bitcoin-api bnb-api
sudo systemctl stop bitcoin-collector bnb-collector


### Restart Services

sudo systemctl restart bitcoin-collector
sudo systemctl restart bitcoin-api
sudo systemctl restart bitcoin-generator


## Service Dependencies


bitcoin-collector  →  bitcoin-api  →  bitcoin-generator
     (5s)                  (API)            (15min)

bnb-collector      →  bnb-api      →  bnb-generator
     (5s)                  (API)            (5min)


## Troubleshooting

### Service Won't Start

# Check detailed error
sudo journalctl -u bitcoin-api -n 50

# Check permissions
ls -la /home/$USER/prediction/
ls -la /var/log/prediction-system/


### API Not Responding

# Check if port is listening
sudo netstat -tlnp | grep 5001
sudo netstat -tlnp | grep 5002

# Test API
curl http://localhost:5001/health
curl http://localhost:5002/health


### Predictions Not Being Made

# Check generator logs
sudo tail -f /var/log/prediction-system/btc-generator.log

# Check if API is accessible from generator
sudo systemctl status bitcoin-api


## File Locations

- **Services**: `/etc/systemd/system/`
- **Logs**: `/var/log/prediction-system/`
- **Code**: `/home/$USER/prediction/`
- **Data**: `/home/$USER/prediction/bitcoin/bitcoin_prices.db`
- **Models**: `/home/$USER/prediction/bitcoin/models/`

## Customization

To change the prediction intervals, edit the generator service files:


sudo systemctl edit --full bitcoin-generator


Modify the `ExecStart` line to pass custom arguments (if supported by the generator script).

## Uninstall


# Stop and disable all services
sudo systemctl stop bitcoin-collector bnb-collector
sudo systemctl stop bitcoin-api bnb-api
sudo systemctl stop bitcoin-generator bnb-generator

sudo systemctl disable bitcoin-collector bnb-collector
sudo systemctl disable bitcoin-api bnb-api
sudo systemctl disable bitcoin-generator bnb-generator

# Remove service files
sudo rm /etc/systemd/system/bitcoin-*.service
sudo rm /etc/systemd/system/bnb-*.service

# Reload systemd
sudo systemctl daemon-reload

