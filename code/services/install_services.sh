#!/bin/bash
# Install systemd services for the prediction system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="/etc/systemd/system"
LOG_DIR="/var/log/prediction-system"

echo "=========================================="
echo "Installing Prediction System Services"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

# Create log directory
echo "Creating log directory: $LOG_DIR"
mkdir -p "$LOG_DIR"
chown chain-deaction:chain-deaction "$LOG_DIR"

# Copy service files
echo "Installing service files..."
cp "$SCRIPT_DIR/prediction-api.service" "$SYSTEM_DIR/"
cp "$SCRIPT_DIR/prediction-collector@.service" "$SYSTEM_DIR/"
cp "$SCRIPT_DIR/prediction-predictor@.service" "$SYSTEM_DIR/"

# Reload systemd
echo "Reloading systemd..."
systemctl daemon-reload

echo ""
echo "=========================================="
echo "Services installed successfully!"
echo "=========================================="
echo ""
echo "To start all services:"
echo "  sudo systemctl start prediction-collector@btc prediction-collector@bnb prediction-collector@hype"
echo "  sudo systemctl start prediction-api"
echo "  sudo systemctl start prediction-predictor@btc prediction-predictor@bnb prediction-predictor@hype"
echo ""
echo "To enable auto-start on boot:"
echo "  sudo systemctl enable prediction-collector@btc prediction-collector@bnb prediction-collector@hype"
echo "  sudo systemctl enable prediction-api"
echo "  sudo systemctl enable prediction-predictor@btc prediction-predictor@bnb prediction-predictor@hype"
echo ""
echo "To check status:"
echo "  sudo systemctl status prediction-collector@btc"
echo "  sudo systemctl status prediction-api"
echo "  sudo systemctl status prediction-predictor@btc"
echo ""
echo "To view logs:"
echo "  sudo tail -f /var/log/prediction-system/api.log"
echo "  sudo tail -f /var/log/prediction-system/btc-collector.log"
echo "  sudo tail -f /var/log/prediction-system/btc-predictor-loop.log"
echo ""
