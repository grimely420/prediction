#!/bin/bash
echo "===== UPTIME / BOOT ====="
uptime
echo
echo "===== TEMPERATURES ====="
for z in /sys/class/thermal/thermal_zone*/temp; do
  name=$(cat ${z%/*}/type 2>/dev/null)
  t=$(cat $z 2>/dev/null)
  [ -n "$t" ] && echo "$name: $((t/1000))°C"
done
for h in /sys/class/hwmon/hwmon*/temp*_input; do
  [ -f "$h" ] || continue
  name=$(cat ${h%input}label 2>/dev/null || basename $h)
  chip=$(cat ${h%/*}/name 2>/dev/null)
  t=$(cat $h)
  echo "$chip $name: $((t/1000))°C"
done | sort -u | head -30
echo
echo "===== CPU THROTTLE / FREQ ====="
grep MHz /proc/cpuinfo | sort -u | head -4
echo
echo "===== DISK SMART ====="
for d in /dev/sd? /dev/nvme?n1; do
  [ -b "$d" ] || continue
  echo "--- $d ---"
  sudo -n smartctl -H "$d" 2>/dev/null | grep -iE "overall|result" || echo "(smartctl unavailable or needs sudo)"
done
echo
echo "===== NVME ERROR LOG ====="
for d in /dev/nvme?n1; do
  [ -b "$d" ] || continue
  sudo -n smartctl -a "$d" 2>/dev/null | grep -iE "critical|error|temperature|percentage used|media" | head -8
done
echo
echo "===== DMESG ERRORS (last boot) ====="
dmesg -l err,crit,alert,emerg 2>/dev/null | head -20 || sudo -n dmesg -l err,crit,alert,emerg 2>/dev/null | head -20
echo
echo "===== RECENT KERNEL HW EVENTS ====="
dmesg 2>/dev/null | grep -iE "mce|machine check|edac|ecc|thermal|throttl|i/o error|ata.*error|nvme.*error|reset" | tail -15
echo
echo "===== MEMORY ====="
free -h
echo
echo "===== DISK FS ====="
df -h / /home 2>/dev/null
echo
echo "===== FAILED SERVICES ====="
systemctl --failed --no-legend
echo
echo "===== PREDICTION SERVICES ====="
systemctl is-active prediction-api.service prediction-collector@btc.service prediction-collector@bnb.service prediction-collector@hype.service prediction-predictor@btc.service prediction-predictor@bnb.service prediction-predictor@hype.service
echo
echo "===== JOURNAL ERRORS SINCE BOOT ====="
journalctl -b -p err --no-pager 2>/dev/null | grep -ivE "fprintd|bluetooth|thermald|spice|colord" | tail -15
