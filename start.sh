#!/bin/bash
set -e
source ./venv/bin/activate
cd ./main/trading-bot
python bot.py
