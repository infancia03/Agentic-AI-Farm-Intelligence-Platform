#!/bin/bash
# Run this ONCE to install Prophet properly
# prophet requires pystan which needs extra steps on some systems

echo "Installing Prophet dependencies..."

# Step 1: pystan (Prophet's backend)
pip install pystan==3.9.1

# Step 2: Prophet itself
pip install prophet==1.1.5

# Step 3: verify
python -c "from prophet import Prophet; print('Prophet installed successfully')"
