#!/bin/bash

# List all installed packages and filter those starting with nvidia
packages=$(pip list --format=freeze | grep '^nvidia' | cut -d '=' -f 1)

# Uninstall each nvidia package
for package in $packages; do
    echo "Uninstalling $package"
    pip uninstall -y $package
done

