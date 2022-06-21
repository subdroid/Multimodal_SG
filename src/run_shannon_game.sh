#!/bin/bash
source ../virt/bin/activate
python3 scripts/shannon_game.py
rm -rf huggingface
mkdir huggingface

 