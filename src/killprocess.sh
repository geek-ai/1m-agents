#!/bin/bash

for pid in `ps aux | grep main.py | awk '{print $2}'`; do
    kill -9 $pid
done
