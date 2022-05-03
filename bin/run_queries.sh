#!/bin/bash

source ./venv/bin/activate

python main.py --addr "localhost"
	       -e 9200
	       -t ./assets/2015_descriptions.txt
	       -i "test_trials"
	       -o"./outputs/embedding.txt"
	       -m "./model/"
	       -q "embedding"
	       -n "2.15"
	       -s 1000

-a "localhost" -e 9200 -t ./assets/2015_descriptions.txt  -i "test_trials"  -o"./outputs/embedding.txt"  -m "./model/"  -q "embedding"  -n "2.15" -s 1000
