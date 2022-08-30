#!/bin/bash

[ -d osm-ml-venv ] && rm -rf osm-ml-venv && echo Removing existing venv...

python3 -m venv osm-ml-venv
source osm-ml-venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install -r requirements.txt

echo
echo ---
echo Created venv \"osm-ml-venv\".  Activate by running \"source osm-ml-venv/bin/activate\".
