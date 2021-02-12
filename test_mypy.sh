#!/bin/bash
serverIp=159.89.154.87
git clone https://github.com/snsnlou2/mypy.git
cd mypy
git checkout release-0.800
git submodule init
git submodule update
cd ..
pip install -U ./mypy
rm -rf ./mypy
python3 typecheck.py
eval "$(ssh-agent -s)"
chmod 600 root_key
ssh-keyscan $serverIp >> ~/.ssh/known_hosts
ssh-add root_key
zip -r mypy_test_cache.zip mypy_test_cache/
yes | scp -i root_key ./mypy_test_cache.zip root@$serverIp:~/cache/pandas-dev---pandas.zip
yes | scp -i root_key ./mypy_test_report.txt root@$serverIp:~/report/pandas-dev---pandas.txt
