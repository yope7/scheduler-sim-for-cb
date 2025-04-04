#!/bin/bash

# プロセスIDを取得して強制終了する
ps aux | grep wandb | grep -v grep | awk '{print $2}' | xargs kill -9

