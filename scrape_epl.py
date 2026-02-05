# -*- coding: utf-8 -*-
"""
英超 2025-2026 赛季完整数据爬取脚本
功能：
1. 爬取英超 25-26 赛季所有已完成比赛数据
2. 每5轮爬取后自动检查数据完整性
3. 自动补充缺失数据
4. 多线程并行处理，提高爬取速度

使用方法：
python scrape_epl_25_26.py
"""

import time
import sys
import io
import pandas as pd
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# 可选依赖：用于设置进程优先级
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdout.reconfigure(line_buffering=True)

# [文件内容与原始文件相同，这里省略完整内容以节省空间]
# 注意：这是一个完整的爬虫脚本，用于展示数据来源
