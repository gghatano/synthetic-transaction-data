#!/usr/bin/env bash
# Pulse Physiology Engine（Kitware, Apache-2.0）を公式 GitLab からビルドする（issue #21）。
# ソース・ビルド・Java・専用の Python 環境はすべて data/pulse/（git 管理外）に置く。
# PyPI の pulse-engine は Kitware の公式パッケージと確認できないため使わない。
#
# 必要なもの: C++17 のコンパイラ（GCC 9 以降）、make、uv（cmake は uvx で使う）
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=$ROOT/data/pulse
TAG=REL_4_4_0
mkdir -p "$WORK"
cd "$WORK"

[ -d src ] || git clone -q --depth 1 --branch $TAG https://gitlab.kitware.com/physiology/engine.git src

# データ生成（Data.xlsx → 患者・薬剤などの JSON）に Java が要る。Eclipse Temurin JDK 21 を手元に置く
if [ ! -x jdk/bin/java ]; then
  curl -sSL -o jdk.tar.gz "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.12.1%2B1/OpenJDK21U-jdk_x64_linux_hotspot_21.0.12.1_1.tar.gz"
  echo "ce79869e1307ed8ee1e2baa86a412b1eb5b75d10a01006d788a6f968bcfaee94  jdk.tar.gz" | sha256sum -c
  mkdir -p jdk && tar -xzf jdk.tar.gz -C jdk --strip-components=1 && rm jdk.tar.gz
fi

# Pulse の Python API とデータ生成用の Python 環境（pip が要る）
if [ ! -x venv/bin/python ]; then
  uv venv -q --python 3.12 venv
  uv pip install -q --python venv/bin/python -r src/requirements.txt pip
fi

export JAVA_HOME=$WORK/jdk VIRTUAL_ENV=$WORK/venv PATH=$WORK/venv/bin:$WORK/jdk/bin:$PATH
PY=$WORK/venv/bin/python
uvx --from cmake cmake -U 'Python*' -S src -B build -DCMAKE_BUILD_TYPE=Release \
  -DPulse_JAVA_API=ON -DPulse_GEN_DATA=ON -DPulse_PYTHON_API=ON -DPulse_JAVA_HOME_DIR="$JAVA_HOME" \
  -DPython3_FIND_VIRTUALENV=ONLY -DPython_FIND_VIRTUALENV=ONLY -DPython3_EXECUTABLE="$PY" -DPython_EXECUTABLE="$PY"
uvx --from cmake cmake --build build -j "$(nproc)"
echo "built: $WORK/build/install"
