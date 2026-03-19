#!/bin/bash
# 在 Thesis 目录下用 xelatex 编译 sdumain.tex
set -e
cd "$(dirname "$0")"

# 确保能找到 xelatex：优先使用常见 TeX 路径（不依赖当前 shell 的 PATH）
TEXBIN=""
if [ -x "/Library/TeX/texbin/xelatex" ]; then
  TEXBIN="/Library/TeX/texbin"
elif ls /usr/local/texlive/*/bin/*/xelatex 1>/dev/null 2>&1; then
  TEXBIN="$(dirname "$(ls /usr/local/texlive/*/bin/*/xelatex 2>/dev/null | head -1)")"
fi
if [ -n "$TEXBIN" ]; then
  export PATH="$TEXBIN:$PATH"
fi

if ! command -v xelatex &>/dev/null; then
  echo "错误: 未找到 xelatex。"
  echo "请安装 MacTeX: brew install --cask mactex"
  echo "若已安装，请检查 /Library/TeX/texbin 或 /usr/local/texlive 下是否存在 xelatex。"
  exit 1
fi

echo ">>> xelatex (1/4)"
xelatex -interaction=nonstopmode -file-line-error sdumain.tex
echo ">>> bibtex"
bibtex sdumain
echo ">>> xelatex (2/4)"
xelatex -interaction=nonstopmode -file-line-error sdumain.tex
echo ">>> xelatex (3/4)"
xelatex -interaction=nonstopmode -file-line-error sdumain.tex
echo ">>> 完成: sdumain.pdf"
