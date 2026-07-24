#!/usr/bin/env bash
# 一键编译 Beamer Slide 脚本
#
# 用法: bash scripts/build_slides.sh（无参数）
# 输入: paper/slides/main.tex 及其引用的素材
# 输出: paper/slides/main.pdf（pdflatex 两轮编译，更新页码/大纲/目录）
# 依赖: 本机需安装 pdflatex；编译失败时打印 main.log 末尾 20 行

# 获取脚本所在目录的绝对路径，确保从任何地方调用都能正确工作
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SLIDES_DIR="$( cd "${SCRIPT_DIR}/../paper/slides" && pwd )"

echo "📂 切换工作目录到 Slide 源码文件夹: ${SLIDES_DIR}"
cd "${SLIDES_DIR}" || { echo "❌ 找不到 slides 目录！"; exit 1; }

echo "⏳ 正在进行第 1 轮 LaTeX 编译..."
pdflatex -interaction=nonstopmode main.tex > /dev/null

if [ $? -ne 0 ]; then
    echo "❌ 第 1 轮编译失败，正在输出错误日志末尾几行..."
    tail -n 20 main.log
    exit 1
fi

echo "⏳ 正在进行第 2 轮 LaTeX 编译（以更新页码、大纲和目录）..."
pdflatex -interaction=nonstopmode main.tex > /dev/null

if [ $? -eq 0 ]; then
    echo "✅ 编译成功！"
    echo "📄 生成的目标 PDF 路径："
    echo "   ${SLIDES_DIR}/main.pdf"
else
    echo "❌ 第 2 轮编译失败，正在输出错误日志..."
    tail -n 20 main.log
    exit 1
fi
