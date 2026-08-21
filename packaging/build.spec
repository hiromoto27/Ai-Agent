# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: сборка Windows-приложения из ai_agent в .exe.

PyInstaller не кросс-компилирует — сборку нужно запускать на Windows
(локально или в CI, см. .github/workflows/build-windows-exe.yml).

Локальная сборка на Windows:
    pip install -e ".[all]"
    pyinstaller packaging/build.spec --noconfirm

Результат: dist/AI-Agent/AI-Agent.exe (папка с exe и зависимостями —
режим "onedir": быстрее стартует и реже ложно триггерит антивирусы,
чем однофайловый exe).
"""

from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata

REPO_ROOT = Path(SPECPATH).resolve().parent

# huggingface_hub (и часть его зависимостей) читает своё же package-метаданные
# через importlib.metadata в рантайме (версия для user-agent заголовка и
# т.п.). PyInstaller по умолчанию тянет только .py-код через hiddenimports,
# а .dist-info с метаданными — нет, из-за чего такие вызовы в frozen-сборке
# либо тихо деградируют, либо (в других версиях/местах, не гарантировано)
# падают с PackageNotFoundError. copy_metadata — стандартный fix для этого
# класса пакетов, дешёвая подстраховка независимо от того, воспроизвели мы
# конкретный сбой локально или нет.
hf_metadata = []
for _pkg in ("huggingface_hub", "requests", "filelock", "packaging", "tqdm", "pyyaml", "fsspec"):
    try:
        hf_metadata += copy_metadata(_pkg)
    except Exception:
        pass  # пакета может не быть в окружении сборки — пропускаем, не критично

a = Analysis(
    [str(REPO_ROOT / "packaging" / "entrypoint.py")],
    pathex=[str(REPO_ROOT)],
    binaries=[],
    datas=[
        (str(REPO_ROOT / "config" / "policy.default.yaml"), "config"),
        *hf_metadata,
    ],
    hiddenimports=[
        "docx",
        "pptx",
        "openpyxl",
        "reportlab",
        "httpx",
        "huggingface_hub",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AI-Agent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AI-Agent",
)
