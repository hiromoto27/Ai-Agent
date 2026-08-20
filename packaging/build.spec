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

REPO_ROOT = Path(SPECPATH).resolve().parent

a = Analysis(
    [str(REPO_ROOT / "packaging" / "entrypoint.py")],
    pathex=[str(REPO_ROOT)],
    binaries=[],
    datas=[
        (str(REPO_ROOT / "config" / "policy.default.yaml"), "config"),
    ],
    hiddenimports=[
        "docx",
        "pptx",
        "openpyxl",
        "reportlab",
        "httpx",
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
