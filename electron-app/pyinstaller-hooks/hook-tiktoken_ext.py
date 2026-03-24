from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules('tiktoken_ext')
datas = collect_data_files('tiktoken_ext')
