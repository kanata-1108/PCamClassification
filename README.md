# PCamClassification

## 使用方法
1. 使用するデータを格納する`data`ディレクトリを作成し、その中にh5ファイルを格納する
2. `h5_to_npy.py`でh5ファイルをnpyファイルに変換する
3. (必要であれば)`check_npy.ipynb`でnpyファイルの中身を確認する
4. `main.py`で学習を実行する。
5. 学習で得られたパラメータ`model_param.pth`と推論用プログラム`eval.py`で推論を実行する