# EvalScope使用教程，包括训练、评估、压测

全过程教程可参考：[train-eval.ipynb](./train-eval.ipynb)

## 安装环境

```bash
pip install -r requirement.txt
```

## 启动训练

**数据集处理**

```bash
python ./ms-swift-train/alpaca2swift_dataset.py
```


---

单卡

```bash
bash train_gpu0.sh
```

---

多卡分布式训练

```bash
bash train_deepspeed.sh
```

## evalscope工具来评估模型

### 1、基本使用

```bash
bash ./evalscope-eval/cli/easy.sh
```

如果本地磁盘空间不足，使用下面的命令行把数据集下载到本地磁盘

```bash
wget https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/benchmark/data.zip
unzip data.zip
```

然后把dataset-args参数中local_path修改成自己的保存数据集地址，然后运行下面的代码。

**需要注意的是，gsm8k的数据集评测时默认的prompt_template为Question: {query}\nLet's think step by step\nAnswer:。测试过后发现enable_thinking设置为false比true要高些，所以可以默认为false，但是其他的没有cot提示的测试集可以设置为true，尤其是数学推理的时候**

```bash
bash evalscope-eval/cli/gsm8k.sh
bash evalscope-eval/cli/multi.sh
```

### 2、模型API服务评测

先运行下面的代码连接服务端口

```bash
python ./evalscope-eval/api_model/url.py 
```

再开启一个新的terminal运行下面的代码

```bash
bash ./evalscope-eval/api_model/eval_api_eval.sh
```

### 3、模型推理性能压测

```bash
bash evalscope-eval/swanlab/perf.sh
```

压测结果可以查看链接👉[SwanLab](https://swanlab.cn/@LiXinYu/perf_benchmark/runs/k0flil25zyxgt1097asz8/chart)
