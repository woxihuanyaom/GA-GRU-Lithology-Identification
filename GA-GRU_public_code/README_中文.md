# GA-GRU代码公开包

本文件夹对应第四版论文的方法与实验代码，供后续创建公开仓库使用。目前尚未创建 GitHub 仓库，也没有对外上传。

## 先看这几个文件

- `run_demo.py`：Spyder 中直接打开运行，使用独立生成的合成数据测试流程。
- `run_experiment.py`：对本地获授权数据按井分别训练，使用论文中已选定的模型参数。
- `run_search.py`：在完整预处理与 SMOTE-Tomek 流程下比较 GA、随机搜索和 TPE。
- `configs/`：主实验参数、独立搜索设置及严格区间敏感性设置。
- `gagru/`：网络、预处理、窗口、平衡、评价等实际实现。
- `research_scripts/`：保留论文相关原始实验、消融、统计与补充验证脚本。
- `docs/SOURCE_MAP.md`：每类实验对应哪个原始脚本、还依赖哪些非公开文件。
- `docs/DATA_AND_CODE_AVAILABILITY.txt`：中英文数据与代码可用性声明。

## 安装与检查

推荐 Python 3.12。在本目录打开终端，执行：

```text
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python run_experiment.py --demo
python run_search.py --demo
python -m pytest -q tests
```

如使用 Spyder，确认 Spyder 使用安装了这些依赖的 Python 环境，然后运行 `run_demo.py`。CPU 可以运行；论文 GPU 耗时不能直接与此 CPU 示例比较。

第一个示例只使用一口合成井、一个划分、一个训练种子和两个训练轮次。搜索示例只使用一个搜索种子，每种方法七个候选及一个训练轮次。示例数据完全人工生成，不来自真实井、真实井的变换结果或真实样本子集。示例分数只用于检查代码能否运行，不是论文实验结果。

## 使用自己的获授权数据

每口井一个 CSV，字段为 `well_id, depth, segment_id, interval_id, class_id, MSFL, LLS, LLD, DEN, DT, GR, NPHI`。详细要求见 `docs/DATA_FORMAT.md`。

深度单位为 m，步长 0.125 m；三条电阻率单位为 ohm m；DEN 为 g/cm^3；DT 为 us/ft；GR 为 API；NPHI 为无量纲小数，不能直接填百分数。`interval_id` 是真实记录的完整标注区间，不是逐点随机编号。标签允许每口井独立编号，代码会映射成该井连续的零起始类别编号。

```text
python run_experiment.py --data-dir "D:/my_private_data" --output "D:/my_private_results/main" --device cpu
python run_search.py --data-dir "D:/my_private_data" --output "D:/my_private_results/search" --device cpu
```

输出目录应为空。支持 CUDA 时将 `cpu` 改为 `cuda`。完整独立搜索为三种方法、三个搜索种子、每次24个候选，每个候选还需在所有输入井分别训练，因此耗时显著高于示例。公开适配入口每次创建新运行，不实现历史任务的断点恢复；原始脚本保留了对应逻辑。

主实验入口直接复用原主实验的特征准备、训练、重训练和指标计算函数。搜索入口复用原独立搜索的 GA、随机搜索、TPE 算子、初始候选和候选排序规则。没有另写一套模型代替论文代码。模型配置不会根据示例分数或新测试集分数自动改写。

## 需要准确理解的范围

论文主实验是井内随机中心插值，各井独立训练，单次运行的目标标签不交叉，但输入曲线上下文可以交叠。不能将其解释成盲井预测，也不能据此声称能够预测整块连续未知井段。完整区间隔离的敏感性代码另行保留。主实验配置曾用于开发过程，因此重复评价不能当作独立嵌套模型选择估计。

真实测井、岩性标签、划分文件、逐点预测、权重、私有实验快照、原稿和审稿材料均未打包。输出的预测 CSV 含有本地数据的深度和标签，后续自行运行后不要将这些文件上传到公开仓库；`.gitignore` 已设置相应排除规则。

只有代码不能独立复现论文数值，还需要依法获授权的原始数据和相应实验快照。`research_scripts/` 中的历史脚本依赖这些非公开文件，不能在空目录中全部依次运行。公开示例和适配入口用于让读者检查实现、在自己获授权的数据上执行流程。

当前没有替作者选定软件许可证。正式建仓库时补充实际仓库地址、版本标记及权利人选定的许可证即可；公开可见并不自动等同于开源授权。
