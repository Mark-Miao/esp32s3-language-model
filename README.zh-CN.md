# esp32s3-language-model

[English](README.md) | **简体中文**

在电脑上训练一个约 250 万参数的语言模型，然后在 ESP32-S3 上运行。

本仓库是**工具链**，不是模型发布。它包含数据处理、训练代码、量化器和固件；
**不包含**数据集和预训练权重——两者都由用户本地生成。

```
语料 ──► BPE 分词器 ──► uint16 ids ──► PyTorch 训练 ──► int8 model.bin ──► 固件
```

## 设计

| 项 | 值 |
|---|---|
| 架构 | decoder-only Transformer，pre-norm RMSNorm，SiLU MLP，权重共享 |
| 参数 | 2,582,560（d=160，6 层，5 头，ctx 512，MLP 4×） |
| 分词器 | byte-level BPE，4096 个 token，特殊标记：用户 / 助手 / 结束 |
| 训练 | 纯 CPU，约 750 万 token，6000 步，4 核需要数小时 |
| 量化 | int8 权重，按输出通道缩放，激活值保持 float32 |
| 模型文件 | 2.5 MiB（int8，v2 格式，内嵌词表） |
| 固件 | ESP-IDF，USB-CDC 协议，KV cache，权重放在 PSRAM |
| 硬件 | ESP32-S3，16 MB flash + 8 MB octal PSRAM（N16R8） |

固件把模型嵌进 app 镜像，所以整个部署就是一个 `eslm.bin` 加上 bootloader
和分区表。

## 环境要求

- Python 3.10+，约 4 GB 内存
- ESP-IDF v5.3 或更新版本（用于固件）
- ESP32-S3 开发板，带 8 MB octal PSRAM（N16R8 或更大）

## 快速开始

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 下载语料（会打印许可声明，见 DATA.md）、构建、训练
scripts/train.sh --accept-license

# 导出 int8 权重并编译固件
. $IDF_PATH/export.sh
scripts/build_firmware.sh

# 烧录（用标着 USB 的口，不是 UART 口）
cd firmware && idf.py -p /dev/ttyACM0 -b 921600 flash
```

## 使用自己的语料

`tools/build_dialogues.py --custom path.jsonl` 每行接受一个 JSON 对象，支持三种格式：

```json
{"turns": [["user", "你好"], ["assistant", "你好呀"]]}
{"conversations": [{"from": "human", "value": "你好"}, {"from": "gpt", "value": "你好呀"}]}
{"conversation": [{"human": "你好", "assistant": "你好呀"}]}
```

轮次会被过滤（2–8 轮，每轮 2–40 字，以中文为主）。然后：

```bash
python tools/build_dialogues.py --custom my.jsonl
python tools/train_bpe.py --raw data/raw.txt --vocab-size 4096 --out models/tokenizer
python tools/build_ids.py --dialogues data/dialogues.jsonl --tokenizer models/tokenizer
python tools/train.py --config configs/chat-lccc.json
```

你需对自己使用的语料拥有合法权利。

## 串口协议

固件监听原生 USB-CDC（板子上标 `USB` 的口），日志输出到 UART0。命令以 `\n`
结尾，生成结果以 UTF-8 流式返回。

```
PING                      -> PONG eslm fw=1 ctx=512
INFO                      -> INFO params=... ctx=... dtype=int8 ...
CHAT <n> <temp> <id...>   -> 生成文本，最后输出 "[OK] n=.. ms=.. tok_s=.."
RESET                     -> OK
```

`CHAT` 接收 `models/tokenizer` 产出的 token id，作为提示喂入，一直生成到
`<|e|>` 或达到 `n` 个 token。最小 pyserial 客户端：

```python
import serial
from tokenizers import ByteLevelBPETokenizer

tok = ByteLevelBPETokenizer("models/tokenizer/vocab.json", "models/tokenizer/merges.txt")
tok.add_special_tokens(["<|u|>", "<|a|>", "<|e|>"])
ids = [tok.token_to_id("<|u|>")] + tok.encode("你好", add_special_tokens=False).ids \
      + [tok.token_to_id("<|a|>")]

ser = serial.Serial("/dev/ttyACM0", 115200, timeout=1)
ser.write(f"CHAT 80 0.6 {' '.join(map(str, ids))}\n".encode())
print(ser.read_until(b"[OK]").decode("utf-8", "replace"))
```

## 模型文件格式（v2）

```
"ESLM" | u32 version=2 | u32 dtype(0=f32, 1=int8) | u32 vocab, d, n_layer, n_head, ctx
       | u32 special_u, special_a, special_e | u32 vocab_blob_bytes
u32 token_offsets[vocab+1] | u8 token_bytes[vocab_blob_bytes]
权重：tok_emb, pos_emb，然后每层 rms1, wq, wk, wv, wo, rms2, w1, w2，最后 rms_f
```

二维矩阵按行存放（行 = 输出通道）。int8 模式下每个矩阵前置 `f32 scales[out]`
并补齐到 4 字节边界；一维 norm 向量保持 float32。`tools/export.py --check`
会重新读取文件并与 PyTorch 对比（f32 时 `max|diff| = 0`）。

## 数据

数据集由 `tools/download_data.py` 下载，从不提交到仓库。来源、许可和处理义务见
[DATA.md](DATA.md)。

- **LCCC** —— 研究用途许可，微博用户生成文本
- **Slim-LCCC** —— Apache-2.0，派生自 LCCC
- **Muice** —— CC-BY-NC-4.0，禁止商业使用，需显式启用

## 局限

- 250 万参数只能学会短中文对话的"形状"，没有知识。输出听起来像回事，但经常是
  错的或空洞的。
- 训练语料是即兴对话，输出可能粗鲁或无意义，不适合面向客户的场景。
- 设备端速度取决于上下文长度和你的板子，请自行实测。2.58M int8 模型在 S3 上
  是计算受限的（每秒个位数到十几 token）。

## 法律

本项目采用 **AGPL-3.0** 许可（见 [LICENSE](LICENSE)）。商业使用**仅**在遵守该
许可证条款时被允许：如果你修改本软件并提供给他人（包括作为网络服务），必须以
同一许可证提供完整的对应源码。

**预期用途**：研究、教育、爱好者实验。

**适用法律文件**：本节引用的法律、行政法规和部门规章，均为中华人民共和国现行
有效的规范性文件。

**禁止用途**：生成违法内容；诈骗、欺诈或欺骗性冒充；对真实人物的深度伪造；
骚扰；监控；医疗、法律或金融建议；安全关键或生产系统。使用本软件必须遵守
以下强制性要求，不得有下列行为：

- 生成、复制、发布、传播危害国家安全、煽动颠覆国家政权、恐怖主义、极端主义、
  淫秽色情、赌博、暴力、诈骗、虚假信息等违法内容，或违反《网络信息内容生态治理
  规定》制作、复制、发布违法和不良信息；
- 侵犯他人名誉权、肖像权、隐私权或个人信息权益，或非法收集、使用、加工、传输
  他人个人信息；
- 冒充真实人物、伪造身份，或制作深度合成内容未按《互联网信息服务深度合成管理
  规定》《人工智能生成合成内容标识办法》添加显著标识；
- 用于电信网络诈骗、虚假广告、误导性营销、骚扰、跟踪、监控或歧视；
- 侵害未成年人权益，制作、复制、发布、传播危害未成年人身心健康的网络信息
  （《未成年人保护法》网络保护专章）；
- 使用非法获取或未经授权的语料训练、分发模型。

**免责声明**：本软件及你训练出的模型均按“原样”提供，不附带任何形式的担保。
你对自己使用的数据、训练的模型及其生成内容负全部责任。使用者必须履行以下义务：

- 向公众提供生成式人工智能服务，应遵守《生成式人工智能服务管理暂行办法》，
  依法履行安全评估、内容标识等义务；
- 生成合成内容应遵守《互联网信息服务深度合成管理规定》《人工智能生成合成内容
  标识办法》，按规定添加显著标识；
- 具有舆论属性或社会动员能力的服务，还应按《互联网信息服务算法推荐管理规定》
  履行算法备案手续（《生成式人工智能服务管理暂行办法》第十七条亦作此要求）；
- 网络信息内容应遵守《网络信息内容生态治理规定》《未成年人保护法》，不得制作、
  传播违法和不良信息，不得危害未成年人身心健康；
- 训练数据与个人信息处理应遵守《网络安全法》《数据安全法》《个人信息保护法》
  《著作权法》等，不得使用非法获取、未经授权或侵犯他人合法权益的数据；
- 情节严重的违法行为可能触犯《刑法》第 217 条（侵犯著作权罪）、第 246 条
  （侮辱、诽谤罪）、第 253 条之一（侵犯公民个人信息罪）、第 266 条（诈骗罪）、
  第 286 条之一（拒不履行信息网络安全管理义务罪）、第 287 条之一/之二（非法
  利用信息网络罪、帮助信息网络犯罪活动罪）、第 363/364 条（淫秽物品相关犯罪）等；
- 本仓库不分发任何数据或模型，训练与部署产生的法律责任由使用者自行承担；
- 本节仅为合规提示，不构成法律意见；具体问题请咨询专业律师。

**责任限制与法律依据**

本项目作者仅在 AGPL-3.0 许可下提供源代码，不提供生成式人工智能服务、不分发
训练数据或模型、不运营任何网络平台。据此：

- 《生成式人工智能服务管理暂行办法》第二条适用于向公众提供生成式人工智能服务的
  提供者，并规定未向公众提供此类服务的不适用本办法。本项目本身不提供此类服务；
  你自行训练、部署并向公众提供服务时，你才是该办法意义上的服务提供者，应自行
  履行相应义务。
- 《民法典》第一千一百六十五条采过错责任原则。作者未参与你的数据收集、模型训练、
  部署和内容发布行为，对由此产生的损害不存在过错。
- 《数据安全法》《个人信息保护法》的义务主体是数据处理者。你处理数据时即为处理者，
  应自行取得处理的合法性基础并履行告知、同意等义务。
- 《著作权法》项下，训练语料与生成内容的权利状态由你自行核查并承担相应后果。
- 刑事责任以行为人实施犯罪行为为前提（《刑法》第十三条）。作者未实施相关行为；
  使用者利用本项目实施犯罪的，由行为人依法承担刑事责任。
- AGPL-3.0 第十五条、第十六条明确不提供担保并限制责任，第十七条为上述条款的解释规则。
- 依据《民法典》第五百零六条，造成对方人身损害、或因故意或重大过失造成对方财产
  损失的免责条款无效；本节的声明同样不能免除作者的故意或重大过失责任。

如你不接受上述条款，请勿使用本仓库。

数据集可能包含个人信息，请勿再分发。

ESP32 和 ESP32-S3 是乐鑫科技（Espressif Systems）的商标。本项目与乐鑫无关联，
也未获其背书。ModelScope 是阿里巴巴集团的服务，本项目与之无关联。

## 目录结构

```
README.zh-CN.md             中文说明
configs/chat-lccc.json      训练配置
tools/
  download_data.py          下载语料（带许可确认）
  build_dialogues.py        清洗/归一化对话，或使用 --custom
  train_bpe.py              训练 BPE 分词器
  build_ids.py              对话 -> uint16 token 流
  train.py                  PyTorch 训练
  export.py                 检查点 -> model.bin（f32 / int8，v1 / v2）
  bpe_utils.py              BPE 词表 -> 设备端字节表
firmware/
  CMakeLists.txt, partitions.csv, sdkconfig.defaults
  main/{main.c, infer.c, infer.h}
scripts/
  train.sh                  端到端：数据 + 训练
  build_firmware.sh         导出 + idf.py build
```

## 贡献

提交 Pull Request 即表示你同意按 AGPL-3.0（与本仓库其他部分相同）授权你的贡献。

## 致谢

- 清华大学 COAI 实验室的 LCCC / CDial-GPT
- ModelScope 上 Slim-LCCC-zh 与 Muice 数据集的维护者
- ESP-IDF 与 ESP32-S3 硬件文档
- `llama2.c` 项目（让 C 语言推理实现变得易读）
