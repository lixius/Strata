#!/usr/bin/env bash
# =============================================================================
#  strata-714.sh — 用 ROCm 7.14 容器启动 Strata（MI50 / gfx906）
#
#  用法:
#    ./strata-714.sh                       # 用默认值启动（前台）
#    ./strata-714.sh -q iq3_s -c 256k      # 指定量化 + 256K 上下文
#    ./strata-714.sh -q iq2_xs -c 32k -k   # 32K + 开 KV 流式
#    ./strata-714.sh -v -q iq3_s -c 256k -k # 开图像输入（编码器跑在 CPU 上）
#    ./strata-714.sh -d                    # 后台启动
#    ./strata-714.sh -H 0.0.0.0            # 监听局域网（会强制要求 API_KEY）
#    API_KEY=$(openssl rand -hex 16) ./strata-714.sh -H 0.0.0.0 -q iq3_s -d
#    ./strata-714.sh stop | status
#    ./strata-714.sh logs [engine]         # 跟踪日志（默认容器/服务端日志）
#    ./strata-714.sh list                  # 列出可用量化与 pack 状态
#    ./strata-714.sh pack -q iq3_s         # 建 pack（缺失时）
#    ./strata-714.sh bench -q iq3_s        # 跑标准测速
#    ./strata-714.sh build                 # 重编 build-714（改过 src/ 或 include/ 之后）
#    ./strata-714.sh build-vision          # 编 CPU 图像编码器 strata-vision（开图像前跑一次）
#    ./strata-714.sh -h                    # 帮助
#
#  参数（也可用同名环境变量）:
#    -q iq3_s      量化: iq2_xs | iq3_xxs | iq3_s          (QUANT)
#    -c 256k       上下文: 32k | 128k | 256k 或数字         (CTX)
#    -k [32768]    开 KV 流式，256K 强烈建议               (KV_RESIDENT)
#    -t            开 decode 分解探针（诊断用，略降速）      (DECODE_TIMING)
#    PREFILL_CPU_SHARE=0  关掉 0.1.41 在 gfx906 上被误开的 CPU 分担（1/auto = 引擎默认）
#    ROUTE_TAIL_SKIP=0    关掉 0.1.42 在 gfx906 上被误开的尾部专家跳过（7/8/9 = 开，改输出）
#    ADAPT_OVERLAP=0      0.1.43 在 gfx906 上被误开的自适应拷贝重叠（0.1.43.1 起上游已改为 opt-in；
#                         本脚本仍显式写 0 固定行为，1 = 开，不改输出改时序）
#    NAN_GUARD=1          0.1.43 的 #879 非有限 logits 自愈守卫（0 = 关，窗口图回到 0.1.42 形状）
#    PROMPT_F16=1         prompt 的 16 位 GEMM 走 FP16（0 = 上游默认的 BF16 回退路径，prefill 慢 24%）
#    ATTN_SWIZZLE=1       gfx906 INT8 批量注意力的 query LDS swizzle（0 = 上游默认关）
#    ATTN_REDUCE12=1      gfx906 同款 score 归约 60→16 次 shuffle（0 = 上游默认关）
#    ATTN_DOT2=1          gfx906 INT8 批量注意力走 V_DOT2_F32_F16 内核（0 = 上游 FP32 内核）
#    EXTRA_ENV=K=V,K2=V2  往引擎 env 块塞任意变量（A/B 探针用；同名覆盖上面的具名项）
#    EXTRA_ARGS="--a b"   往引擎 args 末尾追加任意参数（A/B 探针用，如 --adapt-min-gain 3；按空白切分，值里不能有空格）
#    CONV_CACHE_MIB=5120  会话缓存 RAM 预算 MiB（默认 5120，覆盖满 256K；0 = 关）  (CONV_CACHE_SLOTS=2)
#    CONV_CACHE_MIN_FREE_MIB=2560  停放/恢复时必须留的物理内存
#    -v [cpu]      开图像输入，默认关闭；编码器只能跑 CPU     (VISION)
#    -H 127.0.0.1  监听地址；非本机地址必须配 API_KEY       (HOST)
#    -p 8081       端口                                     (PORT)
#    -d / -f       后台 / 前台
#    API_KEY=...   对外暴露时必填
#    PREFILL=auto  POOL_WORKERS=7  SPEC=4  SPEC_MIN_P=0.7  VRAM_RESERVE_MIB=1024
#    ADAPT_EVERY=  专家缓存自适应档：空 = 引擎默认（每 4 窗口换 96 个专家）；0 = 静态驻留
#    MODEL_ID=qwen3.8-flash                 对外模型 id（配置里的 model_name）
#    VISION_TOKENS=300  VISION_THREADS=3    每图最多 image token / 编码器线程数
#
#  设计要点（都是实测踩出来的，改动前请先读）:
#    * 绝不加 --mmap-experts —— 那是关闭常驻专家 arena 的 A/B 开关，会让 prefill 掉 2.3 倍
#    * --prefill auto 远优于 512 / 2048
#    * 256K 必须配 --kv-resident，否则专家缓存会缩水 ~13%
#    * 容器必须 --network host（服务绑 127.0.0.1，端口映射连不上容器 loopback）
#    * 容器必须 --group-add 110（render 组），只加 video 会让 HIP 看不到设备
#    * 配置里必须写 "backend": "hip" —— 否则 WebUI 的 GPU load / VRAM / 温度 / 功耗全显示 "–"，
#      而且 min_free_vram_mib 的"GPU 被占用"检查会静默失效（server 靠这个键决定读 amdgpu sysfs
#      还是 NVML；NVML 在 A 卡上根本不存在）。setup.py 自己会写，手写配置最容易漏。
#    * 引擎的环境变量只能写进配置的 "env" 块 —— docker exec 不继承宿主 shell 的 export，
#      所以 `STRATA_DECODE_TIMING=1 ./strata-714.sh` 是无效的（-t 会替你写进 env 块）。
#    * 图像（-v）只能走 CPU 编码器：AMD/HIP 后端上游还没有 GPU 图像编码器（setup.py:
#      "the AMD backend has no GPU image encoder yet"），--vision gpu 在 AMD 上等于关闭图像。
#      代价：每图约 3-10 秒 CPU、编码器常驻 ~0.7-1.1 GB 内存、每图默认最多 300 image token
#      （GPU 路径是 1024，所以小字/图表细节更弱）；显存占用≈0，文本速度不受影响。
#      -v 会把 --vision 写进 args 并加 "vision" 段；不加 -v 时配置里完全没有这些，行为与以前一致。
#      编码器线程数默认 3（引擎已占 POOL_WORKERS=7，别给满，否则编码时并发 decode 会掉速）。
#    * 不要写 --adapt-every 0（原来照抄 AMD 文档给 7900 XTX 的配置）。那张卡专家几乎全在显存里，
#      静态驻留更可预测；MI50 只有 54% 专家进显存（13334 槽 / 25.32 GiB），其余靠 8 核 CPU 算，
#      静态档实测 decode 掉一大截（2026-10-09 实测，iq3_s / 32K / 62GB）：
#        - tools/calibrate.py 的专家档环节：静态 37.6 vs 引擎默认 57.9 tok/s（短上下文 512 token 回答）
#        - 同 prompt 直接 A/B（6 次加载）：37.8/37.6 vs 57.8/57.4 tok/s（128 / 512 token 回答）
#        - 服务器路径 bench-decode-fixed 同款代码 prompt（1024 token 回答）：48.1 -> 60.1 tok/s（+25%）
#        - 26K 上下文 + 代码问题（512 token 回答）：45.6 -> 54.0 tok/s（+18%）
#        - 29K 上下文纯文档摘要（512 token 回答）是唯一例外：51.5 -> 48.1（-6.6%，单次样本，
#          同臂内波动就有 ±7%，视为噪声）；15K 文档摘要 +1.2%
#      结论：除非你只做长文档摘要，否则用引擎默认自适应档。要复现旧行为：ADAPT_EVERY=0 ./strata-714.sh ...
#    * STRATA_PREFILL_CPU_SHARE=0（0.1.41 新增，本脚本默认）：0.1.41 上游把这个「<1024 token 的 prompt
#      分给空闲 CPU」的开关在 CUDA 单卡上默认打开，并说明 AMD/HIP 构建应当保持关闭 —— 但 gfx906 走的
#      正是 CUDA 分支，于是被误开（上游 PR #1732 报告的正是这个 gate bug）。实测 MI50 32GB：开了没有任何
#      收益，却让输出不再与 0.1.40.3 逐字节相同（同会话 A/B：关掉 = 32K/256K 各 10/10 完全一致；
#      开着 = 32K 差 1/5、256K 差 2/5 的 reasoning 措辞，content 仍 5/5 相同，与上游 KL ~0.004 一致）。
#      所以这里显式关掉。要用上游默认行为：PREFILL_CPU_SHARE=1 ./strata-714.sh ...
#      （0.1.42 已把 gate 修好 —— dc0ee3ac 给 arm_cpu_share 补上 || defined(STRATA_HIP_GFX906)，
#        gfx906 不再被误开；=0 保留着纯粹为了逐字节可复现，且防上游再回归。）
#    * STRATA_ROUTE_TAIL_SKIP=0（0.1.42 新增，本脚本默认）：上游把「verify 窗口里所有 token 都只以
#      rank>=7 路由的漏掉专家直接跳过」在 CUDA serve 上默认打开（b4e3407b，称 8-16GB 卡 decode
#      +15..25%，输出与 0.1.41 略有差别 KL ~0.003-0.006）。默认门又写成 #if !defined(STRATA_USE_HIP)
#      —— 与 0.1.41 的 arm_cpu_share 同一个 gate bug，gfx906 会被误开（上游注释明说 HIP 应保持关闭）。
#      本机先关掉保持逐字节；=7 值得单独 A/B（本机专家不全驻留，CPU 池在算漏掉专家，方向上有利）。
#      0.1.43 顺带加了 #1884 的「≥90% 专家已缓存就不默认开」线（STRATA_TAIL_SKIP_MAX_CACHED 可调），
#      本机 iq3_s 只有 ~54% 驻留，够不到那条线；env 里显式 =0 仍然必要。
#    * STRATA_ADAPT_OVERLAP=0（0.1.43 新增，本脚本固定写 0）：上游 0.1.43 曾把「自适应专家档的换入拷贝
#      与下一个 verify 窗口重叠」在 Linux-CUDA 上默认打开（f95f2eab），而门写成 #if defined(STRATA_USE_HIP)
#      → HIP 关、#elif _WIN32 → 关、#else → 开；gfx906 不定义 STRATA_USE_HIP，于是落进 #else **被误开**
#      （同一个 gate 模式的第三次：0.1.41 arm_cpu_share、0.1.42 ROUTE_TAIL_SKIP、0.1.43 ADAPT_OVERLAP）。
#      **0.1.43.1 已把默认改成全后端 opt-in**（#2010：RTX 5090 Laptop / 5070 Ti 上它掉 19-21% decode），
#      0.1.45 仍是 opt-in，所以现在这个 0 只是固定窗口时序、不是修 bug。本机实测无可测收益
#      （32K 同会话：开 60.4 / 关 58.3 / 0.1.42 60.8 中位 tok/s，全在 ±7% 噪声内）。要试：
#      ADAPT_OVERLAP=1 ./strata-714.sh ...；生效时日志里会有 "adaptive overlap" 一行，关掉后该行消失。
#    * STRATA_NAN_GUARD=1（0.1.43 新增，本脚本显式写出）：#879 的非有限 logits 守卫 + 自愈。每个 verify
#      窗口的图里多一个只读 kernel（logits_nonfinite_rows，16 blocks/行）扫 head_logits，发现 NaN 就终止
#      该请求、清掉被污染的 live/checkpoints/parked 状态，下一请求从 token 0 重读 —— 治的是 MI50 社区
#      报过的 "!!!!" 刷屏（#606/#1463）。默认开、**没有任何编译门**，gfx906 也执行；代价 ~5 µs/窗口，
#      正常路径输出逐字节不变（上游 3 对逐字节验证）。这里显式写 1 是为了：(a) 防上游改默认，
#      (b) 想复现 0.1.42 的窗口图时 NAN_GUARD=0 即可。撞上时日志是
#      "strata verify: non-finite logits (#879) ..."，服务端重试一次仍失败则回 503。
#    * 会话缓存（--conversation-cache-*）：上游默认 0 = 关，本脚本默认给 **5120 MiB / 2 槽**。
#      它吃的是**主机内存**（不是显存）：快照 = running state + checkpoints + 用到的 K/V 页 +
#      draft 层 K/V。实测大小（本机 256K 配置 / iq3_s / int8 KV / --spec 4）：
#          462 token → 232 MiB      10,254 → 375 MiB      29,868 → 773 MiB
#          99,335 → 2236 MiB        198,631 → 3684 MiB    外推 262,144 → ~4.6 GiB
#      高段拟合 ≈ 787 MiB + 14.9 KiB/token（边际就是 KV 的 14.1 KiB/token）。
#      admission 读 /proc/meminfo 的 MemAvailable，要求 `MemAvailable >= 快照 + min-free`
#      （min-free 默认 2560 MiB）。本机 MemAvailable 9221 MiB，所以：
#          2048 → 只够 ~86K token（99K 会话实测被拒：2236 MiB > 2048）
#          4096 → ~227K（保守档，停满后 MemAvailable 5.1 GiB）
#          5120 → ~297K（默认档，覆盖满 256K；停满后 4.1 GiB，仍高于 floor 1.5 GiB）
#          6656 → 理论天花板（停满后正好等于 floor）
#      **不要用上游文档示例的 8192**（停满后 ~1 GiB，低于 floor，停放直接被拒）。
#      只有**多个会话来回切换**（如 controller/worker 交替）才有收益；单窗口连续追问用默认的
#      --prompt-cache 6 就够了，此时 CONV_CACHE_MIB=0 关掉。
#      观测：/metrics 的 conversation_cache（enabled/parked/bytes/evictions/restores/reused_tokens）
#      与引擎日志的 "skip parking (physical RAM admission ...)" / "evicted N parked conversation(s)"。
#    * --pcie-frac 0 保留：calibrate 把 0.00 选为最优（0.00/0.20/0.35/0.55/0.75 =
#      57.1/56.7/53.5/55.6/53.9 tok/s，全在噪声内），与 AMD 文档一致。
#    * --spec-min-p 0.7（原 0.5）：calibrate 选了 0.70（扫描 0.50->0.70 = 57.2->58.8 tok/s），
#      单引擎交错复测（同一个引擎、per-request 调，6 轮 × 2 长度）0.70 在 12/12 次配对里都赢：
#      128 token 回答 57.2->59.2（+3.5%），512 token 回答 56.4->57.9（+2.7%）。CPU 慢的机器该提高这个门限
#      （每次多猜一个 token 都要多算几个专家），和 calibrate.py 的说明一致。
#    * --pool-workers 7 保留：calibrate 测 7/5/4/2 = 57.7/58.2/57.4/54.8 tok/s，5 只快 0.7%，按噪声处理。
# =============================================================================
set -euo pipefail

# ------------------------------- 可调默认值 ---------------------------------
QUANT="${QUANT:-iq3_s}"            # iq2_xs | iq3_xxs | iq3_s
CTX="${CTX:-32768}"                # 32768 | 262144，或直接写数字
KV_RESIDENT="${KV_RESIDENT:-0}"    # 0 = 关；>0 = 开 KV 流式并保留该格数（建议 32768）
PORT="${PORT:-8081}"
HOST="${HOST:-127.0.0.1}"          # 监听地址：127.0.0.1 = 仅本机；0.0.0.0 = 局域网（必须配 API_KEY）
PREFILL="${PREFILL:-auto}"         # auto | 512 | 2048 ...
POOL_WORKERS="${POOL_WORKERS:-7}"  # 8 核 CPU 用 7
SPEC="${SPEC:-4}"
SPEC_MIN_P="${SPEC_MIN_P:-0.7}"    # 0.7：calibrate 选的；单引擎 6 轮交错复测 12/12 都赢 0.5（+3.5%/+2.7%）
ADAPT_EVERY="${ADAPT_EVERY:-}"     # 空 = 引擎默认自适应档（每 4 窗口换 96 个专家，decay 0.7）；0 = 静态驻留
VRAM_RESERVE_MIB="${VRAM_RESERVE_MIB:-1024}"
DECODE_TIMING="${DECODE_TIMING:-0}" # 1 = 打开引擎的 decode 分解探针（诊断用，略降速）
# 0.1.41 把 STRATA_PREFILL_CPU_SHARE 在 CUDA 单卡上默认打开（<1024 token 的 prompt 分给空闲 CPU）。
# 上游本意是「AMD/HIP 构建保持关闭」，但 gfx906 走的正是 CUDA 分支，于是被误开（上游 PR #1732）。
# 后果：输出与 0.1.40.3 不再逐字节相同（mean KL ~0.004），而 MI50 32GB 上实测没有任何收益。
# 所以这里显式关掉 —— 这是 0.1.41 在 gfx906 上唯一需要改的部署项。想看上游默认行为就设 1。
PREFILL_CPU_SHARE="${PREFILL_CPU_SHARE:-0}"   # 0 = 关（推荐，恢复 0.1.40.3 的逐字节输出）；1/auto = 引擎默认
# 0.1.42 把 STRATA_ROUTE_TAIL_SKIP=7 在 CUDA serve 上默认打开：verify 窗口里每个 token 都只以
# rank>=7 路由的漏掉专家既不拷贝也不算（上游称 8-16GB 卡 decode +15..25%，输出与 0.1.41 略有差别，
# KL ~0.003-0.006，=0 逐字节恢复 0.1.41）。上游本意「HIP and SYCL stay as they were」（关闭），
# 但默认门又写成 #if !defined(STRATA_USE_HIP)（generate.cpp），gfx906 不定义这个宏 —— 和 0.1.41 的
# arm_cpu_share 是同一个 gate bug（那个已在 0.1.42 的 dc0ee3ac 修掉，这个没有）。所以这里显式关掉。
# 想试上游 CUDA 默认行为（可能提 decode）：ROUTE_TAIL_SKIP=7 ./strata-714.sh ...
ROUTE_TAIL_SKIP="${ROUTE_TAIL_SKIP:-0}"       # 0 = 关（推荐，恢复 0.1.41 的逐字节输出）；7/8/9 = 开
# 0.1.43 把 STRATA_ADAPT_OVERLAP 在 Linux-CUDA 上默认打开：自适应专家档的换入拷贝不再在下一个 verify
# 窗口开头等完，而是"边跑边落地"（每层等自己那层的拷贝，AdaptGate）。上游称 tokens 不变、A4000 decode
# +5.5..8.3%。默认门同样是「只有 STRATA_USE_HIP 才关」（generate.cpp 的 adapt_overlap()：HIP 关、
# Windows 关、其余默认开）—— gfx906 不定义 STRATA_USE_HIP，于是**又被误开**：这是同一个 gate 模式的
# 第三次（0.1.41 arm_cpu_share、0.1.42 ROUTE_TAIL_SKIP、0.1.43 ADAPT_OVERLAP）。上游对 HIP 的原话是
# "HIP: not measured yet, opt-in"，本机按 AMD 卡处理 → 显式关掉。
# 本机实测（iq3_s / 32K / 同会话背靠背）：输出逐字节 10/10 相同；decode 开 60.4 vs 关 58.3 中位
# （0.1.42 自己是 60.8）—— 无可测收益。要试上游行为：ADAPT_OVERLAP=1 ./strata-714.sh ...
ADAPT_OVERLAP="${ADAPT_OVERLAP:-0}"           # 0 = 关（推荐，与 0.1.42 的窗口时序一致）；1 = 开
# 0.1.43 新增 STRATA_NAN_GUARD（#879）：每个 verify 窗口多一个只读 kernel 扫 head logits 的非有限值，
# 发现 NaN 就丢掉被污染的会话状态（自愈），不再把 '!' 刷屏。默认开、**无任何编译门**，gfx906 也执行。
# 代价 ~5 µs/窗口，正常路径输出逐字节不变（上游 3 对逐字节验证）。这个守卫对 MI50 有实际价值
# （#1463 的 "!!!!" 症状），所以保持开；要把窗口图恢复成 0.1.42 的形状就设 0。
NAN_GUARD="${NAN_GUARD:-1}"                   # 1 = 开（推荐，#879 自愈守卫）；0 = 关
# ── 0.1.44 本机新增：gfx906 上的两个 prefill 加速开关（都是 opt-in，上游默认关）────────────────
# ① STRATA_HIP_PROMPT_F16=1：prompt 的 16 位 GEMM 改成 FP16 进 / FP16 出（rocBLAS HSS）。
#    默认路径是把 BF16 权重解量化成 FP16 再喂 rocBLAS，而 gfx906 **没有原生 BF16**：rocBLAS 给它的
#    4 个 Type_BB Tensile 库文件名全带 _fallback_，实跑的是 Cijk_..._BSS_BH_MT64x32x8 参考实现
#    （64×32×8 的小 tile），FP16 走的是 MT128x64x16 的大 tile。所以这一步换掉的是 prefill 里
#    最大的一块（旧 profile：BF16 占 GPU 内核时间 21.1%）。
#    代价：每次 16 位 GEMM 的输出多一次 FP16 舍入（权重/激活本身在 FP16 正常范围内，舍入更细：
#    BF16 7 位尾数 → FP16 10 位）。实测 5 题 golden content 5/5 相同；20K 真实文档 5 个埋点
#    全部答对且与关闭时逐字相同。
# ② STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 + STRATA_GFX906_ATTN_REDUCE12=1（上游 PR #1661 / #1718，
#    0FL01 提交，只在 gfx906 构建里编译，默认关）：INT8 KV 的批量 QSA 注意力（**prefill / verify /
#    MTP 都走它** —— gfx906 没有 mma.sync，qsa_prompt_attn_batch 直接 return false，prefill 落回
#    decode 的 attn_chunk_kernel）。QUERY_SWIZZLE 只改 LDS 里 q 的存放顺序（d ^ ((d>>3)&4)，
#    双射且自逆，浮点操作数与运算顺序不变 → 逐字节相同）；REDUCE12 用同样的蝶形配对做 score
#    归约，shuffle 从 60 次降到 16 次（上游原话 "the same butterfly pairs as twelve warp_sum
#    calls"）。两者都不改数值。
#    本机实测（iq3_s / 256K 配置 / 5417 token 提示 / 4096 档，中位）：
#      基线 455.6 → ①564.8（+24.0%）→ ①+SWIZZLE 574.2 → ①+SWIZZLE+REDUCE12 590.4（+29.6%）
#    16K 提示：475.2 → 597.5（+25.7%）；20K 真实文档冷启动：465.9 → 642.9（+38%）。
#    decode 不变（57.4 → 57.2 tok/s，噪声内）。关掉任一：PROMPT_F16=0 / ATTN_SWIZZLE=0 / ATTN_REDUCE12=0
PROMPT_F16="${PROMPT_F16:-1}"                 # 1 = prompt 16 位 GEMM 走 FP16（推荐，+24% prefill）；0 = 上游默认
ATTN_SWIZZLE="${ATTN_SWIZZLE:-1}"             # 1 = gfx906 INT8 批量注意力 query LDS swizzle（+1.7% prefill）
ATTN_REDUCE12="${ATTN_REDUCE12:-1}"           # 1 = 同款归约，shuffle 60→16（再 +2.8% prefill）
# ④ STRATA_GFX906_ATTN_DOT2=1：把上面这个 INT8 批量注意力的两个乘加阶段换成 gfx906 的 V_DOT2_F32_F16
#    （本机新写的 attn_chunk_kernel_gfx906）。gfx906 没有矩阵核，原内核全程 FP32 SIMT；实测（微基准
#    tests/hip/attn_chunk_bench.cpp，ctx 8192 / 2048 query / INT8 池）它的 37.3 ms 里 score 阶段 43%、
#    value 阶段 39%，而**光是 q 的共享内存读就占 41%** —— 瓶颈不是 FMA，是每 lane 每 cell 读 12 头 ×
#    32 字节 FP32 q 的 LDS 流量。新内核把 q 存成 fp16（每头 16 字节，ds_read_b128 无 bank 冲突）、
#    用 4 条 dot2 换 8 条 FMA；value 阶段把**相邻两个 cell** 配成一条 dot2（该指令的累加器是 FP32，
#    两半必须喂同一个输出），12 条 FMA + 12 次 LDS 变成每两 cell 6 条 dot2 + 6 次 LDS。
#    实测 37.33 → 27.92 ms（−25.2%），输出与 FP32 内核 rms 差 0.047%（max|d| 3.6e-3）。
#    ctx 512~32768 五个形状全部 −25%，只有 ctx 64（16 query，总耗时 0.07 ms）略慢，无关紧要。
ATTN_DOT2="${ATTN_DOT2:-1}"                   # 1 = 注意力走 V_DOT2_F32_F16 内核（推荐）；0 = 上游 FP32 内核
# 会话缓存（0.1.39+ 的 --conversation-cache-*）：把整个会话停在**主机内存**里，多个会话来回切换时
# 不用重读 prompt。上游默认 0 = 关。快照大小实测（本机 256K 配置、iq3_s、int8 KV、--spec 4）：
#     462 token → 232 MiB      10,254 → 375 MiB     29,868 → 773 MiB
#     99,335 → 2236 MiB        198,631 → 3684 MiB   外推 262,144 → ~4.6 GiB
#   高段拟合 ≈ 787 MiB + 14.9 KiB/token（边际正好是 KV 的 14.1 KiB/token）。
# 预算能停多长（停满后 MemAvailable = 9221 − 预算；admission 要求 ≥ 快照 + 2560 MiB）：
#     2048 → ~86K token（MemAvailable 7.2 GiB）     4096 → ~227K（5.1 GiB）
#     5120 → ~297K（4.1 GiB）  ← 本脚本默认，覆盖满 256K 上下文
#     5632 → ~332K（3.6 GiB，更紧）                 6656 → 理论天花板（只剩 floor）
# 想更保守用 4096，只跑单会话可以 CONV_CACHE_MIB=0 直接关（默认的 --prompt-cache 6 已在做前缀复用）。
CONV_CACHE_MIB="${CONV_CACHE_MIB:-5120}"          # 会话缓存 RAM 预算 MiB；0 = 关（引擎默认）
CONV_CACHE_SLOTS="${CONV_CACHE_SLOTS:-2}"         # 最多停几个会话（引擎默认 4；预算小就给少点）
CONV_CACHE_MIN_FREE_MIB="${CONV_CACHE_MIN_FREE_MIB:-2560}"  # 停放/恢复时必须留的物理内存（引擎默认 2560）
VISION="${VISION:-0}"              # 0/none = 关图像；cpu = 开图像（CPU 编码器，AMD 唯一选择）
VISION_TOKENS="${VISION_TOKENS:-300}"   # 每图最多 image token（CPU 默认 300；越多越慢，细节越好）
VISION_THREADS="${VISION_THREADS:-3}"   # 编码器线程数（引擎已占 7 个 pool worker，别给满）
API_KEY="${API_KEY:-}"             # 留空则不加鉴权
MODEL_ID="${MODEL_ID:-qwen3.8-flash}"  # 对外暴露的模型 id（/v1/models、WebUI 显示名）
FOREGROUND=1

# ------------------------------- 固定路径 -----------------------------------
# 本脚本在 <repo>/deploy/mi50/ 下。默认布局与作者的机器一致：仓库目录叫 Strata（容器里是
# /work/Strata），模型在它的上级目录的 models/，运行时数据在上级目录的 Strata-data/。
# 换机器时用环境变量覆盖（仓库目录必须仍叫 Strata，脚本里的容器路径 /work/Strata 是写死的）。
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-mixa3607/rocm-gfx906:7.14-complete}"
CTR="${CTR:-strata714}"
HOST_ROOT="${HOST_ROOT:-$(dirname "$REPO_ROOT")}"
MODELS_HOST="${MODELS_HOST:-$HOST_ROOT/models}"
STRATA_HOST="${STRATA_HOST:-$REPO_ROOT}"
DATA_HOST="${DATA_HOST:-$HOST_ROOT/Strata-data}"
# pinned llama.cpp 源（setup.py 的 LLAMA_CPP_COMMIT）。build-714 / pack / 图像编码器都用这一份；
# 作者机器上是 build-906 的 FetchContent 目录，新克隆则用 third_party/llama.cpp。
if [ -z "${GGML_SRC:-}" ]; then
    if [ -d "$STRATA_HOST/third_party/llama.cpp" ]; then
        GGML_SRC="$STRATA_HOST/third_party/llama.cpp"
    else
        GGML_SRC="$STRATA_HOST/build-906/_deps/strata_llamacpp-src"
    fi
fi
VENV="${VENV:-/work/Strata/.venv-714}"

# 容器内路径
C_MODELS="/models"
C_PACKS="/work/Strata-data/packs"
C_PROFILE="/work/Strata/data/expert-profile.bin"
C_MTP="/work/Strata-data/mtp/rt"

# 图像编码器（-v）。文件名与 setup.py 的 MMPROJ 一致；Q8_0 是上游对 CPU 编码器的推荐版本
# （590 MiB vs BF16 865 MiB，编码器里省约 280 MB 内存，300 token 时通常更快），存在就优先用。
MMPROJ_BF16="mmproj-Qwen3.8-Flash-Next-BF16.gguf"
MMPROJ_Q8="mmproj-Qwen3.8-Flash-Next-Q8_0.gguf"
VEXE_HOST="$STRATA_HOST/build-714/strata-vision"   # 由 ./strata-714.sh build-vision 产出

resolve_mmproj() {
    if [ -f "$MODELS_HOST/$MMPROJ_Q8" ]; then
        MMPROJ_NAME="$MMPROJ_Q8"; MMPROJ_KIND="Q8_0"
    else
        MMPROJ_NAME="$MMPROJ_BF16"; MMPROJ_KIND="BF16"
    fi
    C_MMPROJ="$C_MODELS/$MMPROJ_NAME"
    MMPROJ_HOST="$MODELS_HOST/$MMPROJ_NAME"
}

# ------------------------------- 输出工具 -----------------------------------
c_red=$'\033[31m'; c_grn=$'\033[32m'; c_yel=$'\033[33m'; c_dim=$'\033[2m'; c_off=$'\033[0m'
say()  { printf '%s\n' "$*"; }
ok()   { printf '%s✓%s %s\n' "$c_grn" "$c_off" "$*"; }
warn() { printf '%s!%s %s\n' "$c_yel" "$c_off" "$*"; }
die()  { printf '%s✗%s %s\n' "$c_red" "$c_off" "$*" >&2; exit 1; }
dim()  { printf '%s%s%s\n' "$c_dim" "$*" "$c_off"; }

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0; }

# 容器只挂载 HOST_ROOT -> /work；GGML_SRC 在它外面时 pack / build-vision 会拿到错误路径。
case "$GGML_SRC" in
    "$HOST_ROOT"/*) ;;
    *) die "GGML_SRC 必须在 HOST_ROOT 下（容器只挂载 $HOST_ROOT）: $GGML_SRC" ;;
esac

# --------------------------- 量化版本 → 路径映射 -----------------------------
# 用法: quant_info <quant>   →  设置 Q_DIR / Q_TAG / Q_ARENA_GB
quant_info() {
    case "$1" in
        iq2_xs)  Q_TAG="IQ2_XS";  Q_DIR="IQ2_XS";  Q_ARENA_GB=34 ;;
        iq3_xxs) Q_TAG="IQ3_XXS"; Q_DIR="IQ3_XXS"; Q_ARENA_GB=41 ;;
        iq3_s)   Q_TAG="IQ3_S";   Q_DIR="IQ3_S";   Q_ARENA_GB=48 ;;
        *) die "未知量化 '$1'（可选: iq2_xs / iq3_xxs / iq3_s）" ;;
    esac
    Q_PACK="$C_PACKS/$1"
    Q_PACK_HOST="$DATA_HOST/packs/$1"
    Q_MODEL_DIR="$C_MODELS/Qwen3.8-Flash-Next-GSQ-RCO-GGUF-$Q_TAG/$Q_DIR"       # 容器内路径
    Q_MODEL_HOST="$MODELS_HOST/Qwen3.8-Flash-Next-GSQ-RCO-GGUF-$Q_TAG/$Q_DIR"   # 宿主机路径
    Q_SHARD1="$Q_MODEL_DIR/Qwen3.8-Flash-Next-GSQ-RCO-$Q_TAG-00001-of-00002.gguf"
    Q_SHARD2="$Q_MODEL_DIR/Qwen3.8-Flash-Next-GSQ-RCO-$Q_TAG-00002-of-00002.gguf"
    Q_SHARD1_HOST="$Q_MODEL_HOST/Qwen3.8-Flash-Next-GSQ-RCO-$Q_TAG-00001-of-00002.gguf"
    Q_SHARD2_HOST="$Q_MODEL_HOST/Qwen3.8-Flash-Next-GSQ-RCO-$Q_TAG-00002-of-00002.gguf"
}

# ------------------------------- 参数解析 -----------------------------------
ACTION="start"
while [ $# -gt 0 ]; do
    case "$1" in
        start|stop|status|logs|list|pack|bench|build|build-vision) ACTION="$1"; shift ;;
        -q|--quant)   QUANT="$2"; shift 2 ;;
        -c|--ctx)     CTX="$2"; shift 2 ;;
        -k|--kv-resident)
            # 可选数值：`-k 16384` 用给定值；`-k` 单独出现或后面跟别的选项时用默认 32768
            if [ $# -ge 2 ] && [[ "$2" =~ ^[0-9]+$ ]]; then KV_RESIDENT="$2"; shift 2
            else KV_RESIDENT=32768; shift; fi ;;
        -p|--port)    PORT="$2"; shift 2 ;;
        -H|--host)    HOST="$2"; shift 2 ;;
        -t|--decode-timing) DECODE_TIMING=1; shift ;;
        -v|--vision)
            # `-v` 单独出现 = cpu；`-v cpu|none|gpu` 显式指定（gpu 在 AMD 上会被降级为 cpu，见下）
            if [ $# -ge 2 ] && [[ "$2" =~ ^(cpu|gpu|none|no|off|yes|on|0|1)$ ]]; then VISION="$2"; shift 2
            else VISION=cpu; shift; fi ;;
        -d|--daemon)  FOREGROUND=0; shift ;;
        -f|--foreground) FOREGROUND=1; shift ;;
        -h|--help)    usage ;;
        *) die "未知参数 '$1'（-h 看帮助）" ;;
    esac
done

# 上下文简写: 32k / 128k / 256k
case "$CTX" in
    32k|32K)   CTX=32768 ;;
    128k|128K) CTX=131072 ;;
    256k|256K) CTX=262144 ;;
esac
[[ "$CTX" =~ ^[0-9]+$ ]] || die "上下文 '$CTX' 不是数字（可用 32k/128k/256k 或直接写数字）"

# 长上下文自动建议开 KV 流式
if [ "$CTX" -ge 131072 ] && [ "$KV_RESIDENT" = "0" ]; then
    warn "上下文 ${CTX} ≥128K 但未开 --kv-resident，专家缓存会缩水约 13%（用 -k 开启）"
fi

# 图像开关归一化：AMD/HIP 后端没有 GPU 编码器，gpu/yes 只能落到 CPU
case "$VISION" in
    0|""|no|none|off)  VISION=0 ;;
    cpu)               VISION=cpu ;;
    yes|on|1|gpu)
        warn "AMD/HIP 后端没有 GPU 图像编码器（上游 0.1.40.3），--vision gpu 会退化为 CPU 编码器"
        VISION=cpu ;;
    *) die "未知图像模式 '$VISION'（-v / -v cpu / -v none）" ;;
esac
# do_status 也要显示编码器状态，所以无论开不开都解析一次 mmproj 名字
resolve_mmproj

# 对外暴露必须带 API key（Strata 自己的约定：绝不把服务暴露到 127.0.0.1 之外而不设 key）
LOCAL_HOSTS="127.0.0.1 localhost ::1"
IS_LOCAL=1
for h in $LOCAL_HOSTS; do [ "$HOST" = "$h" ] && IS_LOCAL=0; done
if [ "$IS_LOCAL" = "1" ] && [ -z "$API_KEY" ]; then
    die "监听地址 '$HOST' 会把服务暴露到本机之外，必须同时设置 API_KEY，例如：
       API_KEY=\$(openssl rand -hex 16) $0 -H $HOST -q $QUANT -c $((CTX/1024))k"
fi
# 就绪探测/状态/测速用哪个地址：0.0.0.0 是通配地址，不能直接连
CHECK_HOST="$HOST"; [ "$HOST" = "0.0.0.0" ] && CHECK_HOST="127.0.0.1"
CHECK_URL="http://$CHECK_HOST:$PORT"
# 0.0.0.0 时给出局域网地址，方便直接复制
LAN_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' | head -1)"

quant_info "$QUANT"
CFG_NAME="run-$QUANT-ctx$((CTX/1024))k.json"
CFG_HOST="$STRATA_HOST/$CFG_NAME"
LOG_HOST="$DATA_HOST/logs/run-$QUANT-$((CTX/1024))k.log"
CTR_LOG="$DATA_HOST/logs/container-server-$QUANT.log"

# =============================================================================
#  容器管理
# =============================================================================
container_running() {
    [ "$(docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null)" = "true" ]
}

ensure_container() {
    if container_running; then
        local mode
        mode="$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$CTR")"
        [ "$mode" = "host" ] || die "容器 '$CTR' 已存在但不是 host 网络（当前: $mode）。请先 $0 stop 或 docker rm -f $CTR"
        ok "容器 $CTR 已在运行"
        return
    fi
    docker inspect "$CTR" >/dev/null 2>&1 && { dim "移除旧容器 $CTR"; docker rm -f "$CTR" >/dev/null; }

    docker image inspect "$IMAGE" >/dev/null 2>&1 || { say "拉取镜像 $IMAGE ..."; docker pull "$IMAGE"; }

    say "创建容器 $CTR ..."
    docker run -d --name "$CTR" \
        --network host \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video --group-add 110 \
        --ipc=host --ulimit memlock=-1 --shm-size=16g \
        -v "$HOST_ROOT:/work" \
        -v "$MODELS_HOST:/models:ro" \
        "$IMAGE" sleep infinity >/dev/null
    ok "容器已创建（host 网络 / render 组 / memlock 不限）"
    ensure_build_tools
}

# 镜像里没有 cmake/ninja，也没有 g++ 的 C++ 头（ROCm 包只带进来 libstdc++-11-dev）。
# 容器一旦被重建这些就全没了，build-714 会编不动 —— 2026-10-06 就是这样踩到的。
# 注意：cmake 的 Recommends 会顺手带进 gcc-13，而 clang 默认挑最新的 GCC，只装 gcc-13
# 不装 libstdc++-13-dev 的话连 <cstdint>/<cmath> 都找不到。所以三个一起装，
# 并用一次真实编译验证 —— 光查包名查不出这个组合。
ensure_build_tools() {
    if docker exec "$CTR" bash -c '
            command -v cmake >/dev/null && command -v ninja >/dev/null &&
            printf "#include <cstdint>\nint main(){return 0;}\n" > /tmp/.strata_toolcheck.cpp &&
            /opt/rocm/llvm/bin/clang++ -std=c++20 -c /tmp/.strata_toolcheck.cpp -o /tmp/.strata_toolcheck.o' \
            >/dev/null 2>&1; then
        dim "容器内构建工具已就绪（cmake / ninja / libstdc++）"
        return 0
    fi
    say "补齐容器内构建工具（cmake / ninja / libstdc++-13-dev）..."
    # gfx906 的第三方源 s3.arkprojects.space 已 403 失效，apt-get update 会报错；
    # 主源是好的，所以这里不能因为它的退出码中断。
    docker exec "$CTR" bash -c "apt-get update -qq" >/dev/null 2>&1 || true
    docker exec "$CTR" bash -c "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake ninja-build libstdc++-13-dev" >/dev/null 2>&1 \
        && ok "构建工具已安装" || warn "构建工具安装失败（只有重编引擎才需要，可以先不管）"
}

stop_server() {
    docker exec "$CTR" pkill -f serve.server >/dev/null 2>&1 || true
    sleep 2
    docker exec "$CTR" pkill -x strata >/dev/null 2>&1 || true
    sleep 2
}

# =============================================================================
#  前置检查
# =============================================================================
check_prereqs() {
    # 模型文件
    local missing=0
    for f in "$Q_SHARD1_HOST" "$Q_SHARD2_HOST"; do
        [ -f "$f" ] || { warn "缺模型分片: ${f#$MODELS_HOST/}"; missing=1; }
    done
    [ "$missing" = 0 ] || die "模型文件不全。下载到 $Q_MODEL_HOST"

    # pack
    if [ ! -f "$Q_PACK_HOST/experts.bin" ] || [ ! -f "$Q_PACK_HOST/index.txt" ]; then
        die "pack 不存在或不全: $Q_PACK_HOST
     用下面命令建（约 40-50 GB，需要几分钟）:
       $0 pack -q $QUANT"
    fi

    # 图像（-v cpu）：编码器二进制 + mmproj 必须都在，否则引擎起来了也读不了图
    if [ "$VISION" = "cpu" ]; then
        [ -f "$VEXE_HOST" ] || die "缺图像编码器: ${VEXE_HOST#$HOST_ROOT/}
     先编一次（容器内编译，几分钟）:
       $0 build-vision"
        [ -f "$MMPROJ_HOST" ] || die "缺图像编码器权重: ${MMPROJ_HOST#$MODELS_HOST/}
     下载（约 865 MiB）:
       curl -L --fail -C - -o '$MMPROJ_HOST' \\
         'https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/$MMPROJ_BF16'"
        local vis_avail_gb
        vis_avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
        [ "$vis_avail_gb" -lt 4 ] && warn "可用内存 ${vis_avail_gb} GB：编码器常驻还要约 0.7-1.1 GB"
        dim "图像: 开（$MMPROJ_KIND 编码器，CPU $VISION_THREADS 线程，每图最多 $VISION_TOKENS token）"
    fi

    # 内存（arena 常驻 + 约 4 GB 余量）
    local avail_gb
    avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    if [ "$avail_gb" -lt "$Q_ARENA_GB" ]; then
        warn "可用内存 ${avail_gb} GB < arena 约 ${Q_ARENA_GB} GB，可能 OOM"
    fi

    # 先清掉容器内可能残留的引擎 —— 必须排在端口检查之前，否则残留引擎会占着端口让脚本
    # 直接 die（上次异常退出时就会这样），而不是自愈
    #
    # 只算**活着的**进程：容器里的 PID 1 是 `sleep infinity`，它不会回收子进程，所以被 kill 过的
    # 引擎会以僵尸（State: Z）留在进程表里，`pgrep -x strata` 照样匹配得到。只按名字判断的话，
    # 每次启动都会误报"已有引擎在跑"并白等 4 秒（stop_server 里的两次 sleep）。
    # 僵尸不占端口、不占显存、也收不到信号，直接忽略即可 —— 要彻底清掉得重启容器。
    if docker exec "$CTR" sh -c 'ps -eo stat=,comm= | awk "\$1 !~ /^Z/ && \$2 == \"strata\" { found = 1 } END { exit !found }"' 2>/dev/null; then
        warn "容器内已有引擎在跑（上次可能异常退出），先停掉"
        stop_server
    fi

    # 端口
    if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
        die "端口 $PORT 已被占用（换 -p，或先 $0 stop）"
    fi
}

# =============================================================================
#  生成配置
# =============================================================================
write_config() {
    local extra_kv=""
    if [ "$KV_RESIDENT" != "0" ]; then
        extra_kv=", \"--kv-resident\", \"$KV_RESIDENT\""
    fi
    local api=""
    [ -n "$API_KEY" ] && api=", \"api_key\": \"$API_KEY\""
    # 专家缓存自适应档：默认不写 --adapt-every，用引擎自己的自适应（每 4 窗口换 96 个专家）。
    # ADAPT_EVERY=0 会退回静态驻留 —— 那是 AMD 文档给 7900 XTX 的配置，在 MI50 上 decode 掉 25-53%
    # （见文件头"设计要点"）。这里写成条件参数，方便随时用环境变量做 A/B。
    local adapt_arg=""
    [ -n "$ADAPT_EVERY" ] && adapt_arg=", \"--adapt-every\", \"$ADAPT_EVERY\""
    # 引擎的探针环境变量只能走配置的 "env" 块：docker exec 不继承宿主 shell 的环境，server 是用
    # child_env(cfg) 把这里的键并进引擎环境的（-t / DECODE_TIMING=1 打开）。
    # 前四个 STRATA_* 是 gfx906 上曾经被上游默认误开、现在由本脚本锁定的行为：
    #   PREFILL_CPU_SHARE=0   0.1.41 误开的 CPU 分担（上游 PR #1732 的 gate bug）
    #   ROUTE_TAIL_SKIP=0     0.1.42 误开的尾部专家跳过（改输出）
    #   ADAPT_OVERLAP=0       0.1.43 误开的自适应拷贝重叠（0.1.43.1 起上游已改为全后端 opt-in；
    #                         保留 =0 只为固定窗口时序，不改输出）
    #   NAN_GUARD=1           0.1.43 的 #879 非有限 logits 守卫，无门、gfx906 默认执行；显式写出来防回归
    # 后面三个是 0.1.44 本机新增的 prefill 加速档（都是上游 opt-in、默认关；见文件头"设计要点"）：
    #   PROMPT_F16=1          prompt 的 16 位 GEMM 走 FP16（gfx906 无原生 BF16，rocBLAS 走 _fallback_ 小 tile）
    #   ATTN_SWIZZLE=1        INT8 批量 QSA 注意力的 query LDS swizzle（PR #1661；只改布局，逐字节相同）
    #   ATTN_REDUCE12=1       同款 score 归约 60→16 次 shuffle（PR #1718；同样的蝶形配对）
    # 0.1.45 新增、本机实测过但没有写进 env 的默认/开关（写出来只供下次更新对照）：
    #   STRATA_COMMIT_BATCH=1 上游默认开：commit 图从每层两次 launch 合成两次；本机 256K decode 开关两臂
    #                         62.4/65.6 vs 64.1/63.3/66.6 tok/s，在 ±7% 噪声内（逐字节相同），保持默认
    #   STRATA_FILE_CACHE_OLD_RULE=1  gfx906 构建里默认开（#1752）：文件层只在"剩余空间 ≥ 要读的字节数"
    #                         时用 page cache，避免 MI50 上内核回收把 HMM arena 打散（#1691）。本机 32GB
    #                         arena 全驻留，走不到这段；16GB MI50 / 专家从文件读的配置才是受益者
    #   STRATA_ATTN_LANECELL=1 / STRATA_ATTN_MERGE_V2=1 上游 opt-in；本机实测 63.2–66.4 / 65.5–67.0 tok/s，
    #                         与默认 63.3–66.6 同噪声带，故不启用
    local env_pairs="\"STRATA_PREFILL_CPU_SHARE\": \"$PREFILL_CPU_SHARE\", \"STRATA_ROUTE_TAIL_SKIP\": \"$ROUTE_TAIL_SKIP\", \"STRATA_ADAPT_OVERLAP\": \"$ADAPT_OVERLAP\", \"STRATA_NAN_GUARD\": \"$NAN_GUARD\", \"STRATA_HIP_PROMPT_F16\": \"$PROMPT_F16\", \"STRATA_GFX906_ATTN_QUERY_SWIZZLE\": \"$ATTN_SWIZZLE\", \"STRATA_GFX906_ATTN_REDUCE12\": \"$ATTN_REDUCE12\", \"STRATA_GFX906_ATTN_DOT2\": \"$ATTN_DOT2\""
    [ "$DECODE_TIMING" != "0" ] && env_pairs="$env_pairs, \"STRATA_DECODE_TIMING\": \"1\", \"STRATA_VERIFY_PROFILE\": \"1\""
    # EXTRA_ENV="KEY=VAL,KEY2=VAL2"：往引擎的 env 块里再塞任意变量（A/B 探针用，如
    # STRATA_GFX906_ATTN_QUERY_SWIZZLE=1）；同名键以 EXTRA_ENV 为准（后写覆盖前写）。
    if [ -n "${EXTRA_ENV:-}" ]; then
        IFS=',' read -ra _XP <<< "$EXTRA_ENV"
        for _kv in "${_XP[@]}"; do
            [ -z "$_kv" ] && continue
            case "$_kv" in
                *=*) env_pairs="$env_pairs, \"${_kv%%=*}\": \"${_kv#*=}\"" ;;
                *) warn "EXTRA_ENV 忽略无 = 的项: $_kv" ;;
            esac
        done
    fi
    # EXTRA_ARGS="--adapt-min-gain 3"：往引擎 args 末尾追加任意参数（A/B 探针用）。按空白切分，
    # 值里不能有空格；只加参数，不覆盖上面任何具名参数（同名重复时引擎取后出现的那个）。
    local extra_args=""
    if [ -n "${EXTRA_ARGS:-}" ]; then
        for _a in $EXTRA_ARGS; do
            extra_args="$extra_args, \"$_a\""
        done
    fi
    local env_block=",
  \"env\": {$env_pairs}"

    # 图像（-v cpu）：引擎加 --vision（GENI 请求 + M-RoPE 位置表），server 再按 "vision" 段
    # 拉起 build-714/strata-vision。CPU 编码器不占显存，所以不强制改 --vram-reserve-mib
    # （setup.py 会写 700；你这里的 VRAM_RESERVE_MIB 原样保留，想还显存给专家缓存可以调小）。
    local vision_arg="" vision_block=""
    if [ "$VISION" = "cpu" ]; then
        vision_arg=", \"--vision\""
        vision_block=$(printf ',\n  "vision": {\n    "exe": "build-714/strata-vision",\n    "mmproj": "%s",\n    "model": "%s",\n    "gpu": false,\n    "max_tokens": %s,\n    "threads": %s\n  }' \
            "$C_MMPROJ" "$Q_SHARD1" "$VISION_TOKENS" "$VISION_THREADS")
    fi

    # 会话缓存：CONV_CACHE_MIB=0 时完全不写这三个参数（保持引擎默认的关闭状态）。
    local conv_arg=""
    if [ "$CONV_CACHE_MIB" != "0" ]; then
        conv_arg=", \"--conversation-cache-mib\", \"$CONV_CACHE_MIB\", \"--conversation-cache-slots\", \"$CONV_CACHE_SLOTS\", \"--conversation-cache-min-free-mib\", \"$CONV_CACHE_MIN_FREE_MIB\""
    fi

    # "backend": "hip" 不是可选项：server 靠它决定读 amdgpu sysfs 而不是 NVML（A 卡上没有 NVML），
    # 少了它 WebUI 的 GPU load / VRAM / 温度 / 功耗全是 "–"。setup.py 写的配置同样带这个键。
    cat > "$CFG_HOST" <<JSON
{
  "exe": "build-714/strata",
  "args": [
    "--pack", "$Q_PACK",
    "--native", "$Q_SHARD1",
    "--ple-gguf", "$Q_SHARD2",
    "--expert-profile", "$C_PROFILE",
    "--expert-cache", "auto",
    "--prefill", "$PREFILL",
    "--spec", "$SPEC",
    "--spec-min-p", "$SPEC_MIN_P",
    "--mtp", "$C_MTP",
    "--max-context", "$CTX",
    "--kv", "int8"$extra_kv,
    "--pool-workers", "$POOL_WORKERS"$adapt_arg,
    "--pcie-frac", "0",
    "--vram-reserve-mib", "$VRAM_RESERVE_MIB"$vision_arg$conv_arg$extra_args
  ],
  "cwd": "/work/Strata",
  "tokenizer": "$Q_PACK/tokenizer",
  "model_name": "$MODEL_ID",
  "log": "/work/Strata-data/logs/run-$QUANT-$((CTX/1024))k.log",
  "host": "$HOST",
  "port": $PORT,
  "backend": "hip"$api$env_block$vision_block
}
JSON
    ok "配置已写入 Strata/$CFG_NAME"
    dim "  --max-context $CTX | --prefill $PREFILL | --kv int8$([ "$KV_RESIDENT" != 0 ] && echo " --kv-resident $KV_RESIDENT")"
    if [ "$CONV_CACHE_MIB" != "0" ]; then
        dim "  会话缓存: ${CONV_CACHE_MIB} MiB / $CONV_CACHE_SLOTS 槽（floor ${CONV_CACHE_MIN_FREE_MIB} MiB；关掉用 CONV_CACHE_MIB=0）"
    else
        dim "  会话缓存: 关（要开用 CONV_CACHE_MIB=5120）"
    fi
    if [ "$VISION" = "cpu" ]; then
        dim "  --vision（CPU 编码器 $MMPROJ_KIND，$VISION_THREADS 线程，≤$VISION_TOKENS token/图）"
    else
        dim "  图像: 关（要开用 -v）"
    fi
    dim "  （注意：没有 --mmap-experts，这是刻意的）"
    dim "  专家缓存: $([ -n "$ADAPT_EVERY" ] && echo "静态驻留（--adapt-every $ADAPT_EVERY）" || echo "引擎默认自适应档")"
    # 注意：这里必须用 if，不能用 `[ ... ] && dim` —— write_config 的返回码就是最后一条命令的，
    # 而脚本开了 set -e（:129），EXTRA_ARGS 没设时返回 1 会让 do_start 在 write_config 之后直接退出。
    if [ -n "${EXTRA_ARGS:-}" ]; then
        dim "  追加引擎参数: $EXTRA_ARGS"
    fi
}

# =============================================================================
#  启动 / 停止 / 状态
# =============================================================================
# 访问地址 + API Key。前台和后台都要打印（前台曾经漏掉，导致看不到 key）
print_endpoints() {
    say "  OpenAI 接口 : $CHECK_URL/v1"
    say "  网页界面    : $CHECK_URL/"
    if [ "$IS_LOCAL" = "1" ]; then
        [ -n "$LAN_IP" ] && say "  局域网访问  : http://$LAN_IP:$PORT/   ${c_yel}(已开鉴权)${c_off}"
    else
        dim "  仅监听本机（要对外用 -H 0.0.0.0，并设置 API_KEY）"
    fi
    if [ -n "$API_KEY" ]; then
        say "  API Key     : ${c_grn}$API_KEY${c_off}    ${c_dim}客户端用 Authorization: Bearer $API_KEY${c_off}"
    fi
    say "  容器日志    : ${CTR_LOG#$HOST_ROOT/}"
    say "  引擎日志    : ${LOG_HOST#$HOST_ROOT/}"
}

do_start() {
    ensure_container
    check_prereqs
    write_config
    stop_server

    mkdir -p "$DATA_HOST/logs"
    say ""
    say "启动 Strata —— 量化 ${c_grn}$QUANT${c_off}  上下文 ${c_grn}$((CTX/1024))K${c_off}  端口 ${c_grn}$PORT${c_off}"
    if [ "$VISION" = "cpu" ]; then
        dim "  图像: 开（CPU 编码器 $MMPROJ_KIND / $VISION_THREADS 线程 / ≤$VISION_TOKENS token 每图；首图前无预热）"
    else
        dim "  图像: 关（要开用 -v；AMD 只能走 CPU 编码器）"
    fi
    dim "  首次启动要加载专家 arena（$Q_ARENA_GB GB），约 20-60 秒；模型就绪前别关窗口"

    local cmd=(docker exec -u 1000:1000 -e HOME=/tmp -w /work/Strata "$CTR"
               bash -c "export ROCM_PATH=/opt/rocm; exec $VENV/bin/python -m serve.server --engine strata --config $CFG_NAME --port $PORT")

    if [ "$FOREGROUND" = "1" ]; then
        # 兜底：无论脚本因何退出（Ctrl-C / kill / 异常），都清掉容器内的引擎 —— 杀掉 docker exec
        # 本身不会杀掉容器里的进程，不清理就会留下占着端口和显存的孤儿引擎
        trap 'stop_server' EXIT
        trap 'say ""; say "正在停止..."; exit 0' INT TERM
        say ""
        say "前台运行中 —— 就绪后即可使用下面的地址（Ctrl-C 停止）："
        print_endpoints
        say ""
        "${cmd[@]}" 2>&1 | tee "$CTR_LOG"
        stop_server
    else
        nohup "${cmd[@]}" > "$CTR_LOG" 2>&1 &
        local pid=$!
        say "后台启动中 (pid $pid)，等待就绪 ..."
        local t=0
        while [ $t -lt 180 ]; do
            if curl -fsS -m 2 -o /dev/null "$CHECK_URL/" 2>/dev/null; then
                ok "已就绪"
                print_endpoints
                say ""
                say "  停止: $0 stop"
                return 0
            fi
            sleep 3; t=$((t+3))
        done
        warn "180 秒内未就绪，看日志: tail -f ${CTR_LOG#$HOST_ROOT/}"
        return 1
    fi
}

do_stop() {
    container_running || { dim "容器未运行"; return 0; }
    stop_server
    ok "服务已停止（容器 $CTR 保留）"
    rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{printf "  显存已用: %.2f GB\n", $NF/1073741824}'
}

do_status() {
    if container_running; then ok "容器 $CTR: 运行中"; else warn "容器 $CTR: 未运行"; fi
    if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
        local bind
        bind="$(ss -tln 2>/dev/null | awk -v p=":$PORT" '$4 ~ p {print $4}' | head -1)"
        ok "端口 $PORT: 监听中  ${c_dim}($bind)${c_off}"
        [ "$IS_LOCAL" = "1" ] && [ -n "$LAN_IP" ] && dim "  局域网: http://$LAN_IP:$PORT/"
        curl -fsS -m 3 "$CHECK_URL/v1/models" 2>/dev/null | head -c 300 || true; say ""
    else
        dim "端口 $PORT: 未监听"
    fi
    docker exec "$CTR" pgrep -a -x strata 2>/dev/null | head -1 || dim "容器内无引擎进程"
    if [ -f "$VEXE_HOST" ]; then
        if [ -f "$MMPROJ_HOST" ]; then
            dim "  图像编码器: build-714/strata-vision ✓ + $MMPROJ_NAME"
        else
            dim "  图像编码器: build-714/strata-vision ✓，但缺 $MMPROJ_NAME"
        fi
    else
        dim "  图像编码器: 未编译（$0 build-vision；开图像需要它）"
    fi
    say ""
    # rocm-smi 行形如: GPU[0] : VRAM Total Memory (B): 34342961152
    rocm-smi --showmeminfo vram 2>/dev/null | awk '
        /Total Memory/{printf "  显存总量: %.2f GB\n", $NF/1073741824}
        /Used Memory/ {printf "  显存已用: %.2f GB\n", $NF/1073741824}'
    awk '/MemTotal|MemAvailable/{printf "  %s %.1f GB\n", ($1=="MemTotal:"?"内存总量:":"内存可用:"), $2/1048576}' /proc/meminfo
}

do_logs() {
    # logs         → 最新的容器/服务端日志
    # logs engine  → 最新的引擎日志
    local which="${1:-container}" f
    case "$which" in
        engine|run) f="$(ls -t "$DATA_HOST"/logs/run-*.log 2>/dev/null | head -1)" ;;
        *)          f="$(ls -t "$DATA_HOST"/logs/container-server-*.log 2>/dev/null | head -1)" ;;
    esac
    [ -n "$f" ] && [ -f "$f" ] || die "找不到日志。可用: ls ${DATA_HOST#$HOST_ROOT/}/logs/"
    say "== ${f#$HOST_ROOT/} =="
    tail -n 60 -f "$f"
}

# =============================================================================
#  list / pack / bench
# =============================================================================
do_list() {
    printf '%-10s %-12s %-10s %-8s %s\n' "量化" "专家集" "显存覆盖" "arena" "pack"
    printf '%s\n' "---------------------------------------------------------------"
    for q in iq2_xs iq3_xxs iq3_s; do
        quant_info "$q"
        local eb="$Q_PACK_HOST/experts.bin" size="-"
        [ -f "$eb" ] && size="$(awk -v b="$(stat -c%s "$eb")" 'BEGIN{printf "%.1f GiB", b/1073741824}')"
        local slots="-"
        case "$q" in iq2_xs) slots="79%" ;; iq3_xxs) slots="64%" ;; iq3_s) slots="54%" ;; esac
        local mark="✗ 缺"
        [ -f "$Q_PACK_HOST/index.txt" ] && mark="✓"
        printf '%-10s %-12s %-10s %-8s %s %s\n' "$q" "$size" "$slots" "${Q_ARENA_GB}G" "$mark" "${Q_PACK_HOST#$HOST_ROOT/}"
    done
    say ""
    dim "  显存覆盖率 = 26 GiB 专家缓存能装下该量化专家集的比例（决定 decode 命中率）"
    dim "  实测 decode（短上下文）: iq2_xs 60.3 / iq3_xxs 52.8 / iq3_s 46.2 tok/s"
}

do_pack() {
    ensure_container
    [ -f "$Q_SHARD1_HOST" ] || die "模型分片不存在: $Q_SHARD1_HOST"
    say "建 pack: $QUANT → ${Q_PACK_HOST#$HOST_ROOT/}"
    dim "  写 experts.bin（约 $Q_ARENA_GB GB），几分钟"
    docker exec -u 1000:1000 -e HOME=/tmp -w /work/Strata "$CTR" bash -c "
        export STRATA_GGUF_PY=/work${GGML_SRC#$HOST_ROOT}/gguf-py
        mkdir -p '$Q_PACK'
        exec $VENV/bin/python tools/iq_pack.py --gguf '$Q_SHARD1' --out '$Q_PACK' --experts-bin
    "
    ok "pack 完成"
    du -sh "$Q_PACK_HOST"
}

do_bench() {
    local url="$CHECK_URL"
    curl -fsS -m 3 -o /dev/null "$url/" 2>/dev/null || die "服务未就绪（先 $0 start）"
    # 测速客户端跑在宿主上（不走容器），优先用本分支的 .venv-714，退回 setup.py 的 .venv
    local py="$STRATA_HOST/.venv-714/bin/python"
    [ -x "$py" ] || py="$STRATA_HOST/.venv/bin/python"
    say "== 短上下文基准 ($QUANT) =="
    BENCH_OUT_DIR="$HOST_ROOT" "$py" "$STRATA_HOST/deploy/mi50/bench-strata.py" "$url" "run-$QUANT" 2>&1 | tail -12
    say ""
    say "== 固定 prompt decode =="
    BENCH_OUT_DIR="$HOST_ROOT" "$py" "$STRATA_HOST/deploy/mi50/bench-decode-fixed.py" "$url" "run-$QUANT-fixed" 2>&1 | tail -8
}

# 重编容器内的引擎。改过 Strata/src/ 或 Strata/include/ 之后必须跑这一步，
# 否则 start 起来的还是旧的 build-714/strata。
# 注意 build-714 用 $GGML_SRC 作 ggml 源（见 build-714/CMakeCache.txt 的 STRATA_GGML_DIR），
# 所以那个目录必须先存在（third_party/llama.cpp，或作者机器上的 build-906 FetchContent）。
do_build() {
    ensure_container
    [ -d "$STRATA_HOST/build-714" ] || die "缺 $STRATA_HOST/build-714（先用宿主 cmake 配置一次，见 docs/MI50.md）"
    [ -d "$GGML_SRC" ] \
        || die "缺 pinned llama.cpp 源: $GGML_SRC（先按 docs/MI50.md 准备）"
    say "重编 build-714（容器内 ROCm 7.14 / gfx906）..."
    dim "  改过 src/ 或 include/ 之后才需要；几分钟"
    docker exec -u 1000:1000 -e HOME=/tmp -w /work/Strata "$CTR" bash -c '
        export ROCM_PATH=/opt/rocm
        exec cmake --build build-714 --target strata -j "$(nproc)"
    ' || die "编译失败（看上面的 error）"
    ok "build-714/strata 已更新"
    local ver
    ver="$(docker exec -u 1000:1000 -e HOME=/tmp -w /work/Strata "$CTR" \
            bash -c "strings build-714/strata | grep -m1 -E '^0\.1\.[0-9]+'" 2>/dev/null || true)"
    [ -n "$ver" ] && dim "  引擎版本: $ver"
    dim "  改完记得重启: $0 stop && $0 -q $QUANT -c $((CTX/1024))k start"
}

# =============================================================================
#  编 CPU 图像编码器（tools/vision：llama.cpp 的 mtmd）
# =============================================================================
# 源码用 $GGML_SRC 的 pinned llama.cpp（3cf03257）——和引擎的 ggml 是同一份，不需要额外下载。
# 容器里没有 g++，用 ROCm 自带的 clang++（纯 CPU 构建，不链 ROCm）。
# -j 4 是刻意的：引擎在跑时给满 16 线程会把 CPU 专家池挤爆。
do_build_vision() {
    ensure_container
    local llama="$GGML_SRC"
    [ -d "$llama/tools/mtmd" ] || die "缺 pinned llama.cpp 源: $llama/tools/mtmd（见 docs/MI50.md）"
    say "编 CPU 图像编码器 strata-vision（容器内 clang++，几分钟）..."
    local llama_c="/work${llama#$HOST_ROOT}"
    docker exec -u 1000:1000 -e HOME=/tmp -w /work/Strata "$CTR" bash -c "
        set -e
        export ROCM_PATH=/opt/rocm
        cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DLLAMA_DIR=$llama_c \
          -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=OFF \
          -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
          -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ >/tmp/vision-cmake.log 2>&1 || { tail -20 /tmp/vision-cmake.log; exit 1; }
        cmake --build build-vision --target strata-vision -j 4 || exit 1
        cp build-vision/bin/strata-vision build-714/strata-vision
    " || die "编译失败（看上面的 error）"
    ok "build-714/strata-vision 已就绪"
    [ -f "$MMPROJ_HOST" ] || warn "还缺 $MMPROJ_NAME（下载命令见 $0 -v 的报错提示）"
    dim "  开图像: $0 -v -q $QUANT -c $((CTX/1024))k${KV_RESIDENT:+ -k}"
}

# =============================================================================
case "$ACTION" in
    start)  do_start ;;
    stop)   do_stop ;;
    status) do_status ;;
    logs)   do_logs ;;
    list)   do_list ;;
    pack)   do_pack ;;
    bench)  do_bench ;;
    build)  do_build ;;
    build-vision) do_build_vision ;;
esac
