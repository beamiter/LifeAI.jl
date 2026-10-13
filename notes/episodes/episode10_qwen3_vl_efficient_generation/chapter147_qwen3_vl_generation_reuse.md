# Chapter 147 — Qwen3-VL 显式复用高层生成 cache

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-13
>
> 关闭：2026-10-13

## 接口

Chapter 146 取消了不同请求对可变 CUDA workspace 的隐式共享。本章让调用方
显式保留同一个 `Qwen3VLStaticKVCache`，并将它作为高层生成接口的 `cache`
参数传回。token 输入接口和原始图文聊天接口都支持这一方式，现有
`cache=:dynamic` / `cache=:static` 行为保留。

```julia
state = init_qwen3_vl_static_kv_cache(parameters; capacity=256)
result = generate_hf_qwen3_vl_tokens(
    parameters, input_ids, rope_layout;
    vision_features, cache=state, max_new_tokens=32,
)
@assert result.cache === state
@assert result.cache_mode === :static
```

下一次请求可以使用同一 `state`。它的 capacity 必须覆盖本次 prompt 与实际
处理的生成前缀；不能同时指定 `static_capacity`。返回的 cache 是同一个可变
对象，调用方应顺序使用，不能把同一对象交给并发请求。

## 失败边界

参数、cache 几何/dtype/device、容量、token、mRoPE 和视觉输入契约通过后，
才 reset 并开始新请求。原始图文入口还要先完成图像处理和视觉前向。
这些准备阶段拒绝请求时，原有 K/V、position、rope_delta、Ready、workspace
和校验 bookkeeping 保留。验证用临时 metadata wrapper，不复制 K/V。

成功的零 token 请求也会把 supplied cache reset 为空；一 token 或 EOS
提前停止请求仍遵守原有物理 cache 长度语义。不同参数的后续请求继续使用
Chapter 145 的重新绑定规则。

## 验证

CPU Float32/BF16 定向回归共 `421` 项通过：显式复用 `142` 项、无效请求保留
`176` 项、损坏 cache 校验 `48` 项、原始图文聊天失败边界 `55` 项。默认测试
不依赖真实模型；图文边界用例带有冻结的官方模板，SHA256 为
`3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4`。

RTX 4090 D 脚本使用冻结的真实 2B checkpoint，比较
image → image → text → image 的生成 token、逐步 logits 与有效 K/V。
32-token 图像请求复用前后完全一致，8-token 短文本请求与独立 fresh cache
完全一致；切回图像请求后仍与首次请求一致。Ready、workspace 和全部 K/V
object/device-pointer identity 保持，一个独立 text cache 同时存活且存储
地址范围无重叠。图像请求 position 为 `107`，短文本请求 position 为 `11`。

[真实 GPU 验证报告](../../../benchmark_results/qwen3_vl_generation_reuse/summary.json)
SHA256 为
`40643c36feb9d24deca3824bc5ec0866473935241c2041f18aee88f6df4fc49f`，
包含模型/参考资产和源码哈希。本章验证请求复用的 exact 行为，不新增 HF
跨 dtype strict parity 或延迟改善结论。Episode 10 保持 Open。
